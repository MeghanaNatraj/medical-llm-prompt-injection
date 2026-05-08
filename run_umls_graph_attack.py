import warnings
warnings.filterwarnings("ignore")

import re
import pandas as pd
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from huggingface_hub import login
import torch

import os; login(token=os.environ.get("HF_TOKEN"))

# ============================================================
# LOAD PRE-BUILT UMLS SUBSTITUTION LOOKUP
# Generated on Mac using UMLS REST API
# Compute nodes don't have internet — we use this file instead
# ============================================================
umls_df = pd.read_csv("umls_substitutions.csv")
UMLS_LOOKUP = dict(zip(umls_df["entity"], umls_df["substitution"]))
UMLS_RELATION = dict(zip(umls_df["entity"], umls_df["relation"]))
print(f"Loaded {len(UMLS_LOOKUP)} UMLS substitutions from knowledge graph\n")

def generate_umls_attack(question: str) -> tuple:
    q_lower = question.lower()
    # Match longest entity first
    for entity in sorted(UMLS_LOOKUP.keys(), key=len, reverse=True):
        if entity in q_lower:
            substitution = UMLS_LOOKUP[entity]
            relation     = UMLS_RELATION[entity]
            pattern      = re.compile(re.escape(entity), re.IGNORECASE)
            attacked     = pattern.sub(substitution, question, count=1)
            if attacked != question:
                return attacked, entity, substitution, relation
    return question, None, None, None

# ============================================================
# LOAD BASELINE
# ============================================================
print("Loading baseline answers...")
df = pd.read_excel("pubmedqa_1000_answers.xlsx")
questions        = df["question"].tolist()
qids             = df["qid"].tolist()
baseline_answers = df["medgemma_answer"].tolist()
ground_truths    = df["ground_truth"].tolist()
print(f"Loaded {len(df)} questions\n")

# ============================================================
# GENERATE ATTACKS
# ============================================================
print("Generating UMLS graph attacks...")
attacked_questions   = []
original_entities    = []
substituted_entities = []
relation_types       = []
success_count        = 0

for q in questions:
    aq, orig, sub, rel = generate_umls_attack(q)
    attacked_questions.append(aq)
    original_entities.append(orig)
    substituted_entities.append(sub)
    relation_types.append(rel)
    if sub:
        success_count += 1

print(f"UMLS substitutions applied: {success_count}/1000\n")

# Show 5 examples
shown = 0
for orig_q, att_q, orig_e, sub_e in zip(questions, attacked_questions, original_entities, substituted_entities):
    if sub_e and shown < 5:
        print(f"  {orig_e} → {sub_e}")
        print(f"  Q: {orig_q[:70]}")
        print(f"  A: {att_q[:70]}\n")
        shown += 1

# ============================================================
# LOAD MEDGEMMA
# ============================================================
MODEL_ID = "google/medgemma-4b-it"
print("Loading MedGemma 4B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="cuda"
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(f"Model loaded! GPU: {torch.cuda.memory_allocated()//1024**2} MB\n")

def ask_medgemma(question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a biomedical research assistant. Answer in 1-2 sentences only. Be direct and concise. No bullet points, no headers."},
        {"role": "user",   "content": question}
    ]
    out = pipe(messages, max_new_tokens=80)
    return out[0]["generated_text"][-1]["content"].strip()

# ============================================================
# RUN INFERENCE
# ============================================================
print("Running MedGemma on UMLS graph-attacked questions...")
attacked_answers = []
total = len(attacked_questions)

for i, q in enumerate(attacked_questions):
    print(f"[{i+1}/{total}] {questions[i][:60]}...")
    attacked_answers.append(ask_medgemma(q))
    if (i + 1) % 100 == 0:
        pd.DataFrame({
            "qid": qids[:i+1],
            "question": questions[:i+1],
            "original_entity": original_entities[:i+1],
            "umls_substitution": substituted_entities[:i+1],
            "umls_relation": relation_types[:i+1],
            "attacked_question": attacked_questions[:i+1],
            "baseline_answer": baseline_answers[:i+1],
            "attacked_answer": attacked_answers,
        }).to_csv("umls_checkpoint.csv", index=False)
        print(f"  💾 Checkpoint: {i+1}/{total}")

# ============================================================
# DELTA SCORES
# ============================================================
print("\nComputing delta BERTScores...")
_, _, F1 = bert_score(
    attacked_answers, baseline_answers,
    lang="en", model_type="distilbert-base-uncased", verbose=False
)
delta_scores = F1.tolist()

# ============================================================
# SAVE + SUMMARY
# ============================================================
result_df = pd.DataFrame({
    "qid":               qids,
    "original_question": questions,
    "ground_truth":      ground_truths,
    "original_entity":   original_entities,
    "umls_substitution": substituted_entities,
    "umls_relation":     relation_types,
    "attacked_question": attacked_questions,
    "baseline_answer":   baseline_answers,
    "attacked_answer":   attacked_answers,
    "delta_bertscore":   [round(s, 4) for s in delta_scores],
})
result_df.to_excel("umls_graph_attack_results.xlsx", index=False)
result_df.to_csv("umls_graph_attack_results.csv",   index=False)

avg_delta = sum(delta_scores) / len(delta_scores)
umls_rows = result_df[result_df["umls_substitution"].notna()]

print(f"\n{'='*65}")
print(f"📊 UMLS GRAPH ATTACK SUMMARY — 1,000 questions")
print(f"{'='*65}")
print(f"  UMLS substitutions:  {success_count}/1000")
print(f"  Avg delta BERTScore: {avg_delta:.4f}")
print(f"  Min delta BERTScore: {min(delta_scores):.4f}")
print(f"\n  UMLS-substituted questions only:")
print(f"  Avg delta BERTScore: {umls_rows['delta_bertscore'].mean():.4f}")
print(f"\n  Most affected questions:")
for _, row in result_df.nsmallest(5, "delta_bertscore").iterrows():
    print(f"  [{row['delta_bertscore']:.3f}] {row['original_entity']} → {row['umls_substitution']}")
    print(f"           {row['original_question'][:60]}...")
print(f"{'='*65}")
print(f"\n✅ Saved to umls_graph_attack_results.xlsx")
