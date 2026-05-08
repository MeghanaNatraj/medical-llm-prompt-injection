# Prompt Injection Attacks on Medical LLMs

A research project investigating the vulnerability of medical large language models to prompt injection attacks, with a novel graph-based detection approach using UMLS knowledge graphs.

**Course:** CS 598 – Security & Privacy in ML  
**University:** University of Illinois Chicago (UIC)  
**Author:** Meghana Nataraju

---

## Overview

Medical LLMs are increasingly deployed in clinical decision support, yet their robustness against adversarial manipulation remains understudied. This project evaluates **8 prompt injection attack types** across **4 medical LLMs** and **2 biomedical QA datasets**, and proposes a **Graph Attention Network (GAT)-based detector** that leverages UMLS biomedical knowledge graph structure to identify attacked queries.

---

## Models Evaluated

| Model | Type | Parameters |
|---|---|---|
| MedGemma 4B | Medical fine-tuned | 4B |
| BioMistral 7B | Medical fine-tuned | 7B |
| Mistral 7B Instruct v0.2 | General purpose | 7B |
| MedAlpaca 7B | Medical fine-tuned | 7B |

---

## Datasets

- **PubMedQA** — 1,000 open-ended free-text biomedical research Q&A
- **MedQA USMLE** — 1,273 multiple-choice clinical exam questions (A/B/C/D)

---

## Attack Types

| Attack | Description |
|---|---|
| Typo/Abbreviation | Introduce medical abbreviations and typos |
| Rephrasing | Semantically equivalent rewrites |
| Meaning Change | Subtle negation and factual flips |
| Brand Manipulation | Replace generic drug names with fictional brands |
| Task Switch | Redirect model to non-medical tasks |
| Medical Hijack | Inject false clinical information |
| Stealth | Hidden instruction injection |
| UMLS Graph | Substitute entities via UMLS RB (related broader) edges |

---

## Key Results

### Attack Effectiveness (BERTScore Delta, PubMedQA)

| Attack | MedGemma | BioMistral | Mistral | MedAlpaca |
|---|---|---|---|---|
| Brand Manipulation | 0.8795 | 0.8801 | 0.8617 | 0.9310 |
| Task Switch | 0.8067 | 0.7897 | 0.8337 | 0.8383 |
| UMLS Graph | 0.9049 | 0.9631 | 0.8942 | 0.9746 |
| **Average** | **0.8929** | **0.9305** | **0.8843** | **0.9533** |

### Brand Attack Success Rate

| Model | PubMedQA | MedQA USMLE |
|---|---|---|
| MedGemma | 76.4% | 0.3% |
| BioMistral | 70.1% | 1.3% |
| Mistral | 75.5% | 5.7% |
| MedAlpaca | 24.9% | 0.0% |

### Detectability

- **7 out of 8 attacks** are undetectable by simple output monitoring
- UMLS graph attack: **0.6% detection rate**
- Medical hijack and stealth: **< 0.1% detection rate**

### GNN Detector Performance

| Version | Entity Coverage | ROC-AUC | Accuracy | Clean F1 | Attacked F1 |
|---|---|---|---|---|---|
| v1 | 40 entities | 0.616 ± 0.027 | 0.620 | 0.71 | 0.44 |
| v2 | 208 entities | **0.631 ± 0.038** | **0.688** | **0.79** | 0.43 |

---

## GNN Detector Architecture

```
Medical Query
     ↓
Entity Extraction (scispaCy NER)
     ↓
UMLS Subgraph Construction G_q = (V_q, E_q)
     ↓
Featurization
  • Node: semantic type one-hot (33 types) + entity length = 34-dim
  • Edge: relation type one-hot (RO/RB/RN/SY/RQ/PAR/CHD/OTHER) = 8-dim
     ↓
3-Layer Graph Attention Network
  • 4 attention heads, hidden dim 128
  • BatchNorm + dropout 0.3
     ↓
Global Mean Pooling
     ↓
MLP Classifier → {0: Clean, 1: Attacked}
```

Training: Adam lr=5e-4, weighted BCE (w+=2.18), 5-fold CV, 80 epochs

---

## Project Structure

```
├── run_all_attacks.py              # Main attack runner (all models × datasets)
├── run_brand_attack.py             # Brand manipulation attack
├── run_context_switch_attack.py    # Task switch attack
├── run_rephrase_meaning_attack.py  # Rephrasing and meaning change attacks
├── run_stealth_attack.py           # Stealth injection attack
├── run_umls_graph_attack.py        # UMLS knowledge graph attack
├── run_attack_mistakes.py          # Typo/abbreviation attack
├── run_full_dataset.py             # Full dataset evaluation
├── run_gnn_detector.py             # GNN detector v1 (40 entities)
├── run_gnn_detector_v2.py          # GNN detector v2 (208 entities)
├── run_bert_embedding_analysis.py  # BERT embedding analysis
├── run_output_intent_monitor.py    # Output intent monitoring
├── run_comparison_analysis.py      # Cross-model comparison
├── expand_umls_lookup.py           # UMLS entity lookup expansion
├── prepare_medqa.py                # MedQA dataset preparation
├── evaluate.py                     # Evaluation metrics (ROUGE, BLEU, BERTScore)
├── visualize.py                    # Result visualization
├── config.py                       # API keys (gitignored — create locally)
├── .gitignore
└── job_*.sb                        # SLURM job scripts (Delta HPC)
```

---

## Setup

### Requirements

```bash
pip install transformers torch torch_geometric \
            sentence-transformers rouge-score nltk \
            pandas openpyxl requests scispacy
```

### Configuration

Create a `config.py` file in the project root (this file is gitignored):

```python
import os

UMLS_API_KEY = os.environ.get("UMLS_API_KEY", "your-umls-key-here")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
```

Or set environment variables:

```bash
export HF_TOKEN=hf_...
export UMLS_API_KEY=your-umls-key
```

A UMLS API key can be obtained at: https://uts.nlm.nih.gov/uts/signup-login

---

## Running Experiments

### Run all attacks on a single model

```bash
python run_all_attacks.py --model medgemma --dataset pubmedqa
```

### Run UMLS graph attack

```bash
python run_umls_graph_attack.py
```

### Train GNN detector

```bash
python run_gnn_detector_v2.py
```

### On Delta HPC (SLURM)

```bash
sbatch job_medgemma_pubmedqa_final.sb
sbatch job_gnn_v2.sb
```

---

## Key Findings

1. **Brand manipulation is systemic** — 3 of 4 models exceed 70% success on open-ended questions
2. **Multiple-choice format is a natural defense** — brand attack drops to 0–5.7% on MedQA
3. **Medical fine-tuning does not protect** — Mistral (general) is the most vulnerable (avg BERTScore 0.8744)
4. **MedAlpaca is most robust** — 24.9% brand success on PubMedQA, 0% on MedQA
5. **7 of 8 attacks are undetectable** in realistic deployment scenarios
6. **UMLS graph attack is completely stealthy** — 0.6% detection rate
7. **GNN detector improves with entity coverage** — AUC 0.616 → 0.631 with 5× more entities

---

## Infrastructure

Experiments run on **NCSA Delta HPC** (NVIDIA A40 GPUs, 40GB VRAM)  
Account: `beqz-delta-gpu` | Partition: `gpuA40x4`

---

## Citation

```
@misc{nataraju2026promptinjection,
  title={Prompt Injection Attacks on Medical Large Language Models},
  author={Nataraju, Meghana},
  year={2026},
  institution={University of Illinois Chicago},
  note={CS598 Security and Privacy in ML}
}
```
