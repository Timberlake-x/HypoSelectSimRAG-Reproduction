# HypoSelectSimRAG — Reproduction & Critical Analysis

> Independent reproduction of **"HypoSelectSimRAG: Enhancing Answer Accuracy in RAG via Multi-Path Self-Consistent Query Translation"** (ICCEIC 2025, EI Compendex)
>
> This project goes beyond reproducing numbers — it explains *why* each method works, identifies conditions where improvements disappear, and documents real limitations found through hands-on experimentation.

---

## Table of Contents

- [Background](#background)
- [Method Overview](#method-overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Results](#results)
- [Key Findings](#key-findings)
- [Limitations of the Original Paper](#limitations-of-the-original-paper)
- [Environment](#environment)

---

## Background

### Why RAG?

Large language models have a fundamental problem: their knowledge is frozen at training time. Ask about recent events, private documents, or specialized domains — they either don't know or confidently fabricate answers (hallucination).

**RAG (Retrieval-Augmented Generation)** solves this by giving the model an open book:

```
User Query → Retrieve relevant documents → LLM generates answer based on documents
```

Instead of relying on memorized knowledge, the model retrieves real content first, then generates grounded answers.

### The Core Problem RAG Still Has

Standard RAG has a semantic gap problem. User queries are typically short (one sentence), while documents are long and information-dense. When you search for:

> *"Who wrote For The Fallen?"*

...the embedding of this short query may not closely match the embedding of a long article that contains the answer buried in paragraph 3.

**HyDE** addresses this by generating a "hypothetical document" — a fake but plausible answer — and using *that* for retrieval. A detailed fake answer is semantically closer to a real document than a short query is.

**HypoSelectSimRAG** (this paper) extends HyDE: instead of generating one hypothetical document, generate four via different strategies, then select the best one using vector similarity.

---

## Method Overview

### Three Methods Side by Side

```
Standard RAG:
Query ──────────────────────────────► Retrieve ──► Generate Answer

HyDE:
Query ──► LLM generates 1 fake doc ──► Retrieve ──► Generate Answer

HypoSelectSimRAG:
         ┌── Few-shot (T=0.1) ──────────────────────────────┐
         ├── Few-shot (T=0.9) ──────────────────────────────┤
Query ───┤                                                   ├── Best Vector ──► Retrieve ──► Answer
         ├── Question-oriented (T=0.1) ─────────────────────┤
         └── Question-oriented (T=0.9) ─────────────────────┘
```

### The Four Generation Paths

| Path | Strategy | Temperature | Effect |
|------|----------|-------------|--------|
| 1 | Few-shot prompting | 0.1 | Conservative, format-consistent |
| 2 | Few-shot prompting | 0.9 | Creative, semantically diverse |
| 3 | Question-oriented | 0.1 | Task-specific template, precise |
| 4 | Question-oriented | 0.9 | Task-specific template, exploratory |

**Few-shot**: Provides 2 example question-answer pairs so the LLM learns the expected format and style before generating.

**Question-oriented**: First classifies the query into one of 14 types (ExtractiveQA, FactCheck, ComparativeQA, etc.), then applies a type-specific prompt template for more targeted generation.

**Best Vector Selection**: All four hypothetical documents are embedded into vector space. The one with the highest cosine similarity to the original query is selected for retrieval.

### Why Best Vector Works

```
Original query embedding:  [0.12, -0.34, 0.87, ...]   (short, sparse signal)

Path 1 embedding:          [0.15, -0.31, 0.84, ...]   similarity: 0.79
Path 2 embedding:          [0.09, -0.28, 0.91, ...]   similarity: 0.83  ← selected
Path 3 embedding:          [0.21, -0.44, 0.76, ...]   similarity: 0.71
Path 4 embedding:          [0.18, -0.39, 0.79, ...]   similarity: 0.74
```

By measuring cosine similarity between each hypothetical document and the original query, we directly identify which generated document best preserves the original intent — without relying on indirect voting or LLM scoring.

---

## Project Structure

```
HypoSelectSimRAG-Reproduction/
│
├── README.md                    ← You are here
├── requirements.txt             ← Dependencies
│
├── src/
│   ├── retrieval.py             ← Vector store + FAISS retrieval
│   ├── generation.py            ← Hypothetical document generation
│   ├── rag_methods.py           ← Standard RAG / HyDE / HypoSelectSimRAG
│   └── evaluation.py            ← RAGAS evaluation utilities
│
├── notebook/
│   └── full_experiment.ipynb   ← Step-by-step Kaggle notebook
│
└── results/
    └── comparison_results.md   ← Experimental results and analysis
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/HypoSelectSimRAG-Reproduction.git
cd HypoSelectSimRAG-Reproduction
pip install -r requirements.txt
```

### 2. Set up your API key

This reproduction uses **Kimi (Moonshot AI)** instead of GPT-3.5-turbo.
Get your API key at [platform.moonshot.cn](https://platform.moonshot.cn).

```python
from openai import OpenAI

client = OpenAI(
    api_key="your_kimi_api_key",
    base_url="https://api.moonshot.cn/v1"
)
```

> **Note on model substitution**: The original paper uses GPT-3.5-turbo with temperature up to 1.1. Kimi caps temperature at 1.0, so we use 0.1 and 0.9 instead. This is a reproducibility limitation documented in our findings.

### 3. Run a quick test

```python
from src.rag_methods import standard_rag, hyde_rag, hypo_select_sim_rag
from src.retrieval import build_vector_store
from datasets import load_dataset

dataset = load_dataset("neural-bridge/rag-dataset-1200")
faiss_index, doc_store = build_vector_store(dataset, sample_size=200)

question = "Who did LeBron James decide to join for the 2010-11 NBA season?"

print(standard_rag(question, faiss_index, doc_store))
print(hyde_rag(question, faiss_index, doc_store))
print(hypo_select_sim_rag(question, faiss_index, doc_store))
```

---

## Results

### Qualitative Comparison (6 Test Cases)

| Question Type | Standard RAG | HyDE | HypoSelectSimRAG |
|--------------|-------------|------|-----------------|
| Simple factual (LeBron James) | ✅ Partial | ✅ Partial | ✅ Partial |
| Named entity (Moses / Parsons) | ✅ Correct | ✅ Correct | ✅ Correct |
| Financial definition (Free Cash Flow) | ❌ Not found | ❌ Not found | ❌ Not found |
| Simple age requirement (forklift) | ✅ Correct | ✅ Correct | ✅ Correct |
| Historical fact (Proserpina dam) | ❌ Hallucinated | ❌ Hallucinated | ❌ Hallucinated |
| Complex opinion (Bollywood / Wuthering Heights) | ❌ Not found | ❌ Not found | ❌ Not found |

### Similarity Scores — Best Vector Selection

For the Moses question, the four paths scored:

```
Path 1 (Few-shot T=0.1):         0.7896
Path 2 (Few-shot T=0.9):         0.8202
Path 3 (Question-oriented T=0.1): 0.8276
Path 4 (Question-oriented T=0.9): 0.8608  ← selected
```

Question-oriented paths consistently scored higher on factual questions, consistent with the paper's ablation findings.

---

## Key Findings

### Finding 1: Advantage is concentrated in semantically complex queries

On simple factual questions, all three methods return identical answers. Standard RAG already retrieves correctly when the query is clear. HypoSelectSimRAG's multi-path expansion adds value only when the original query is too vague or short to retrieve the right document on its own.

**Practical implication**: The method is most valuable for knowledge-intensive, open-domain QA — not for simple lookup tasks.

### Finding 2: The reported improvement is a statistical aggregate

The paper reports Factual Correctness improving from 0.52 (standard RAG) to 0.69 (HypoSelectSimRAG) on 200 test cases. Our case-by-case analysis shows this improvement is not uniform:

- Simple questions: no difference across methods
- Missing documents: all methods fail equally  
- Semantically ambiguous questions: HypoSelectSimRAG genuinely outperforms

The aggregate number is real but masks this heterogeneity. Users should not expect uniform improvement across all query types.

### Finding 3: Temperature parameter is model-dependent

The paper uses temperature=1.1 (GPT-3.5-turbo supports up to 2.0). Kimi caps at 1.0. This means:
- The exact diversity between "conservative" and "exploratory" paths cannot be reproduced
- Any model with temperature cap ≤ 1.0 will produce less path diversity than the original
- The paper does not acknowledge this as a reproducibility constraint

### Finding 4: Database coverage is the binding constraint

When the relevant document is absent from the vector store, all three methods fail — often with hallucinated answers that sound confident. No retrieval strategy can compensate for missing content.

This is the most important practical insight: **investing in document coverage gives more return than investing in query expansion sophistication**.

### Finding 5: Hallucination propagates through the pipeline

For the Proserpina dam question, the hypothetical documents generated by HyDE and HypoSelectSimRAG contained plausible but incorrect details (wrong city names). These hallucinated documents then retrieved slightly wrong real documents, and the final answer was wrong. The multi-path selection did not prevent this — all four paths hallucinated similarly.

---

## Limitations of the Original Paper

| Category | Limitation |
|----------|-----------|
| **Experimental scale** | 200 samples from one dataset; statistical significance not verified |
| **Dataset diversity** | All results from rag-dataset-1200; no cross-domain or cross-language validation |
| **Hyperparameter justification** | Temperature values (0.1, 1.1) and path count (4) are not ablated — why not 3 or 6 paths? |
| **Error analysis** | Paper shows only where the method wins; no analysis of failure cases |
| **Evaluation circularity** | RAGAS uses an LLM to evaluate LLM-generated answers — potential self-reinforcement bias |
| **Computational cost** | HypoSelectSimRAG requires ~6x more LLM calls per query than standard RAG; not discussed |
| **Classification robustness** | Question-type misclassification propagates through the whole pipeline; never analyzed |
| **Reproducibility** | Temperature=1.1 is GPT-specific; cannot be reproduced exactly with most other models |

---

## Environment

| Component | This Reproduction | Original Paper |
|-----------|------------------|---------------|
| LLM | Kimi moonshot-v1-8k | GPT-3.5-turbo |
| Embedding | BAAI/bge-large-en-v1.5 | text-embedding-ada-002 |
| Temperature range | 0.1 / 0.9 | 0.1 / 1.1 |
| Vector store | FAISS (faiss-cpu) | FAISS |
| Evaluation | RAGAS 0.4.3 | RAGAS |
| Runtime | Kaggle Notebook (CPU) | Not specified |
| Dataset | neural-bridge/rag-dataset-1200 | Same |

```
sentence-transformers==5.2.3
faiss-cpu==1.13.2
ragas==0.4.3
datasets==4.8.3
openai==2.30.0
numpy==2.0.2
```

---

## About This Reproduction

This project was built as a self-directed deep learning exercise — not just to verify the paper's claims, but to understand *exactly* why each design decision was made, and where the boundaries of the method lie.

The most valuable outcome was not the numbers. It was developing the habit of asking: *under what conditions does this work, and under what conditions does it fail?* That question is the foundation of doing research rather than just reading it.

