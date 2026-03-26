"""
rag_methods.py
--------------
Three RAG pipelines compared in the paper:

  1. standard_rag         — baseline, query → retrieve → generate
  2. hyde_rag             — single hypothetical document → retrieve → generate
  3. hypo_select_sim_rag  — four hypothetical documents → Best Vector → retrieve → generate
"""

import numpy as np
from src.retrieval import retrieve_documents, embed_text
from src.generation import generate_four_paths, generate_few_shot_doc


# ---------------------------------------------------------------------------
# Shared answer generation
# ---------------------------------------------------------------------------

def generate_answer(question, contexts, client, model="moonshot-v1-8k", temperature=0.1):
    """
    Generate a final answer given the question and retrieved context passages.
    """
    context_text = "\n\n---\n\n".join(contexts)

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": (
                    "Based on the following context, answer the question concisely.\n"
                    "If the context does not contain enough information, say so.\n\n"
                    f"Context:\n{context_text}\n\n"
                    f"Question: {question}\n\n"
                    "Answer:"
                )
            }
        ]
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Method 1: Standard RAG
# ---------------------------------------------------------------------------

def standard_rag(question, faiss_index, doc_store, client,
                 model="moonshot-v1-8k", top_k=3):
    """
    Baseline RAG: use the original query directly for retrieval.

    Pipeline:
        query → embed → FAISS search → LLM answer
    """
    results  = retrieve_documents(question, faiss_index, doc_store, top_k=top_k)
    contexts = [r['context'] for r in results]
    return generate_answer(question, contexts, client, model)


# ---------------------------------------------------------------------------
# Method 2: HyDE
# ---------------------------------------------------------------------------

def hyde_rag(question, faiss_index, doc_store, client,
             model="moonshot-v1-8k", top_k=3, temperature=0.1):
    """
    HyDE: generate one hypothetical document, use it for retrieval.

    Pipeline:
        query → LLM generates fake doc → embed fake doc → FAISS search → LLM answer

    Reference: Gao et al., "Precise Zero-Shot Dense Retrieval Without Relevance Labels" (ACL 2023)
    """
    hypo_doc = generate_few_shot_doc(question, client, model, temperature=temperature)
    results  = retrieve_documents(hypo_doc, faiss_index, doc_store, top_k=top_k)
    contexts = [r['context'] for r in results]
    return generate_answer(question, contexts, client, model)


# ---------------------------------------------------------------------------
# Method 3: HypoSelectSimRAG (this paper)
# ---------------------------------------------------------------------------

def best_vector_select(question, hypo_docs, verbose=False):
    """
    Select the hypothetical document most semantically aligned with the
    original query using cosine similarity in embedding space.

    Args:
        question:  Original user query string
        hypo_docs: List of four hypothetical document strings
        verbose:   If True, print similarity scores for all paths

    Returns:
        best_doc:  The selected hypothetical document string
        scores:    List of similarity scores for all four paths
    """
    query_vec = embed_text(question)
    scores    = []

    for doc in hypo_docs:
        doc_vec = embed_text(doc)
        score   = float(np.dot(query_vec, doc_vec))  # vectors are normalized → dot = cosine
        scores.append(round(score, 4))

    if verbose:
        labels = [
            "Few-shot (T=0.1)",
            "Few-shot (T=0.9)",
            "Question-oriented (T=0.1)",
            "Question-oriented (T=0.9)",
        ]
        print("Path similarity scores:")
        for label, score in zip(labels, scores):
            marker = " ← selected" if score == max(scores) else ""
            print(f"  {label}: {score}{marker}")

    best_idx = scores.index(max(scores))
    return hypo_docs[best_idx], scores


def hypo_select_sim_rag(question, faiss_index, doc_store, client,
                        model="moonshot-v1-8k", top_k=3, verbose=False):
    """
    HypoSelectSimRAG: generate four hypothetical documents, select the best
    one via cosine similarity (Best Vector), then retrieve and generate.

    Pipeline:
        query
          ├─ Few-shot (T=0.1)        ─┐
          ├─ Few-shot (T=0.9)         │
          ├─ Question-oriented (T=0.1)├─ Best Vector selection
          └─ Question-oriented (T=0.9)┘
                    │
                    ▼
             FAISS retrieval
                    │
                    ▼
              LLM final answer

    Args:
        question:    User query string
        faiss_index: FAISS index from build_vector_store()
        doc_store:   Doc store dict from build_vector_store()
        client:      OpenAI-compatible API client
        model:       Model name string
        top_k:       Number of documents to retrieve
        verbose:     If True, print intermediate outputs

    Returns:
        answer:  Final generated answer string
    """
    # Step 1: Generate four hypothetical documents
    if verbose:
        print("Generating four hypothetical documents...")
    hypo_docs = generate_four_paths(question, client, model)

    # Step 2: Select best document via cosine similarity
    best_doc, scores = best_vector_select(question, hypo_docs, verbose=verbose)

    # Step 3: Retrieve real documents using the best hypothetical doc
    results  = retrieve_documents(best_doc, faiss_index, doc_store, top_k=top_k)
    contexts = [r['context'] for r in results]

    # Step 4: Generate final answer
    return generate_answer(question, contexts, client, model)
