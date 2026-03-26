"""
retrieval.py
------------
Vector store construction and document retrieval using FAISS + bge-large-en-v1.5.
"""

import random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load embedding model once at module level
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')


def build_vector_store(dataset, sample_size=200, seed=42):
    """
    Build a FAISS vector store from the training split of the dataset.

    Args:
        dataset:     HuggingFace DatasetDict with 'train' split
        sample_size: Number of documents to index
        seed:        Random seed for reproducibility

    Returns:
        faiss_index: FAISS IndexFlatIP (inner product, works as cosine sim after normalization)
        doc_store:   Dict with keys 'contexts', 'answers', 'questions'
    """
    print(f"Building vector store with {sample_size} documents...")

    random.seed(seed)
    indices = random.sample(range(len(dataset['train'])), sample_size)

    contexts  = [dataset['train'][i]['context']  for i in indices]
    answers   = [dataset['train'][i]['answer']   for i in indices]
    questions = [dataset['train'][i]['question'] for i in indices]

    # bge models perform best with this prefix for retrieval tasks
    prefixed = ["Represent this sentence for retrieval: " + c for c in contexts]

    print("Encoding documents (this may take a few minutes on CPU)...")
    embeddings = embedding_model.encode(
        prefixed,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True   # normalize so dot product = cosine similarity
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))

    doc_store = {
        'contexts':  contexts,
        'answers':   answers,
        'questions': questions,
    }

    print(f"Vector store ready: {sample_size} documents, {dimension}-dim embeddings.")
    return index, doc_store


def retrieve_documents(query_text, faiss_index, doc_store, top_k=3):
    """
    Retrieve the top-k most relevant documents for a given query string.

    Args:
        query_text:  The text to search with (can be original query or hypothetical doc)
        faiss_index: FAISS index built by build_vector_store()
        doc_store:   Dict returned by build_vector_store()
        top_k:       Number of documents to return

    Returns:
        List of dicts: [{'rank', 'score', 'context', 'answer'}]
    """
    query_embedding = embedding_model.encode(
        ["Represent this sentence for retrieval: " + query_text],
        normalize_embeddings=True
    )

    scores, indices = faiss_index.search(
        query_embedding.astype('float32'),
        top_k
    )

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        results.append({
            'rank':    rank + 1,
            'score':   round(float(score), 4),
            'context': doc_store['contexts'][idx],
            'answer':  doc_store['answers'][idx],
        })

    return results


def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two numpy vectors."""
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def embed_text(text):
    """Embed a single text string and return a normalized numpy vector."""
    return embedding_model.encode(
        ["Represent this sentence for retrieval: " + text],
        normalize_embeddings=True
    )[0]
