import hashlib
import json

from .embedding import embed_text
from .hybrid_retrieval import BM25Index, reciprocal_rank_fusion
from .vector_store import VectorStore
from .embedding import EMBEDDING_MODEL


def retrieval_fingerprint(documents, chunk_size, chunk_overlap):
    payload = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "documents": documents,
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class Retriever:
    def __init__(self, vector_store, chunks=None, metadata=None, bm25_index=None):
        self.vector_store = vector_store
        if chunks is None or metadata is None:
            chunks = [item["text"] for item in vector_store.items]
            metadata = [item["metadata"] for item in vector_store.items]
        if len(chunks) != len(metadata):
            raise ValueError("chunks and metadata must have the same length")
        self.chunks = list(chunks)
        self.metadata = list(metadata)
        self.bm25_index = bm25_index or BM25Index(self.chunks, self.metadata)

    def search(self, query, top_k=3, mode="dense"):
        if mode not in {"dense", "hybrid"}:
            raise ValueError("mode must be 'dense' or 'hybrid'")
        query_embedding = embed_text(query)
        return self.search_with_embedding(query, query_embedding, top_k, mode)

    def search_with_embedding(self, query, query_embedding, top_k=3, mode="dense"):
        if mode not in {"dense", "hybrid"}:
            raise ValueError("mode must be 'dense' or 'hybrid'")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        candidate_count = len(self.chunks)
        dense_results = self.vector_store.search(
            query_embedding,
            top_k=candidate_count if mode == "hybrid" else top_k,
        )
        if mode == "dense":
            return [
                {**result, "score_type": "cosine similarity"}
                for result in dense_results
            ]

        lexical_results = self.bm25_index.search(query, top_k=candidate_count)
        results = reciprocal_rank_fusion(
            dense_results, lexical_results, top_k=top_k
        )
        return [
            {**result, "score_type": "reciprocal-rank fusion score"}
            for result in results
        ]


def build_retriever(chunks, embeddings, metadata):
    if not (len(chunks) == len(embeddings) == len(metadata)):
        raise ValueError("chunks, embeddings, and metadata must have equal lengths")
    vector_store = VectorStore()
    for chunk, embedding, item_metadata in zip(chunks, embeddings, metadata):
        vector_store.add(chunk, embedding, item_metadata)
    return Retriever(vector_store, chunks, metadata)


def retrieve(query, vector_store, top_k=3, mode="dense", retriever=None):
    """Compatibility wrapper; long-lived callers should reuse a Retriever."""
    if retriever is None:
        retriever = Retriever(vector_store)
    return retriever.search(query, top_k=top_k, mode=mode)


if __name__ == "__main__":
    from embedding import embed_texts
    from vector_store import VectorStore

    chunks = [
        "RAG retrieves relevant external information before generation.",
        "Docker packages applications into containers.",
        "Gradient descent is used to minimize a loss function."
    ]

    embeddings = embed_texts(chunks)

    store = VectorStore()

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        store.add(
            text=chunk,
            embedding=embedding,
            metadata={
                "source": "example.txt",
                "chunk_id": i
            }
        )

    query = "How does RAG use outside knowledge?"

    results = retrieve(
        query,
        store,
        top_k=2
    )

    for result in results:
        print("Score:", result["score"])
        print("Text:", result["text"])
        print("Metadata:", result["metadata"])
        print()
