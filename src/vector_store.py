import numpy as np

class VectorStore:
    def __init__(self):
        self.items = []
        self._normalized_embeddings = None

    def add(self, text, embedding, metadata = None):
        embedding = np.asarray(embedding, dtype=np.float32)
        item = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {}
        }

        self.items.append(item)
        self._normalized_embeddings = None

    def cosine_similarity(self, vector_a, vector_b):
        dot_product = np.dot(vector_a, vector_b)

        magnitude_a = np.linalg.norm(vector_a)
        magnitude_b = np.linalg.norm(vector_b)

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def prepare_dense_index(self):
        if self._normalized_embeddings is None and self.items:
            embeddings = np.stack(
                [item["embedding"] for item in self.items]
            )
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self._normalized_embeddings = np.divide(
                embeddings,
                norms,
                out=np.zeros_like(embeddings),
                where=norms != 0,
            )

    def search(self, query_embedding, top_k = 3):
        if top_k <= 0 or not self.items:
            return []

        self.prepare_dense_index()

        query = np.asarray(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        normalized_query = (
            query / query_norm if query_norm != 0 else np.zeros_like(query)
        )
        scores = self._normalized_embeddings @ normalized_query

        # A stable sort preserves the original insertion-order tie breaking.
        ranked_indices = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {
                "score": scores[index],
                "text": self.items[index]["text"],
                "metadata": self.items[index]["metadata"],
            }
            for index in ranked_indices
        ]

if __name__ == "__main__":
    from embedding import embed_text, embed_texts

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

    query_embedding = embed_text(query)

    results = store.search(
        query_embedding,
        top_k=2
    )

    for result in results:
        print("Score:", result["score"])
        print("Text:", result["text"])
        print("Metadata:", result["metadata"])
        print()
