from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_text(text):
    return _get_model().encode(text)


def embed_texts(texts):
    return _get_model().encode(texts)


if __name__ == "__main__":
    chunks = [
        "Gradient descent minimizes the loss function.",
        "RAG retrieves external information.",
        "Docker packages software into containers.",
    ]
    embeddings = embed_texts(chunks)
    print(embeddings)
    print("Shape:", embeddings.shape)
    print("Type:", type(embeddings))
