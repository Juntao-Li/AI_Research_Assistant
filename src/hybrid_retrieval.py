import math
import re
from collections import Counter


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _tokens(text):
    return _TOKEN_PATTERN.findall(text.casefold())


def bm25_search(query, chunks, metadata, top_k, k1=1.5, b=0.75):
    tokenized_chunks = [_tokens(chunk) for chunk in chunks]
    query_terms = _tokens(query)
    if not tokenized_chunks:
        return []

    document_frequencies = Counter(
        term
        for terms in tokenized_chunks
        for term in set(terms)
    )
    average_length = sum(map(len, tokenized_chunks)) / len(tokenized_chunks)
    scores = []
    for chunk, terms, item_metadata in zip(chunks, tokenized_chunks, metadata):
        frequencies = Counter(terms)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                1
                + (len(tokenized_chunks) - document_frequencies[term] + 0.5)
                / (document_frequencies[term] + 0.5)
            )
            normalization = frequency + k1 * (
                1 - b + b * len(terms) / max(average_length, 1)
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1) / normalization
            )
        scores.append(
            {
                "score": score,
                "text": chunk,
                "metadata": item_metadata,
            }
        )

    scores.sort(key=lambda result: result["score"], reverse=True)
    return scores[:top_k]


class BM25Index:
    """Reusable BM25 state with precomputed postings and scoring factors."""

    def __init__(self, chunks, metadata, k1=1.5, b=0.75):
        if len(chunks) != len(metadata):
            raise ValueError("chunks and metadata must have the same length")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 parameters require k1 > 0 and 0 <= b <= 1")
        self.chunks = list(chunks)
        self.metadata = list(metadata)
        self.k1 = k1
        self.b = b
        self.lengths = []
        self.term_frequencies = []
        self.document_frequencies = Counter()
        self.postings = {}

        for document_id, chunk in enumerate(self.chunks):
            terms = _tokens(chunk)
            frequencies = Counter(terms)
            self.lengths.append(len(terms))
            self.term_frequencies.append(frequencies)
            self.document_frequencies.update(frequencies.keys())
            for term, frequency in frequencies.items():
                self.postings.setdefault(term, []).append(
                    (document_id, frequency)
                )

        count = len(self.chunks)
        self.average_length = sum(self.lengths) / count if count else 0.0
        self.inverse_document_frequencies = {
            term: math.log(
                1
                + (count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for term, document_frequency in self.document_frequencies.items()
        }
        self.length_normalizers = [
            k1 * (1 - b + b * length / max(self.average_length, 1))
            for length in self.lengths
        ]

    def search(self, query, top_k):
        if top_k <= 0 or not self.chunks:
            return []
        scores = [0.0] * len(self.chunks)
        for term in _tokens(query):
            idf = self.inverse_document_frequencies.get(term)
            if idf is None:
                continue
            for document_id, frequency in self.postings[term]:
                normalization = frequency + self.length_normalizers[document_id]
                scores[document_id] += idf * (
                    frequency * (self.k1 + 1) / normalization
                )
        ranked_indices = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )[:top_k]
        return [
            {
                "score": scores[index],
                "text": self.chunks[index],
                "metadata": self.metadata[index],
            }
            for index in ranked_indices
        ]


def reciprocal_rank_fusion(dense_results, lexical_results, top_k, constant=60):
    fused = {}
    payloads = {}
    for results in (dense_results, lexical_results):
        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]
            key = (
                metadata.get("source"),
                metadata.get("page"),
                metadata.get("chunk_id"),
            )
            fused[key] = fused.get(key, 0.0) + 1 / (constant + rank)
            payloads[key] = result

    ranked = []
    for key, score in fused.items():
        ranked.append(
            {
                **payloads[key],
                "score": score,
            }
        )
    ranked.sort(key=lambda result: result["score"], reverse=True)
    return ranked[:top_k]
