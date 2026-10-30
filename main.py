import argparse
import os
from pathlib import Path

from src.documents import load_documents
from src.generator import generate_answer
from src.indexer import load_or_create_index
from src.retriever import build_retriever, retrieve
from scripts.benchmark_corpus import load_benchmark_corpus


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ask questions about your local text, Markdown, Word, and PDF documents."
        )
    )
    corpus_group = parser.add_mutually_exclusive_group()
    corpus_group.add_argument(
        "--documents-dir",
        type=Path,
        help="Directory containing .txt, .md, .docx, and text-based .pdf documents.",
    )
    corpus_group.add_argument(
        "--corpus-manifest",
        type=Path,
        help="Manifest listing the corpus documents to use.",
    )
    parser.set_defaults(documents_dir=Path("data/documents"))
    parser.add_argument(
        "--question",
        help="Question to answer. If omitted, the CLI prompts for one.",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--retrieval-mode",
        choices=("dense", "hybrid"),
        default="dense",
        help="Retrieval ranking mode (default: dense).",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Show matching local passages without calling the chat API.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/.rag_cache/index.npz"),
        help="Path for the cached document embeddings and index.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="Chat completions model (or set OPENAI_MODEL).",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL (or set OPENAI_BASE_URL).",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be greater than 0")

    api_key = os.getenv("OPENAI_API_KEY")
    if not args.retrieve_only and not api_key:
        parser.error("Set OPENAI_API_KEY before running the assistant.")

    try:
        if args.corpus_manifest:
            documents, _ = load_benchmark_corpus(
                corpus_manifest=args.corpus_manifest
            )
        else:
            documents = load_documents(args.documents_dir)
        chunks, embeddings, chunk_sources = load_or_create_index(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            cache_path=args.cache_path,
        )
        retriever = build_retriever(chunks, embeddings, chunk_sources)

        question = (args.question or input("Ask a question: ")).strip()
        if not question:
            raise ValueError("The question cannot be empty.")

        results = retrieve(
            question,
            retriever.vector_store,
            top_k=args.top_k,
            mode=args.retrieval_mode,
            retriever=retriever,
        )
        print("\nRetrieved passages:")
        for index, result in enumerate(results, start=1):
            metadata = result["metadata"]
            page = f", page {metadata['page']}" if metadata["page"] else ""
            print(
                f"\n[{index}] {metadata['source']} "
                f"(chunk {metadata['chunk_id']}{page}, "
                f"{result['score_type']} {result['score']:.3f})\n"
                f"{result['text']}"
            )
        if args.retrieve_only:
            return

        context_parts = []
        for index, result in enumerate(results, start=1):
            metadata = result["metadata"]
            page = f", page {metadata['page']}" if metadata["page"] else ""
            context_parts.append(
                f"[{index}] Source: {metadata['source']} "
                f"(chunk {metadata['chunk_id']}{page})\n"
                f"{result['text']}"
            )
        context = "\n\n".join(context_parts)
        answer = generate_answer(
            question=question,
            context=context,
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
        )

        print(f"\n{answer}\n")
        print("Sources:")
        for index, result in enumerate(results, start=1):
            metadata = result["metadata"]
            page = f", page {metadata['page']}" if metadata["page"] else ""
            print(
                f"[{index}] {metadata['source']} "
                f"(chunk {metadata['chunk_id']}{page}, "
                f"score {result['score']:.3f})"
            )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
