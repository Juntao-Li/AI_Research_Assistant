def get_overlap_text(text, chunk_overlap):
    if chunk_overlap == 0:
        return ""

    overlap_start = max(0, len(text) - chunk_overlap)

    if overlap_start > 0:
        next_space = text.find(" ", overlap_start)

        if next_space != -1:
            overlap_start = next_space + 1

    return text[overlap_start : ].strip()


def chunk_text(text, chunk_size=500, chunk_overlap=100):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not text:
        return []

    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Paragraph itself is too large.
        if len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            long_chunks = split_long_text(
                paragraph,
                chunk_size,
                chunk_overlap
            )

            chunks.extend(long_chunks)
            continue

        if not current_chunk:
            current_chunk = paragraph
            continue

        #proposed new version of the current chunk after adding the next paragraph
        candidate = current_chunk + "\n\n" + paragraph

        #test whether the next paragraph can safely join the chunk which is currently built.
        if len(candidate) <= chunk_size:
            current_chunk = candidate

        else:
            chunks.append(current_chunk)

            overlap_text = get_overlap_text(
                current_chunk,
                chunk_overlap
            )

            if overlap_text:
                candidate_with_overlap = (
                    overlap_text + "\n\n" + paragraph
                )

                if len(candidate_with_overlap) <= chunk_size:
                    current_chunk = candidate_with_overlap

                else:
                    current_chunk = paragraph

            else:
                current_chunk = paragraph

    # Save anything remaining.
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def split_long_text(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0

    while start < len(text):
        target_end = min(start + chunk_size, len(text))
        actual_end = target_end

        if target_end < len(text):
            current_text = text[start:target_end]

            sentence_end = max(
                current_text.rfind("."),
                current_text.rfind("!"),
                current_text.rfind("?")
            )

            if sentence_end != -1:
                actual_end = start + sentence_end + 1

            else:
                word_end = current_text.rfind(" ")

                if word_end != -1:
                    actual_end = start + word_end

        chunk = text[start:actual_end].strip()

        if chunk:
            chunks.append(chunk)

        if actual_end == len(text):
            break

        next_start = max(0, actual_end - chunk_overlap)

        if next_start > start:
            next_space = text.find(" ", next_start, actual_end)

            if next_space != -1:
                next_start = next_space + 1

        # Make sure we always move forward.
        if next_start <= start:
            next_start = actual_end

        start = next_start

    return chunks



if __name__ == "__main__":

    text = (
        "RAG combines retrieval with language generation. "
        "It allows an LLM to use external information.\n\n"

        "The retrieval component searches for relevant information. "
        "This information is then provided to the language model.\n\n"

        "Evaluation is important because good generation depends on "
        "good retrieval. A production system should measure both."
    )

    chunks = chunk_text(
        text,
        chunk_size=180,
        chunk_overlap=30
    )

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk)
        print("Length:", len(chunk))

    print()
    print(chunk_text(""))