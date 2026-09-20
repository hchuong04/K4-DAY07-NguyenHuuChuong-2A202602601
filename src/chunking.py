from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        cleaned_text = text.strip()
        if not cleaned_text:
            return []
        
        raw_sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        if not sentences:
            return []

        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_batch = sentences[i : i + self.max_sentences_per_chunk]
            merged_chunk = " ".join(chunk_batch).strip()
            if merged_chunk:
                chunks.append(merged_chunk)
        
        return chunks



class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        raw_chunks = self._split(text.strip(), self.separators)
        return [c.strip() for c in raw_chunks if c.strip()]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]
        
        if not remaining_separators:
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        sep = remaining_separators[0]
        next_separators = remaining_separators[1:]

        if sep =="":
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        if sep not in current_text:
            return self._split(current_text, next_separators)

        splits = current_text.split(sep)
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_len = 0

        for part in splits:
            if not part:
                continue

            if len(part) > self.chunk_size:
                if current_chunk:
                    chunks.append(sep.join(current_chunk))
                    current_chunk = []
                    current_len = 0

                sub_chunks = self._split(part, next_separators)
                chunks.extend(sub_chunks)
            else:
                additional_len = len(part) + (len(sep) if current_chunk else 0)

                if current_len + additional_len <= self.chunk_size:
                    current_chunk.append(part)
                    current_len += additional_len
                else: 
                    if current_chunk:
                        chunks.append(sep.join(current_chunk))
                    current_chunk = [part]
                    current_len = len(part)
                
            
        if current_chunk:
            chunks.append(sep.join(current_chunk))
        
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    # 1. Tính tích vô hướng (dot product)
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    # 2. Tính độ dài (magnitude / L2 norm) của từng vector
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    # 3. Kiểm tra trường hợp vector 0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        """
        Runs multiple chunking strategies on the input text, 
        computes length statistics, and returns a summary dict.
        """
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            # Ước lượng ~60-80 ký tự/câu để ánh xạ chunk_size sang số câu tương ứng
            "by_sentences": SentenceChunker(max_sentences_per_chunk=max(1, chunk_size // 70)),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        comparison_results = {}

        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)

            if chunks:
                lengths = [len(c) for c in chunks]
                stats = {
                    "count": len(chunks),
                    "min_chunk_size": min(lengths),
                    "max_chunk_size": max(lengths),
                    "avg_length": round(sum(lengths) / len(lengths), 2),
                    "chunks": chunks,
                }
            else:
                stats = {
                    "count": 0,
                    "min_chunk_size": 0,
                    "max_chunk_size": 0,
                    "avg_length": 0.0,
                    "chunks": [],
                }

            comparison_results[name] = stats

        return comparison_results