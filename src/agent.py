from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if self.store.get_collection_size() == 0:
            return "Kho lưu trữ hiện đang trống. Không thể trả lời câu hỏi."
            
        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy ngữ cảnh nào phù hợp với câu hỏi."
            
        context_parts = []
        for i, res in enumerate(results, 1):
            source = res["metadata"].get("doc_id", "Unknown")
            context_parts.append(f"[{i}] Nguồn: {source}\n{res['content']}")
            
        context_str = "\n\n".join(context_parts)
        
        prompt = (
            f"Dựa vào các đoạn ngữ cảnh sau đây, hãy trả lời câu hỏi.\n"
            f"Yêu cầu:\n"
            f"1. Chỉ sử dụng thông tin trong ngữ cảnh được cung cấp. Nếu thông tin không có, hãy trả lời rõ là không tìm thấy.\n"
            f"2. Trích dẫn số thứ tự của đoạn văn chứa thông tin ở cuối câu (ví dụ: [1], [2]).\n\n"
            f"Ngữ cảnh:\n{context_str}\n\n"
            f"Câu hỏi: {question}\n"
            f"Trả lời:"
        )
        
        return self.llm_fn(prompt)
