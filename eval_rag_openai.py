import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from src.models import Document
from src.store import EmbeddingStore
from src.chunking import RecursiveChunker
from src.embeddings import LocalEmbedder
from bench import parse_md

# Load biến môi trường (bao gồm OPENAI_API_KEY)
load_dotenv()

def generate_rag_answer(client, query, context):
    """Gọi OpenAI API để sinh câu trả lời dựa trên context"""
    prompt = f"""Bạn là một trợ lý ảo của Shopee. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa VÀO DUY NHẤT ngữ cảnh (context) được cung cấp dưới đây.
Nếu ngữ cảnh không chứa thông tin để trả lời, hãy nói 'Tôi không tìm thấy đủ thông tin trong tài liệu'.
Hãy trả lời ngắn gọn, súc tích và đúng trọng tâm.

Ngữ cảnh:
{context}

Câu hỏi: {query}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Hoặc gpt-3.5-turbo tùy theo cấu hình của bạn
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip().replace('\n', ' ')
    except Exception as e:
        return f"Lỗi gọi API: {str(e)}"

def main():
    # Khởi tạo OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    data_dir = Path("data/shoppe_warranty")
    md_files = sorted(data_dir.glob("*.md"))
    
    print("Đang khởi tạo chunker và embedder...")
    chunker = RecursiveChunker(chunk_size=500)
    embedder = LocalEmbedder()
    store = EmbeddingStore(embedding_fn=embedder)
    
    documents = []
    
    # 1. Chunk và Nạp tài liệu
    for p in md_files:
        metadata, content = parse_md(p)
        chunks = chunker.chunk(content)
        
        for i, chunk in enumerate(chunks):
            chunk_meta = {**metadata, "doc_id": p.stem}
            doc = Document(id=f"{p.stem}#{i}", content=chunk, metadata=chunk_meta)
            documents.append(doc)
            
    store.add_documents(documents)
    
    # 2. Đọc Golden Set
    golden_file = data_dir / "golden_set.json"
    if not golden_file.exists():
        print(f"Không tìm thấy file {golden_file}")
        return
        
    with open(golden_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    # 3. In header của bảng Markdown
    print("\n--- BẮT ĐẦU CHẠY ĐÁNH GIÁ RAG ---\n")
    print("| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |")
    print("|---|---|---|---|---|---|")
    
    # 4. Truy vấn và sinh câu trả lời
    for i, q in enumerate(queries, 1):
        query_text = q["query"]
        
        results = store.search_with_filter(
            query=query_text,
            top_k=3,
            metadata_filter=q.get("metadata_filter")
        )
        
        if not results:
            print(f"| {i} | {query_text} | Không có kết quả | N/A | Không | |")
            continue
            
        top1 = results[0]
        score = top1["score"]
        source_doc = top1["metadata"].get("doc_id", "Unknown")
        
        # Rút gọn chunk để in ra bảng (lấy 80 ký tự đầu)
        chunk_snippet = top1["content"].replace("\n", " ")

        # Đánh giá Relevant dựa trên việc source truy xuất được có khớp source chuẩn không
        is_relevant = "Có (✅)" if f"{source_doc}.md" == q["source_file"] else "Không (❌)"
        
        # GỘP NỘI DUNG CẢ 3 CHUNKS LẠI LÀM CONTEXT
        combined_context = "\n---\n".join([res["content"] for res in results])
        
        # Gọi Agent trả lời
        agent_answer = generate_rag_answer(client, query_text, combined_context)
        
        # Rút gọn câu trả lời để in ra bảng
        answer_snippet = agent_answer

            
        print(f"| {i} | {query_text} | {chunk_snippet} | {score:.4f} | {is_relevant} | {answer_snippet} |")

if __name__ == "__main__":
    main()
