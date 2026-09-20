import json
import re
import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from src.models import Document
from src.store import EmbeddingStore
from src.chunking import RecursiveChunker, FixedSizeChunker, SentenceChunker
from src.embeddings import LocalEmbedder



def parse_md(path: Path):
    """Đọc file .md, tách frontmatter thành metadata, phần còn lại là nội dung."""
    text = path.read_text(encoding="utf-8")
    parts = text.split("---")
    if len(parts) >= 3:
        fm_text = parts[1]
        content = "---".join(parts[2:]).strip() # Ghép lại nếu thân bài có ---
        metadata = dict(re.findall(r'^(\w+):\s*(.+)$', fm_text, re.M))
    else:
        metadata = {}
        content = text.strip()
    return metadata, content


def main():
    data_dir = Path("data/shoppe_warranty")
    md_files = sorted(data_dir.glob("*.md"))
    
    # -------------------------------------------------------------------
    # MỖI NGƯỜI CHỈ ĐỔI DÒNG NÀY (Chiến lược Chunking của riêng bạn)
    # -------------------------------------------------------------------
    chunker = RecursiveChunker(chunk_size=500)
    # chunker = SentenceChunker(max_sentences_per_chunk=3)
    # chunker = FixedSizeChunker(chunk_size=300, overlap=50)
    
    try:
        print("Đang tải model LocalEmbedder (có thể mất 1-2 phút lần đầu)...")
        embedder = LocalEmbedder()
        print(f"Sử dụng LocalEmbedder: {embedder.model_name}")
    except Exception as e:
        print(f"Lỗi khởi tạo Local Embedder: {e}\nHãy đảm bảo đã chạy: pip install -r requirements-local.txt")
        return

    store = EmbeddingStore(embedding_fn=embedder)
    documents = []
    
    # BƯỚC 1 & 2: Parse MD và Chunking
    print("Đang xử lý tài liệu và chunking...")
    for p in md_files:
        metadata, content = parse_md(p)
        chunks = chunker.chunk(content)
        
        for i, chunk in enumerate(chunks):
            # Nhồi frontmatter vào metadata của chunk, lưu lại file gốc ở doc_id
            chunk_meta = {**metadata, "doc_id": p.stem}
            doc = Document(id=f"{p.stem}#{i}", content=chunk, metadata=chunk_meta)
            documents.append(doc)
            
    # BƯỚC 3: Nạp vào store
    print(f"Đã tạo {len(documents)} chunks. Đang nạp vào store...")
    store.add_documents(documents)
    
    # Chạy tập Golden Queries
    golden_file = data_dir / "golden_set.json"
    if not golden_file.exists():
        print(f"Không tìm thấy file {golden_file}")
        return
        
    with open(golden_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    print("\n" + "="*80)
    print(" KẾT QUẢ BENCHMARK ".center(80))
    print("="*80)
    
    # BƯỚC 4: Tìm kiếm và in top-3
    for q in queries:
        print(f"\n📝 Câu hỏi: {q['query']}")
        print(f"🔍 Filter : {q.get('metadata_filter')}")
        print(f"🎯 Gold   : {q['gold_answer']}")
        print(f"📁 Nguồn chuẩn: {q['source_file']}")
        print("-" * 60)
        
        results = store.search_with_filter(
            query=q["query"],
            top_k=3,
            metadata_filter=q.get("metadata_filter")
        )
        
        if not results:
            print("❌ Không tìm thấy chunk nào phù hợp!")
            continue
            
        for i, res in enumerate(results, 1):
            source = res["metadata"].get("doc_id", "Unknown")
            score = res["score"]
            snippet = res["content"].replace("\n", " ").strip()
            # Cắt ngắn snippet nếu quá dài
            if len(snippet) > 150:
                snippet = snippet[:150] + "..."
            
            # Highlight nếu source trùng với nguồn chuẩn
            match = "✅" if f"{source}.md" == q["source_file"] else "❌"
            
            print(f"  {match} Top {i} | Score: {score:.4f} | Nguồn: {source}.md")
            print(f"          {snippet}")

if __name__ == "__main__":
    main()
