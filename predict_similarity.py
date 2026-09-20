from sentence_transformers import SentenceTransformer, util

# Tải một mô hình đa ngôn ngữ nhỏ gọn, hỗ trợ tốt tiếng Việt
print("Đang tải mô hình...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Danh sách 5 câu A
sentences_A = [
    "Con mèo đang ngủ trên ghế sofa.",
    "Bầu trời hôm nay rất trong xanh và đầy nắng.",
    "Anh ấy đang đọc một cuốn tiểu thuyết lịch sử.",
    "Lab coach rất đáng yêu",
    "Giảng viên dạy rất hay và dễ hiểu."
]

# Danh sách 5 câu B tương ứng
sentences_B = [
    "Một chú mèo con đang nằm thiu thiu trên chiếc ghế dài.",
    "Thời tiết hôm nay thật đẹp, trời nắng và không một gợn mây.",
    "Anh ấy đang chơi đàn piano trong phòng khách.",
    "Lab coach rất dễ thương.",
    "Vlearn tutor dạy rất khó hiểu."
]

print("\n--- KẾT QUẢ ĐIỂM THỰC TẾ ---")
for i in range(5):
    # Mã hóa (Encode) các câu thành các vector số học
    embedding_A = model.encode(sentences_A[i])
    embedding_B = model.encode(sentences_B[i])
    
    # Tính độ tương tự Cosine giữa 2 vector (kết quả từ -1.0 đến 1.0)
    # Càng gần 1.0 nghĩa là 2 câu càng giống nhau về mặt ý nghĩa
    cosine_score = util.cos_sim(embedding_A, embedding_B).item()
    
    # In ra màn hình (chuyển sang hệ số 0-1 cho dễ nhìn)
    print(f"Cặp {i+1}:")
    print(f"  Câu A: {sentences_A[i]}")
    print(f"  Câu B: {sentences_B[i]}")
    print(f"  -> Điểm tương tự: {cosine_score:.4f} ({(cosine_score * 100):.2f}%)\n")