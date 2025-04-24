#♟️ ChessAI
ChessAI là một dự án trí tuệ nhân tạo chơi cờ vua được phát triển bằng Python. Dự án sử dụng TensorFlow để huấn luyện mô hình AI và tích hợp với Stockfish để đánh giá nước đi. Ngoài ra, dự án còn hỗ trợ kết nối với Lichess thông qua API để thi đấu trực tuyến và đánh giá mức Elo của AI.

📌 Tính năng
Huấn luyện mô hình AI chơi cờ vua bằng TensorFlow.

Sử dụng Stockfish để đánh giá chất lượng nước đi.

Kết nối với Lichess thông qua API để thi đấu và đánh giá Elo.

Hỗ trợ chơi cờ trực tiếp với người dùng hoặc các bot khác.

🛠️ Cài đặt
Yêu cầu hệ thống
Python 3.12

pip

Cài đặt môi trường ảo và các thư viện cần thiết

# Cài đặt các thư viện cần thiết
pip install tensorflow
pip install numpy


# Tạo và kích hoạt môi trường ảo
python -m venv venv
venv\Scripts\activate  # Trên Windows
# source venv/bin/activate  # Trên macOS/Linux





python src/ai/lichess_bot.py
🧠 Kiến trúc dự án
src/ai/app.py: Tập tin chính để huấn luyện mô hình AI.

src/ai/lichess_bot.py: Kết nối và thi đấu trên Lichess.

src/ai/stockfish_eval.py: Đánh giá nước đi bằng Stockfish.

src/ai/utils.py: Các hàm tiện ích hỗ trợ.

📝 Góp ý và đóng góp
Chúng tôi hoan nghênh mọi đóng góp từ cộng đồng. Nếu bạn muốn đóng góp, hãy fork dự án và gửi pull request. Mọi ý kiến và phản hồi đều được trân trọng.
