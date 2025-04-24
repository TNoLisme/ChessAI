# Dự án AI Chess

  Dự án này triển khai một web chơi cờ vua đơn giản với hai chế độ chơi với người và với AI. Với chế độ đánh với AI, chúng tôi sử dụng kỹ thuật học sâu để dự đoán nước đi tiếp theo trong một ván cờ. AI được xây dựng bằng TensorFlow và tích hợp với ứng dụng web Flask để tương tác với người dùng. Dự án bao gồm các bước tiền xử lý dữ liệu, xây dựng mô hình, huấn luyện mô hình, đánh giá và cơ chế lưu lịch sử ván đấu để tiếp tục huấn luyện.

## Mục lục
- [Tổng quan dự án](#tổng-quan-dự-án)
- [Quy trình xử lý dữ liệu](#quy-trình-xử-lý-dữ-liệu)
  - [Nguồn dữ liệu](#nguồn-dữ-liệu)
  - [Tiền xử lý dữ liệu](#tiền-xử-lý-dữ-liệu)
- [Kiến trúc mô hình](#kiến-trúc-mô-hình)
  - [Tại sao chọn kiến trúc này?](#tại-sao-chọn-kiến-trúc-này)
  - [Tổng quan kiến trúc](#tổng-quan-kiến-trúc)
- [Quy trình huấn luyện](#quy-trình-huấn-luyện)
- [Tích hợp lịch sử ván đấu](#tích-hợp-lịch-sử-ván-đấu)
- [Cài đặt và thiết lập](#cài-đặt-và-thiết-lập)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cải tiến trong tương lai](#cải-tiến-trong-tương-lai)
- [Đóng góp](#đóng-góp)
- [Giấy phép](#giấy-phép)

## Tổng quan dự án
  AI Chess là một mô hình AI dự đoán nước đi tiếp theo dựa trên trạng thái bàn cờ, sử dụng mạng nơ-ron tích chập (CNN) kết hợp với các tầng hồi tiếp và dày đặc. Hệ thống xử lý các ván cờ được lưu trong định dạng CSV, chuyển đổi chúng thành định dạng phù hợp để huấn luyện, và cung cấp dự đoán thông qua giao diện web dựa trên Flask. Sau mỗi ván đấu, lịch sử nước đi được lưu lại và có thể được sử dụng để bổ sung dữ liệu huấn luyện.

## Quy trình xử lý dữ liệu

### Nguồn dữ liệu
  Dữ liệu thô được lấy từ trên Kaggle.com chứa khoảng 270 nghìn các ván cờ, mỗi ván có nhiều thông tin về ván, cụ thể ở đây chúng tôi quan tâm là chuỗi các nước đi, dùng để làm dữ liệu đầu vào cho mô hình.

### Tiền xử lý dữ liệu
  Quy trình tiền xử lý, được triển khai trong `process_csv_to_npy.py`, chuyển đổi dữ liệu ván cờ thô thành định dạng phù hợp cho học máy:

1. **Phân tích nước đi**: Lớp `ChessDataProcessor` đọc tệp CSV và phân tích chuỗi nước đi của mỗi ván cờ bằng thư viện `python-chess`. Các nước đi được xác thực để đảm bảo tính hợp lệ.
2. **Biểu diễn bàn cờ**: Bàn cờ được chuyển thành mảng 3D (8x8x12), trong đó:
   - 8x8 biểu thị lưới bàn cờ.
   - 12 kênh mã hóa loại quân cờ (6 cho quân trắng, 6 cho quân đen) bằng mã hóa one-hot.
3. **Đặc trưng trạng thái ván cờ**: Với mỗi nước đi, các đặc trưng sau được trích xuất:
   - Trạng thái bàn cờ (mảng 8x8x12).
   - Bên đi (0 cho trắng, 1 cho đen).
   - Quyền nhập thành (4 giá trị nhị phân cho trắng/đen nhập thành gần/xa).
   - Lịch sử nước đi (8 nước đi trước, mỗi nước mã hóa thành [from_square, to_square, captured_piece, special_move]).
   - Số nước đi (được chuẩn hóa).
   - Giai đoạn ván cờ (mã hóa one-hot: khai cuộc, trung cuộc, tàn cuộc dựa trên tổng giá trị vật chất).
4. **Nhãn đầu ra**: Đầu ra mục tiêu cho mỗi nước đi là một bộ ba (from_square, to_square, is_promotion), trong đó các ô là chỉ số (0-63) và is_promotion là nhị phân.
5. **Lưu dữ liệu**: Dữ liệu đã xử lý được lưu dưới dạng tệp `.npy`. Trong đó chúng tôi chia ra thành các tập train-val-test với tỉ lệ 8-1-1 trong các thư mục `data/train`, `data/val`, và `data/test` để tải hiệu quả trong quá trình huấn luyện.

  Lớp `ChessDataLoader` (`chess_data_loader.py`) quản lý việc tải các tệp `.npy` này theo batch, hỗ trợ xáo trộn và tạo generator để cung cấp dữ liệu cho huấn luyện.

## Kiến trúc mô hình

### Tại sao chọn kiến trúc này?
  Kiến trúc được chọn dựa trên các yêu cầu đặc thù của bài toán cờ vua:
- **Tính không gian của bàn cờ**: Bàn cờ có cấu trúc lưới 8x8, phù hợp với mạng CNN để trích xuất các mẫu không gian (quan hệ giữa các quân cờ).
- **Lịch sử nước đi**: Các nước đi trước ảnh hưởng đến chiến lược, vì vậy tầng GRU được sử dụng để xử lý thông tin thời gian.
- **Metadata phức tạp**: Các yếu tố như quyền nhập thành, giai đoạn ván cờ, và số nước đi được xử lý bằng MLP để bổ sung thông tin ngữ cảnh.
- **Đầu ra đa dạng**: Dự đoán nước đi yêu cầu xác định ô nguồn, ô đích, và khả năng phong cấp, dẫn đến việc sử dụng nhiều đầu ra softmax và sigmoid.
- **Khả năng mở rộng**: Kiến trúc ResNet với các khối residual giúp mô hình học sâu hơn mà không gặp vấn đề vanishing gradient.

  Kiến trúc này cân bằng giữa độ phức tạp tính toán và hiệu suất, phù hợp với dữ liệu cờ vua có cấu trúc cao.

### Tổng quan kiến trúc
  Mô hình, được triển khai trong `chess_model.py`, bao gồm các thành phần chính:
1. **Xử lý bàn cờ**:
   - Đầu vào: Mảng 8x8x12.
   - Tầng CNN với 64 bộ lọc, tiếp theo là 8 khối residual (mỗi khối có 2 Conv2D với BatchNorm và ReLU).
   - Global Average Pooling để giảm chiều dữ liệu.
2. **Xử lý lịch sử nước đi**:
   - Đầu vào: Mảng (8, 4) cho 8 nước đi trước.
   - Tầng GRU với 32 đơn vị, tiếp theo là tầng Dense với 16 đơn vị.
3. **Xử lý metadata**:
   - Đầu vào: Quyền nhập thành (4), bên đi (1), số nước đi (1), giai đoạn ván cờ (3).
   - MLP với hai tầng Dense (16 và 8 đơn vị).
4. **Kết hợp đặc trưng**:
   - Các đặc trưng từ bàn cờ, lịch sử, và metadata được nối lại.
   - Hai tầng Dense (64 đơn vị) với BatchNorm, ReLU, và Dropout (0.3).
5. **Đầu ra**:
   - `from_square`: Softmax cho 64 ô (chọn ô nguồn).
   - `to_square`: Softmax cho 64 ô (chọn ô đích).
   - `is_promotion`: Sigmoid cho khả năng phong cấp.

  Mô hình sử dụng L2 regularization (1e-4) và được tối ưu bằng Adam với các hàm mất mát:
- Categorical Crossentropy cho `from_square` và `to_square`.
- Binary Crossentropy cho `is_promotion`.

## Quy trình huấn luyện
  Quy trình huấn luyện, được triển khai trong `train_model.py` và `chess_model_trainer.py`, bao gồm các bước sau:
1. **Chuẩn bị dữ liệu**:
   - Tải dữ liệu từ các thư mục `data/train`, `data/val`, `data/test` bằng `ChessDataLoader`.
   - Tiền xử lý dữ liệu bằng `ChessDataPreprocessor`, chuyển đổi thành TensorFlow Dataset với batch size 64.
2. **Huấn luyện**:
   - Số epoch: 20-50 (phụ thuộc vào quá trình train và cấu hình).
   - Callbacks:
     - `ReduceLROnPlateau`: Giảm learning rate khi val_loss không cải thiện.
     - `ModelCheckpoint`: Lưu mô hình tốt nhất tại `models/chess_model_best.keras`.
     - `EarlyStopping`: Dừng sớm nếu val_loss không cải thiện sau 10 epoch.
3. **Đánh giá**:
   - Đánh giá trên tập validation và test sau khi huấn luyện, báo cáo các chỉ số như accuracy và top-3 accuracy.
4. **Lưu mô hình**:
   - Mô hình cuối cùng được lưu tại `models/chess_model_final.keras`.

  Mô hình cũng được đánh giá Elo trong `evaluate_elo.py` bằng cách đấu với Stockfish. Elo được tính dựa trên kết quả 20-40 ván (thắng: 1, hòa: 0.5, thua: 0).

## Tích hợp lịch sử ván đấu
  Sau mỗi ván chơi trên giao diện web, lịch sử nước đi được lưu vào tệp `data/games.csv` thông qua endpoint `/save_game` trong `app.py`. Các bước:
1. **Lưu lịch sử**:
   - Nhận chuỗi nước đi (ví dụ: "e2e4 e7e5 ...") từ client.
   - Kiểm tra định dạng nước đi bằng regex (`[a-h][1-8][a-h][1-8]`).
   - Ghi vào CSV với `game_id` tăng dần.
2. **Tái sử dụng dữ liệu**:
   - Lịch sử ván đấu có thể được xử lý lại bằng `process_csv_to_npy.py` để bổ sung vào tập huấn luyện, cải thiện mô hình qua thời gian.

## Cài đặt và thiết lập
  Để có thể triển khai dự án từ đầu, hãy làm theo các bước sau:

1. **Yêu cầu hệ thống**:
   - Python 3.10.x hoặc cao hơn, có hỗ trợ tensorflow.
   - Các thư viện cần thiết: `tensorflow`, `numpy`, `pandas`, `python-chess`, `flask`, `tqdm`, `matplotlib`.
   - Stockfish (dùng để đánh giá Elo, cần cài đặt và cập nhật đường dẫn trong `evaluate_elo.py`).

2. **Cài đặt môi trường**:
   - Tạo môi trường ảo (khuyến nghị):
     ```bash
     python -m venv venv
     source venv/bin/activate  # Linux/Mac
     venv\Scripts\activate     # Windows
     ```
   - Cài đặt các thư viện:
     ```bash
     pip install tensorflow numpy pandas python-chess flask psutil tqdm matplotlib
     ```

3. **Chuẩn bị dữ liệu**:
   - Đặt tệp `pgn_chess_data.csv` vào thư mục `src/ai/data`.
   - Chạy script tiền xử lý để tạo các tệp `.npy`:
     ```bash
     python src/ai/process_csv_to_npy.py
     ```
     Script này sẽ xử lý dữ liệu CSV và lưu các tệp `.npy` vào `src/ai/data/train`, `src/ai/data/val`, và `src/ai/data/test`.

4. **Tải Stockfish**:
   - Tải Stockfish từ [trang chính thức](https://stockfishchess.org/download/) và đặt vào thư mục `src/lib`.
   - Cập nhật đường dẫn tới tệp thực thi Stockfish trong `evaluate_elo.py` (ví dụ: `stockfish_path = "src/lib/stockfish.exe"`).

5. **Huấn luyện mô hình**:
   - Chạy script huấn luyện:
     ```bash
     python src/ai/train_model.py
     ```
     - Mô hình sẽ được huấn luyện trong 20 epoch và lưu tại `src/ai/models/chess_model_best.keras` (mô hình tốt nhất) và `src/ai/models/chess_model_final.keras` (mô hình cuối cùng).

6. **Chạy ứng dụng web**:
   - Khởi động server Flask:
     ```bash
     python src/ai/app.py
     ```
   - Truy cập giao diện web tại `http://localhost:5000` để chơi cờ với AI.

## Hướng dẫn sử dụng
- **Chơi cờ với AI**:
  - Mở trình duyệt và truy cập `http://localhost:5000`.
  - Sử dụng giao diện web để thực hiện nước đi. AI sẽ trả lời bằng cách dự đoán nước đi qua endpoint `/get_ai_move`.
- **Lưu ván cờ**:
  - Sau mỗi ván, lịch sử nước đi được tự động lưu vào `src/ai/data/games.csv`.
- **Đánh giá Elo**:
  - Chạy script đánh giá để so sánh AI với Stockfish:
    ```bash
    python src/ai/evaluate_elo.py
    ```
  - Script sẽ chạy 20 ván và tính toán Elo của AI dựa trên kết quả.

## Cấu trúc thư mục
```
chess-ai/
├── src/
│   ├── ai/
│   │   ├── data/                   # Thư mục chứa dữ liệu train/val/test và games.csv
│   │   ├── models/                 # Thư mục lưu mô hình đã huấn luyện
│   │   ├── app.py                  # Ứng dụng web Flask
│   │   ├── chess_data_loader.py    # Tải dữ liệu .npy
│   │   ├── chess_data_preprocessor.py # Tiền xử lý dữ liệu
│   │   ├── chess_model.py          # Kiến trúc mô hình AI
│   │   ├── chess_model_trainer.py  # Quản lý huấn luyện mô hình
│   │   ├── evaluate_elo.py         # Đánh giá Elo so với Stockfish
│   │   ├── process_csv_to_npy.py   # Chuyển đổi CSV thành .npy
│   │   ├── train_model.py          # Script khởi động huấn luyện
│   ├── index.html                  # Giao diện web
│   ├── lib/                        # Thư viện bên ngoài (Stockfish,...)
```

## Cải tiến trong tương lai
- Tích hợp thêm dữ liệu ván cờ từ các nguồn trực tuyến để tăng cường tập huấn luyện.
- Tối ưu hóa mô hình bằng cách sử dụng kiến trúc nhẹ hơn hoặc áp dụng kỹ thuật pruning.
- Thêm tính năng phân tích ván cờ, đề xuất chiến lược hoặc giải thích nước đi của AI.
- Hỗ trợ chế độ chơi giữa hai người hoặc đấu với AI ở nhiều cấp độ khó khác nhau.
