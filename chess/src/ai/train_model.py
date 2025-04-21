import os
from chess_model_trainer import ChessModelTrainer
from chess_model import ChessModel

def main():
    # Đường dẫn tuyệt đối đến thư mục gốc của dự án
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Khởi tạo trainer với đường dẫn tuyệt đối
    data_folder = os.path.join(base_dir, 'ai/data')
    model_save_dir = os.path.join(base_dir, 'ai', 'models')

    # Kiểm tra xem thư mục data có tồn tại và chứa file .npy không
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder not found at {data_folder}")
    npy_files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {data_folder}")

    trainer = ChessModelTrainer(
        data_folder=data_folder,
        model_save_dir=model_save_dir,
        history_length=8
    )

    # Tải mô hình có sẵn để tiếp tục huấn luyện
    model_path = os.path.join(model_save_dir, 'chess_model_best.h5')  # Đường dẫn đến file mô hình đã lưu
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print(f"⚠️ Loading existing model from {model_path} for continued training...")
    model = ChessModel()
    model.load(model_path)  # Tải mô hình đã lưu

    # Huấn luyện tiếp tục
    print("🔄 Continuing training process...")
    model = trainer.train_model(
        model=model,
        epochs=1,
        batch_size=32
    )

    # Đánh giá mô hình
    print("🔍 Evaluating the trained model...")
    metrics = trainer.evaluate_model(model)
    print("✅ Evaluation completed. Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

if __name__ == "__main__":
    main()