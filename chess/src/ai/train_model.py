import numpy as np
import os
from chess_model_trainer import ChessModelTrainer
from chess_model import ChessModel
from chess_data_loader import ChessDataLoader

def estimate_total_samples(loader: ChessDataLoader, dataset_type: str) -> int:
    """
    Ước lượng tổng số mẫu dữ liệu từ các file .npy trong thư mục tương ứng.

    Args:
        loader (ChessDataLoader): Loader để load danh sách file.
        dataset_type (str): Loại dataset ('train', 'val', 'test').

    Returns:
        int: Tổng số mẫu dữ liệu trong tất cả các file .npy.
    """
    total_samples = 0
    for file in loader.list_data_files(dataset_type):
        folder = loader.train_folder if dataset_type == "train" else loader.val_folder if dataset_type == "val" else loader.test_folder
        # Load dữ liệu từ file
        data = np.load(os.path.join(folder, file), allow_pickle=True).item()
        # Cộng tổng số lượng mẫu
        total_samples += len(data['X'])
    return total_samples

def main():
    # Đường dẫn tuyệt đối đến thư mục gốc của dự án
    base_dir = "D:/AI chess/chess2/src/ai"
    
    # Đường dẫn đến các thư mục dữ liệu
    train_folder = os.path.join(base_dir, 'data', 'train')
    val_folder = os.path.join(base_dir, 'data', 'val')
    test_folder = os.path.join(base_dir, 'data', 'test')
    model_save_dir = os.path.join(base_dir, 'src', 'ai', 'models')

    # Kiểm tra các thư mục dữ liệu đã tồn tại và chứa file .npy
    for folder, name in [(train_folder, "train"), (val_folder, "val"), (test_folder, "test")]:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"{name.capitalize()} folder not found at {folder}")
        npy_files = [f for f in os.listdir(folder) if f.endswith('.npy')]
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {folder}")
        print(f"📂 Found {len(npy_files)} .npy files in {folder}")

    # Ước lượng tổng số mẫu trong train, val, test
    loader = ChessDataLoader(train_folder=train_folder, val_folder=val_folder, test_folder=test_folder)
    train_samples = estimate_total_samples(loader, "train")
    val_samples = estimate_total_samples(loader, "val")
    test_samples = estimate_total_samples(loader, "test")
    print(f"📊 Estimated data samples: Train: {train_samples}, Val: {val_samples}, Test: {test_samples}")

    # Khởi tạo trainer
    trainer = ChessModelTrainer(
        train_folder=train_folder,
        val_folder=val_folder,
        test_folder=test_folder,
        model_save_dir=model_save_dir,
        history_length=8
    )

    # Khởi tạo mô hình
    model = None
    model_path = os.path.join(model_save_dir, 'chess_model_best.keras')

    # Nếu có mô hình cũ, load để tiếp tục training
    if os.path.exists(model_path):
        print(f"⚠️ Loading existing model from {model_path} for continued training...")
        try:
            model = ChessModel()
            model.load(model_path)
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}. Initializing new model...")
            model = ChessModel()
    else:
        # Nếu không có mô hình cũ, khởi tạo mô hình mới
        print(f"⚠️ Model file not found at {model_path}. Training a new model from scratch...")

    # Bắt đầu quá trình huấn luyện
    print("🔄 Starting training process...")
    model = trainer.train_model(
        model=model,
        epochs=1,
        batch_size=64
    )

# Entry point của chương trình
if __name__ == "__main__":
    main()
