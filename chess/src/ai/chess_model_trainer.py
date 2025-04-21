import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Optional

from chess_data_loader import ChessDataLoader
from chess_data_preprocessor import ChessDataPreprocessor
from chess_model import ChessModel

class ChessModelTrainer:
    """
    Class quản lý huấn luyện và đánh giá mô hình AI cờ vua.
    """
    
    def __init__(self, data_folder: str = "data", model_save_dir: str = "models", history_length: int = 8):
        """
        Khởi tạo trainer.

        Args:
            data_folder (str): Thư mục chứa dữ liệu .npy.
            model_save_dir (str): Thư mục lưu mô hình.
            history_length (int): Độ dài lịch sử nước đi.
        """
        self.data_folder = data_folder
        self.model_save_dir = model_save_dir
        self.history_length = history_length
        # os.makedirs(model_save_dir, exist_ok=True)
        self.data_loader = ChessDataLoader(data_folder=data_folder)
        self.preprocessor = ChessDataPreprocessor(history_length=history_length)

    def prepare_datasets(self, batch_size: int = 32, val_split: float = 0.1, test_split: float = 0.1, 
                         shuffle: bool = True) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Chuẩn bị các dataset cho huấn luyện, kiểm định, và kiểm tra.

        Args:
            batch_size (int): Kích thước batch.
            val_split (float): Tỷ lệ dữ liệu kiểm định.
            test_split (float): Tỷ lệ dữ liệu kiểm tra.
            shuffle (bool): Có xáo trộn dữ liệu hay không.

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Dataset huấn luyện, kiểm định, và kiểm tra.
        """
        print("🔄 Loading data...")
        X_raw, y_raw = [], []
        data_generator = self.data_loader.load_data_generator(batch_size=1000, shuffle=True)
        for X_batch, y_batch in data_generator:
            X_raw.extend(X_batch.tolist())
            y_raw.extend(y_batch.tolist())
        print(f"✅ Loaded {len(X_raw)} data samples")

        # Thêm giá trị value giả (0.5) vì dữ liệu hiện tại không có nhãn value
        # Trong thực tế, cần tạo nhãn value (ví dụ: 1 nếu thắng, 0 nếu hòa, -1 nếu thua)
        y_raw_with_value = [(y[0], y[1], y[2], 0.5) for y in y_raw]

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_raw, y_raw_with_value, test_size=test_split, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_split/(1-test_split), random_state=42
        )
        print(f"📊 Data split: Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        print("🔄 Preprocessing and creating datasets...")
        train_ds = self.preprocessor.create_tensorflow_dataset(X_train, y_train, batch_size, shuffle)
        val_ds = self.preprocessor.create_tensorflow_dataset(X_val, y_val, batch_size, shuffle=False)
        test_ds = self.preprocessor.create_tensorflow_dataset(X_test, y_test, batch_size, shuffle=False)
        print("✅ Datasets created")
        return train_ds, val_ds, test_ds

    def train_model(self, model: Optional[ChessModel] = None, epochs: int = 100, batch_size: int = 32) -> ChessModel:
        """
        Huấn luyện mô hình AI cờ vua.

        Args:
            model (Optional[ChessModel]): Mô hình cần huấn luyện (tạo mới nếu None).
            epochs (int): Số epoch huấn luyện.
            batch_size (int): Kích thước batch.

        Returns:
            ChessModel: Mô hình đã huấn luyện.
        """
        train_ds, val_ds, _ = self.prepare_datasets(batch_size=batch_size)
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        print("🔄 Initializing new model...")
        model = ChessModel()
        print("✅ Model initialized")

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_save_dir, 'chess_model_best.h5'),
                monitor='val_loss', save_best_only=True, verbose=1
            )
        ]

        print(f"🏋️ Starting training with {epochs} epochs...")
        history = model.train(train_ds, val_ds, epochs, callbacks)
        print("✅ Training completed")

        final_model_path = os.path.join(self.model_save_dir, 'chess_model_final.h5')
        model.save(final_model_path)
        print(f"💾 Saved final model at: {final_model_path}")

        self._plot_training_history(history)
        return model
       
    def _plot_training_history(self, history: Dict) -> None:
        """
        Vẽ và lưu biểu đồ lịch sử huấn luyện.

        Args:
            history (Dict): Dictionary chứa lịch sử huấn luyện.
        """
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(history['from_square_accuracy'], label='Train From Acc')
        plt.plot(history['val_from_square_accuracy'], label='Val From Acc')
        plt.title('From Square Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(history['to_square_accuracy'], label='Train To Acc')
        plt.plot(history['val_to_square_accuracy'], label='Val To Acc')
        plt.title('To Square Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(history['value_mae'], label='Train Value MAE')
        plt.plot(history['val_value_mae'], label='Val Value MAE')
        plt.title('Value MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_dir, 'training_history.png'))
        print("📊 Saved training history plot")

    def evaluate_model(self, model: Optional[ChessModel] = None) -> Dict:
        """
        Đánh giá hiệu suất mô hình trên tập kiểm tra.

        Args:
            model (Optional[ChessModel]): Mô hình cần đánh giá (tải từ file nếu None).

        Returns:
            Dict: Dictionary chứa các metric đánh giá.
        """
        if model is None:
            best_model_path = os.path.join(self.model_save_dir, 'chess_model_best.h5')
            model = ChessModel.load(best_model_path)
            print(f"✅ Loaded model from: {best_model_path}")

        _, _, test_ds = self.prepare_datasets()
        print("🔍 Evaluating model...")
        metrics = model.model.evaluate(test_ds, verbose=1)
        result = {name: value for name, value in zip(model.model.metrics_names, metrics)}
        for name, value in result.items():
            print(f"📈 {name}: {value:.4f}")
        return result