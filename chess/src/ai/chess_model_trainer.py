import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import psutil
from typing import Dict, Tuple, List, Optional, Iterator

from chess_data_loader import ChessDataLoader
from chess_data_preprocessor import ChessDataPreprocessor
from chess_model import ChessModel

class ChessModelTrainer:
    """
    Class quản lý huấn luyện và đánh giá mô hình AI cờ vua.
    """
    
    def __init__(self, train_folder: str = "data/train", val_folder: str = "data/val", 
                 test_folder: str = "data/test", model_save_dir: str = "models", history_length: int = 8):
        """
        Khởi tạo trainer với các thư mục dữ liệu và tham số.
        """
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.test_folder = test_folder
        self.model_save_dir = model_save_dir
        self.history_length = history_length
        self.data_loader = ChessDataLoader(
            train_folder=train_folder,
            val_folder=val_folder,
            test_folder=test_folder
        )
        self.preprocessor = ChessDataPreprocessor(history_length=self.history_length)

    def prepare_train_dataset(self, batch_size: int = 16, shuffle: bool = True) -> Tuple[tf.data.Dataset, int]:
        """
        Chuẩn bị dataset cho tập huấn luyện.
        
        Returns:
            Tuple[tf.data.Dataset, int]: Dataset và số steps per epoch.
        """
        print("🔄 Creating train data generator...")
        train_generator = self.data_loader.load_data_generator(
            batch_size=1000, shuffle=shuffle, dataset_type="train"
        )
        
        # Ước lượng số mẫu
        total_samples = sum(len(data['X']) for file in self.data_loader.list_data_files("train")
                           for data in [np.load(os.path.join(self.train_folder, file), allow_pickle=True).item()])
        steps_per_epoch = (total_samples + batch_size - 1) // batch_size
        print(f"Train dataset: {total_samples} samples, {steps_per_epoch} steps per epoch")

        train_ds = self.preprocessor.create_tensorflow_dataset_from_generator(
            train_generator, batch_size, shuffle=shuffle
        )
        print("Train dataset created")
        return train_ds, steps_per_epoch

    def prepare_validation_dataset(self, batch_size: int = 16, shuffle: bool = False) -> Tuple[tf.data.Dataset, int]:
        """
        Chuẩn bị dataset cho tập validation.
        
        Returns:
            Tuple[tf.data.Dataset, int]: Dataset và số validation steps.
        """
        print("Creating validation data generator...")
        val_generator = self.data_loader.load_data_generator(
            batch_size=1000, shuffle=shuffle, dataset_type="val"
        )
        
        # Ước lượng số mẫu
        total_samples = sum(len(data['X']) for file in self.data_loader.list_data_files("val")
                           for data in [np.load(os.path.join(self.val_folder, file), allow_pickle=True).item()])
        validation_steps = (total_samples + batch_size - 1) // batch_size if total_samples > 0 else 1
        print(f"Validation dataset: {total_samples} samples, {validation_steps} validation steps")

        val_ds = self.preprocessor.create_tensorflow_dataset_from_generator(
            val_generator, batch_size, shuffle=shuffle
        ).cache()
        print("Validation dataset created")
        return val_ds, validation_steps

    def prepare_test_dataset(self, batch_size: int = 16, shuffle: bool = False) -> Tuple[tf.data.Dataset, int]:
        """
        Chuẩn bị dataset cho tập test.
        
        Returns:
            Tuple[tf.data.Dataset, int]: Dataset và số test steps.
        """
        print("Creating test data generator...")
        test_generator = self.data_loader.load_data_generator(
            batch_size=1000, shuffle=shuffle, dataset_type="test"
        )
        
        # Ước lượng số mẫu
        total_samples = sum(len(data['X']) for file in self.data_loader.list_data_files("test")
                           for data in [np.load(os.path.join(self.test_folder, file), allow_pickle=True).item()])
        test_steps = (total_samples + batch_size - 1) // batch_size if total_samples > 0 else 1
        print(f"Test dataset: {total_samples} samples, {test_steps} test steps")

        test_ds = self.preprocessor.create_tensorflow_dataset_from_generator(
            test_generator, batch_size, shuffle=shuffle
        ).cache()
        print("Test dataset created")
        return test_ds, test_steps

    def train_model(self, model: Optional[ChessModel] = None, epochs: int = 100, batch_size: int = 16) -> ChessModel:
        """
        Huấn luyện mô hình AI cờ vua, sau đó đánh giá trên tập validation và test.
        """
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        print("Initializing new model...")
        model = ChessModel() if model is None else model
        print("Model initialized")

        # Chuẩn bị dataset huấn luyện và validation
        train_ds, steps_per_epoch = self.prepare_train_dataset(batch_size=batch_size, shuffle=True)
        val_ds, validation_steps = self.prepare_validation_dataset(batch_size=batch_size, shuffle=False)
        
        # Callback cho huấn luyện
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_save_dir, 'chess_model_best.keras'),
                monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1
            )
        ]

        print(f"🏋️ Starting training with {epochs} epochs, {steps_per_epoch} steps per epoch, {validation_steps} validation steps...")
        mem_info = psutil.virtual_memory()
        print(f"RAM usage before training: {mem_info.percent}% ({mem_info.used/1024**3:.2f}GB / {mem_info.total/1024**3:.2f}GB)")
        
        history = model.train(
            train_dataset=train_ds,
            validation_dataset=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps
        )
        print("Training completed")

        mem_info = psutil.virtual_memory()
        print(f"RAM usage after training: {mem_info.percent}% ({mem_info.used/1024**3:.2f}GB / {mem_info.total/1024**3:.2f}GB)")

        final_model_path = os.path.join(self.model_save_dir, 'chess_model_final.keras')
        model.save(final_model_path)
        print(f"Saved final model at: {final_model_path}")

        # Đánh giá trên tập validation
        print("Validating model...")
        val_metrics = self.validate_model(model, batch_size=batch_size)
        print("Validation completed. Metrics:")
        for name, value in val_metrics.items():
            print(f"  {name}: {value:.4f}")

        # Đánh giá trên tập test
        print("Testing model...")
        test_metrics = self.test_model(model, batch_size=batch_size)
        print("Test completed. Metrics:")
        for name, value in test_metrics.items():
            print(f"  {name}: {value:.4f}")

        return model

    def validate_model(self, model: Optional[ChessModel] = None, batch_size: int = 16) -> Dict:
        """
        Đánh giá mô hình trên tập validation.
        """
        if model is None:
            best_model_path = os.path.join(self.model_save_dir, 'chess_model_best.keras')
            model = ChessModel.load(best_model_path)
            print(f"Loaded model from: {best_model_path}")

        val_ds, validation_steps = self.prepare_validation_dataset(batch_size=batch_size)
        print("🔍 Evaluating on validation dataset...")
        metrics = model.model.evaluate(val_ds, steps=validation_steps, verbose=1)
        result = {name: value for name, value in zip(model.model.metrics_names, metrics)}
        return result

    def test_model(self, model: Optional[ChessModel] = None, batch_size: int = 16) -> Dict:
        """
        Đánh giá mô hình trên tập test.
        """
        if model is None:
            best_model_path = os.path.join(self.model_save_dir, 'chess_model_best.keras')
            model = ChessModel.load(best_model_path)
            print(f"Loaded model from: {best_model_path}")

        test_ds, test_steps = self.prepare_test_dataset(batch_size=batch_size)
        print("🔍 Evaluating on test dataset...")
        metrics = model.model.evaluate(test_ds, steps=test_steps, verbose=1)
        result = {name: value for name, value in zip(model.model.metrics_names, metrics)}
        return result