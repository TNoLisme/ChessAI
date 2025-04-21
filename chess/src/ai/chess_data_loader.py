
import numpy as np
import os
from typing import List, Tuple, Dict, Optional, Iterator

class ChessDataLoader:
    """
    Class tải dữ liệu cờ vua từ các file .npy đã được xử lý.
    """
    
    def __init__(self, data_folder: str = "data"):
        """
        Khởi tạo bộ tải dữ liệu.
        Args:
            data_folder (str): Thư mục chứa các file .npy.
        """
        self.data_folder = data_folder

    def list_data_files(self) -> List[str]:
        """
        Liệt kê tất cả các file .npy trong thư mục dữ liệu.
        Returns:
            List[str]: Danh sách tên các file .npy.
        """
        return [f for f in os.listdir(self.data_folder) 
                if f.endswith('.npy') and f.startswith('chess_data_')]

    def load_data_file(self, filename: str) -> Tuple[List[Dict], List[Tuple[int, int, Optional[int]]]]:
        """
        Tải dữ liệu từ một file .npy.
        Args:
            filename (str): Tên file (không bao gồm đường dẫn).
        Returns:
            Tuple[List[Dict], List[Tuple[int, int, Optional[int]]]]: Dữ liệu đầu vào (X) và đầu ra (y).
        """
        full_path = os.path.join(self.data_folder, filename)
        data = np.load(full_path, allow_pickle=True).item()
        return data["X"], data["y"]

    def load_all_data(self) -> Tuple[List[Dict], List[Tuple[int, int, Optional[int]]]]:
        """
        Tải toàn bộ dữ liệu từ tất cả các file .npy.
        Returns:
            Tuple[List[Dict], List[Tuple[int, int, Optional[int]]]]: Toàn bộ dữ liệu đầu vào và đầu ra.
        """
        all_X, all_y = [], []
        for file in self.list_data_files():
            X, y = self.load_data_file(file)
            all_X.extend(X)
            all_y.extend(y)
        return all_X, all_y
    
    def load_data_generator(self, batch_size: int = 32, shuffle: bool = True) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Tạo generator để tải dữ liệu theo batch, tiết kiệm bộ nhớ.
        Args:
            batch_size (int): Kích thước mỗi batch.
            shuffle (bool): Có xáo trộn dữ liệu hay không.
        Yields:
            Iterator[Tuple[np.ndarray, np.ndarray]]: Batch dữ liệu đầu vào (X) và đầu ra (y).
        """
        file_list = self.list_data_files()
        if shuffle:
            np.random.shuffle(file_list)

        for file in file_list:
            X, y = self.load_data_file(file)
            indices = np.arange(len(X))
            if shuffle:
                np.random.shuffle(indices)

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                X_batch = np.array([X[j] for j in batch_indices])
                y_batch = np.array([y[j] for j in batch_indices])
                yield X_batch, y_batch
