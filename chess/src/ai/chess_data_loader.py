import os
import numpy as np
from typing import List, Tuple, Iterator

class ChessDataLoader:
    """
    Class quản lý việc load dữ liệu từ các file .npy trong các thư mục train, val, test.
    """

    def __init__(self, train_folder: str = "data/train", val_folder: str = "data/val", test_folder: str = "data/test"):
        """
        Khởi tạo loader với các thư mục train, val, test.

        Args:
            train_folder (str): Đường dẫn đến thư mục chứa file .npy cho tập train.
            val_folder (str): Đường dẫn đến thư mục chứa file .npy cho tập validation.
            test_folder (str): Đường dẫn đến thư mục chứa file .npy cho tập test.
        """
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.test_folder = test_folder

    def list_data_files(self, dataset_type: str = "train") -> List[str]:
        """
        Lấy danh sách các file .npy trong thư mục tương ứng.

        Args:
            dataset_type (str): Loại dataset ('train', 'val', 'test').

        Returns:
            List[str]: Danh sách tên file .npy.
        """
        # Chọn thư mục tương ứng theo loại dataset
        folder = self.train_folder if dataset_type == "train" else self.val_folder if dataset_type == "val" else self.test_folder
        
        # Nếu thư mục không tồn tại, raise lỗi
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder} not found")
        
        # Lấy tất cả file có đuôi .npy
        files = [f for f in os.listdir(folder) if f.endswith('.npy')]
        
        # Nếu không có file .npy nào, raise lỗi
        if not files:
            raise FileNotFoundError(f"No .npy files found in {folder}")
        
        # Trả về danh sách file đã sort
        return sorted(files)

    def load_data_generator(self, batch_size: int = 1000, shuffle: bool = True, dataset_type: str = "train") -> Iterator[Tuple[List, List]]:
        """
        Tạo generator để load dữ liệu từ file .npy theo batch.

        Args:
            batch_size (int): Kích thước batch.
            shuffle (bool): Có xáo trộn dữ liệu không.
            dataset_type (str): Loại dataset ('train', 'val', 'test').

        Yields:
            Tuple[List, List]: (X_batch, y_batch) với X_batch và y_batch là danh sách dữ liệu.
        """
        # Lấy danh sách file .npy cần load
        files = self.list_data_files(dataset_type)
        
        # Chọn thư mục tương ứng theo loại dataset
        folder = self.train_folder if dataset_type == "train" else self.val_folder if dataset_type == "val" else self.test_folder
        
        # Nếu yêu cầu shuffle file
        if shuffle:
            np.random.shuffle(files)
        
        # Duyệt từng file
        for file in files:
            file_path = os.path.join(folder, file)
            try:
                # Load file .npy
                data = np.load(file_path, allow_pickle=True).item()
                X, y = data.get('X', []), data.get('y', [])
                
                # Nếu file trống thì bỏ qua
                if not X or not y:
                    print(f"Warning: Empty data in {file_path}, skipping...")
                    continue
                
                # Tạo danh sách chỉ số
                indices = np.arange(len(X))
                
                # Nếu yêu cầu shuffle trong file
                if shuffle:
                    np.random.shuffle(indices)
                
                # Chia dữ liệu thành các batch
                for start_idx in range(0, len(X), batch_size):
                    batch_indices = indices[start_idx:start_idx + batch_size]
                    X_batch = [X[i] for i in batch_indices]
                    y_batch = [y[i] for i in batch_indices]
                    
                    # Trả về batch
                    yield X_batch, y_batch
            except Exception as e:
                # Nếu lỗi khi load file, in lỗi và bỏ qua
                print(f"Error loading {file_path}: {e}")
                continue
