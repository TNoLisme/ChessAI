import numpy as np
import tensorflow as tf
import chess
from typing import List, Tuple, Dict, Optional, Iterator

class ChessDataPreprocessor:
    """
    Class tiền xử lý dữ liệu cờ vua để chuẩn bị cho mô hình học máy.
    """

    def __init__(self, history_length: int = 8):
        """
        Khởi tạo bộ tiền xử lý dữ liệu.

        Args:
            history_length (int): Độ dài lịch sử các nước đi được lưu giữ.
        """
        self.history_length = 8
        self.num_squares = 64

    def preprocess_input(self, X_raw: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Tiền xử lý dữ liệu đầu vào thành các mảng numpy.

        Args:
            X_raw (List[Dict]): Danh sách các ván cờ ở dạng dictionary.

        Returns:
            Dict[str, np.ndarray]: Dictionary chứa các đặc trưng đầu vào đã được chuẩn hóa.

        Raises:
            ValueError: Nếu dữ liệu đầu vào rỗng hoặc thiếu key bắt buộc.
        """
        if len(X_raw) == 0:
            raise ValueError("Input data is empty")

        # Danh sách các key bắt buộc phải có trong mỗi phần tử
        required_keys = ["board", "side_to_move", "castling_rights", "history", "move_count", "game_phase"]
        for i, data_point in enumerate(X_raw):
            if not isinstance(data_point, dict):
                raise ValueError(f"Data point {i} is not a dictionary: {data_point}")
            if not all(key in data_point for key in required_keys):
                raise ValueError(f"Missing required keys in data point {i}: {data_point}")

        n_samples = len(X_raw)

        # Tạo các mảng numpy để lưu dữ liệu
        boards = np.zeros((n_samples, 8, 8, 12), dtype=np.float32)
        side_to_moves = np.zeros((n_samples, 1), dtype=np.float32)
        castling_rights = np.zeros((n_samples, 4), dtype=np.float32)
        histories = np.zeros((n_samples, self.history_length, 4), dtype=np.float32)
        move_counts = np.zeros((n_samples, 1), dtype=np.float32)
        game_phases = np.zeros((n_samples, 3), dtype=np.float32)

        def square_to_index(square):
            """
            Chuyển đổi ký hiệu ô cờ (VD: 'e4') thành chỉ số (0-63).
            """
            if isinstance(square, (int, float)):
                return int(square)
            if not isinstance(square, str) or len(square) != 2:
                return -1
            file = ord(square[0]) - ord('a')
            rank = 8 - int(square[1])
            if 0 <= file <= 7 and 0 <= rank <= 7:
                return rank * 8 + file
            return -1

        for i, data_point in enumerate(X_raw):
            boards[i] = data_point["board"]
            side_to_moves[i, 0] = data_point["side_to_move"]
            castling_rights[i] = data_point["castling_rights"]

            # Cắt hoặc thêm lịch sử nước đi để đúng độ dài yêu cầu
            history = data_point["history"][-self.history_length:] if len(data_point["history"]) > self.history_length else data_point["history"]
            history_padded = [None] * (self.history_length - len(history)) + history

            # Chuyển mỗi nước đi thành vector 4 chiều
            for j, move in enumerate(history_padded):
                if move is not None and len(move) >= 2:
                    from_idx = square_to_index(move[0])
                    to_idx = square_to_index(move[1])
                    captured = int(move[2]) if len(move) > 2 and move[2] is not None else 0
                    special = int(move[3]) if len(move) > 3 and move[3] is not None else 0
                    if from_idx >= 0 and to_idx >= 0:
                        histories[i, j, 0] = from_idx / self.num_squares
                        histories[i, j, 1] = to_idx / self.num_squares
                        histories[i, j, 2] = captured / 12 if captured >= 0 else 0
                        histories[i, j, 3] = special

            move_counts[i, 0] = data_point["move_count"] / 200
            game_phases[i] = data_point["game_phase"]

        return {
            "board": boards,
            "side_to_move": side_to_moves,
            "castling_rights": castling_rights,
            "history": histories,
            "move_count": move_counts,
            "game_phase": game_phases
        }

    def preprocess_output(self, y_raw: List[Tuple[int, int, int]]) -> Dict[str, np.ndarray]:
        """
        Tiền xử lý dữ liệu đầu ra thành dạng one-hot encoding.

        Args:
            y_raw (List[Tuple[int, int, int]]): Danh sách nhãn (from_square, to_square, is_promotion).

        Returns:
            Dict[str, np.ndarray]: Dictionary chứa nhãn ở dạng tensor.

        Raises:
            ValueError: Nếu dữ liệu đầu ra rỗng.
        """
        if len(y_raw) == 0:
            raise ValueError("Output data is empty")

        n_samples = len(y_raw)
        from_squares = np.zeros((n_samples, self.num_squares), dtype=np.float32)
        to_squares = np.zeros((n_samples, self.num_squares), dtype=np.float32)
        is_promotions = np.zeros((n_samples, 1), dtype=np.float32)

        for i, (from_sq, to_sq, is_promotion) in enumerate(y_raw):
            from_squares[i, from_sq] = 1.0
            to_squares[i, to_sq] = 1.0
            is_promotions[i, 0] = is_promotion

        return {
            "from_square": from_squares,
            "to_square": to_squares,
            "is_promotion": is_promotions
        }

    def create_tensorflow_dataset(self, X_raw: List[Dict], y_raw: List[Tuple[int, int, int]], 
                                  batch_size: int = 32, shuffle: bool = True, buffer_size: int = 10000) -> tf.data.Dataset:
        """
        Tạo TensorFlow dataset từ dữ liệu thô.

        Args:
            X_raw (List[Dict]): Dữ liệu đầu vào thô.
            y_raw (List[Tuple[int, int, int]]): Nhãn đầu ra thô.
            batch_size (int): Kích thước mỗi batch.
            shuffle (bool): Có xáo trộn dữ liệu hay không.
            buffer_size (int): Bộ đệm cho shuffle.

        Returns:
            tf.data.Dataset: Dataset huấn luyện.
        """
        X_processed = self.preprocess_input(X_raw)
        y_processed = self.preprocess_output(y_raw)
        dataset = tf.data.Dataset.from_tensor_slices((X_processed, y_processed))
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def create_tensorflow_dataset_from_generator(self, data_generator: Iterator[Tuple[List[Dict], List[Tuple[int, int, int]]]], 
                                                batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
        """
        Tạo TensorFlow dataset từ generator (dành cho dữ liệu lớn hoặc streaming).

        Args:
            data_generator (Iterator): Generator sinh ra các batch (X_batch, y_batch).
            batch_size (int): Kích thước mỗi batch.
            shuffle (bool): Có xáo trộn dataset cuối cùng không.

        Returns:
            tf.data.Dataset: Dataset được tạo từ generator.
        """
        batch_count = 0

        def generator():
            """
            Generator con xử lý từng batch và yield từng mẫu riêng lẻ.
            """
            nonlocal batch_count
            for X_batch, y_batch in data_generator:
                if len(X_batch) == 0 or len(y_batch) == 0:
                    print(f"Warning: Empty batch encountered at batch {batch_count}, skipping...")
                    continue
                if len(X_batch) != len(y_batch):
                    raise ValueError(f"Mismatch between X_batch ({len(X_batch)}) and y_batch ({len(y_batch)}) at batch {batch_count}")
                X_processed = self.preprocess_input(X_batch)
                y_processed = self.preprocess_output(y_batch)
                batch_count += 1
                if batch_count % 100 == 0:
                    print(f"Processed {batch_count} batches...")
                for i in range(len(X_batch)):
                    yield (
                        {
                            "board": X_processed["board"][i],
                            "side_to_move": X_processed["side_to_move"][i],
                            "castling_rights": X_processed["castling_rights"][i],
                            "history": X_processed["history"][i],
                            "move_count": X_processed["move_count"][i],
                            "game_phase": X_processed["game_phase"][i]
                        },
                        {
                            "from_square": y_processed["from_square"][i],
                            "to_square": y_processed["to_square"][i],
                            "is_promotion": y_processed["is_promotion"][i]
                        }
                    )

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                {
                    "board": tf.TensorSpec(shape=(8, 8, 12), dtype=tf.float32),
                    "side_to_move": tf.TensorSpec(shape=(1,), dtype=tf.float32),
                    "castling_rights": tf.TensorSpec(shape=(4,), dtype=tf.float32),
                    "history": tf.TensorSpec(shape=(self.history_length, 4), dtype=tf.float32),
                    "move_count": tf.TensorSpec(shape=(1,), dtype=tf.float32),
                    "game_phase": tf.TensorSpec(shape=(3,), dtype=tf.float32)
                },
                {
                    "from_square": tf.TensorSpec(shape=(self.num_squares,), dtype=tf.float32),
                    "to_square": tf.TensorSpec(shape=(self.num_squares,), dtype=tf.float32),
                    "is_promotion": tf.TensorSpec(shape=(1,), dtype=tf.float32)
                }
            )
        )
        if shuffle:
            dataset = dataset.shuffle(10000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        print(f"Created dataset with batch_size={batch_size}, shuffle={shuffle}")
        return dataset
