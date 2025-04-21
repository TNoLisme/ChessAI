import numpy as np
import tensorflow as tf
import chess
from typing import List, Tuple, Dict, Optional

class ChessDataPreprocessor:
    """
    Class tiền xử lý dữ liệu cờ vua để chuẩn bị cho mô hình học máy.
    """
    
    def __init__(self, history_length: int = 8, max_elo: float = 3000):
        """
        Khởi tạo bộ tiền xử lý dữ liệu.

        Args:
            history_length (int): Số lượng nước đi lịch sử cho mỗi trạng thái.
            max_elo (float): Giá trị Elo tối đa để chuẩn hóa (mặc định 3000).
        """
        self.history_length = history_length
        self.num_squares = 64
        self.max_elo = max_elo

    def preprocess_input(self, X_raw: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Tiền xử lý dữ liệu đầu vào thành các mảng numpy.

        Args:
            X_raw (List[Dict]): Danh sách các dictionary chứa dữ liệu đầu vào thô.

        Returns:
            Dict[str, np.ndarray]: Dictionary chứa các mảng numpy đã tiền xử lý.
        """
        if not X_raw:
            raise ValueError("Input data is empty")
        required_keys = ["board", "side_to_move", "castling_rights", "history", "move_count", "game_phase", "white_elo", "black_elo"]
        for data_point in X_raw:
            if not all(key in data_point for key in required_keys):
                raise ValueError(f"Missing required keys in data point: {data_point}")

        n_samples = len(X_raw)
        boards = np.zeros((n_samples, 8, 8, 12), dtype=np.float32)
        side_to_moves = np.zeros((n_samples, 1), dtype=np.float32)
        castling_rights = np.zeros((n_samples, 4), dtype=np.float32)
        histories = np.zeros((n_samples, self.history_length, 4), dtype=np.float32)
        move_counts = np.zeros((n_samples, 1), dtype=np.float32)
        game_phases = np.zeros((n_samples, 3), dtype=np.float32)
        white_elos = np.zeros((n_samples, 1), dtype=np.float32)
        black_elos = np.zeros((n_samples, 1), dtype=np.float32)

        # Hàm chuyển đổi ô (như "g1") thành chỉ số (0-63)
        def square_to_index(square):
            if not isinstance(square, str) or len(square) != 2:
                return -1
            file = ord(square[0]) - ord('a')  # a-h -> 0-7
            rank = 8 - int(square[1])        # 1-8 -> 7-0
            if 0 <= file <= 7 and 0 <= rank <= 7:
                return rank * 8 + file
            return -1

        for i, data_point in enumerate(X_raw):
            boards[i] = data_point["board"]
            side_to_moves[i, 0] = data_point["side_to_move"]
            castling_rights[i] = data_point["castling_rights"]
            for j, move in enumerate(data_point["history"]):
                if move is not None:
                    # Chuyển đổi from_square và to_square thành chỉ số
                    from_idx = square_to_index(move[0])  # move[0] là from_square (ví dụ: "g1")
                    to_idx = square_to_index(move[1])    # move[1] là to_square (ví dụ: "f3")
                    captured = int(move[2]) if move[2] is not None else 0  # move[2] là captured
                    special = int(move[3]) if move[3] is not None else 0   # move[3] là special

                    if from_idx >= 0 and to_idx >= 0:  # Kiểm tra chỉ số hợp lệ
                        histories[i, j, 0] = from_idx / self.num_squares  # Chuẩn hóa
                        histories[i, j, 1] = to_idx / self.num_squares    # Chuẩn hóa
                        histories[i, j, 2] = captured / 12 if captured >= 0 else 0  # Chuẩn hóa
                        histories[i, j, 3] = special
                    else:
                        histories[i, j, 0] = 0
                        histories[i, j, 1] = 0
                        histories[i, j, 2] = 0
                        histories[i, j, 3] = 0
                else:
                    histories[i, j, 0] = 0
                    histories[i, j, 1] = 0
                    histories[i, j, 2] = 0
                    histories[i, j, 3] = 0
            move_counts[i, 0] = data_point["move_count"] / 200  # Chuẩn hóa
            game_phases[i] = data_point["game_phase"]
            white_elos[i, 0] = data_point["white_elo"] / self.max_elo
            black_elos[i, 0] = data_point["black_elo"] / self.max_elo

        return {
            "board": boards,
            "side_to_move": side_to_moves,
            "castling_rights": castling_rights,
            "history": histories,
            "move_count": move_counts,
            "game_phase": game_phases,
            "white_elo": white_elos,
            "black_elo": black_elos
        }

    def preprocess_output(self, y_raw: List[Tuple[int, int, int, float]]) -> Dict[str, np.ndarray]:
        """
        Tiền xử lý dữ liệu đầu ra thành one-hot encoding.

        Args:
            y_raw (List[Tuple[int, int, int, float]]): Danh sách các tuple (from_square, to_square, is_promotion, value).

        Returns:
            Dict[str, np.ndarray]: Dictionary chứa one-hot encoding cho from_square, to_square, is_promotion, và value.
        """
        n_samples = len(y_raw)
        from_squares = np.zeros((n_samples, self.num_squares), dtype=np.float32)
        to_squares = np.zeros((n_samples, self.num_squares), dtype=np.float32)
        is_promotions = np.zeros((n_samples, 1), dtype=np.float32)
        values = np.zeros((n_samples, 1), dtype=np.float32)

        for i, (from_sq, to_sq, is_promotion, value) in enumerate(y_raw):
            from_squares[i, from_sq] = 1.0
            to_squares[i, to_sq] = 1.0
            is_promotions[i, 0] = is_promotion
            values[i, 0] = value

        return {
            "from_square": from_squares,
            "to_square": to_squares,
            "is_promotion": is_promotions,
            "value": values
        }

    def create_tensorflow_dataset(self, X_raw: List[Dict], y_raw: List[Tuple[int, int, int]], 
                                  batch_size: int = 32, shuffle: bool = True, buffer_size: int = 10000) -> tf.data.Dataset:
        """
        Tạo TensorFlow dataset từ dữ liệu thô.

        Args:
            X_raw (List[Dict]): Danh sách các dictionary chứa dữ liệu đầu vào thô.
            y_raw (List[Tuple[int, int, int]]): Danh sách các tuple (from_square, to_square, is_promotion, value).
            batch_size (int): Kích thước batch.
            shuffle (bool): Có xáo trộn dữ liệu hay không.
            buffer_size (int): Kích thước buffer cho việc xáo trộn.

        Returns:
            tf.data.Dataset: Dataset TensorFlow đã được tiền xử lý.
        """
        X_processed = self.preprocess_input(X_raw)
        y_processed = self.preprocess_output(y_raw)
        dataset = tf.data.Dataset.from_tensor_slices((X_processed, y_processed))
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset