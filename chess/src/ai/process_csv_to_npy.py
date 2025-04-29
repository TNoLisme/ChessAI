import pandas as pd
import numpy as np
import chess
import os
from tqdm import tqdm
import logging
from typing import List, Tuple, Dict, Optional

# Thiết lập cấu hình logging để hiển thị thông báo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ChessDataProcessor:
    """
    Class xử lý dữ liệu cờ vua từ file CSV, chuyển thành vector và lưu dưới dạng file .npy.
    """

    def __init__(self, history_length: int = 12):
        """
        Khởi tạo bộ xử lý dữ liệu cờ vua.
        Args:
            history_length (int): Số lượng nước đi lịch sử được lưu cho mỗi trạng thái.
        """
        self.history_length = 12  # Đặt lại thành 8 để khớp với mô hình
        self.piece_to_index = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # Quân trắng
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Quân đen
        }
        self.material_values = {
            'P': 1, 'p': 1,
            'N': 3, 'n': 3,
            'B': 3, 'b': 3,
            'R': 5, 'r': 5,
            'Q': 9, 'q': 9,
            'K': 0, 'k': 0
        }

    def board_to_matrix(self, board: chess.Board) -> np.ndarray:
        """
        Chuyển trạng thái bàn cờ thành ma trận (8, 8, 12) dạng one-hot.
        Mỗi lớp trong trục thứ 3 là một loại quân cờ (trắng hoặc đen).
        """
        piece_map = board.piece_map()
        board_matrix = np.zeros((8, 8, 12), dtype=np.int8)

        for square, piece in piece_map.items():
            row, col = divmod(square, 8)
            index = self.piece_to_index.get(piece.symbol())
            if index is not None:
                board_matrix[row][col][index] = 1

        return board_matrix

    def uci_to_index(self, uci_move: str) -> Optional[Tuple[int, int]]:
        """
        Chuyển nước đi dạng UCI (ví dụ: 'e2e4') thành chỉ số (from_square, to_square).
        Trả về None nếu nước đi không hợp lệ.
        """
        try:
            from_square = chess.SQUARE_NAMES.index(uci_move[:2])
            to_square = chess.SQUARE_NAMES.index(uci_move[2:4])
            return (from_square, to_square)
        except Exception as e:
            logging.warning(f"Invalid UCI move: {uci_move}, error: {e}")
            return None

    def calculate_material(self, board: chess.Board) -> int:
        """
        Tính tổng giá trị vật chất trên bàn cờ hiện tại.
        Dùng để xác định giai đoạn ván cờ (khai cuộc, trung cuộc, tàn cuộc).
        """
        total_material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                total_material += self.material_values.get(piece.symbol(), 0)
        return total_material

    def process_game_data(self, row: pd.Series, validate_moves: bool = True) -> Tuple[List[Dict], List[Tuple[int, int, int]]]:
        """
        Xử lý một ván cờ thành danh sách đặc trưng (X) và nhãn (y).
        Mỗi bước đi trong ván cờ được mã hóa dưới dạng dữ liệu đầu vào (board state, side to move, ...) và đầu ra là nước đi tiếp theo.
        """
        moves = row['Moves'].split()

        if not moves:
            logging.warning(f"Game data has no valid moves: {row}")
            return [], []

        board = chess.Board()
        X, y = [], []
        history = []
        move_count = 0

        for i in range(len(moves) - 1):
            move_uci = moves[i]
            next_move = self.uci_to_index(moves[i + 1])

            try:
                if validate_moves:
                    move = chess.Move.from_uci(move_uci)
                    if move not in board.legal_moves:
                        logging.warning(f"Invalid move {move_uci} in game: {row}")
                        break
                side_to_move = 0 if board.turn == chess.WHITE else 1
                castling_rights = [
                    1 if board.has_kingside_castling_rights(chess.WHITE) else 0,
                    1 if board.has_queenside_castling_rights(chess.WHITE) else 0,
                    1 if board.has_kingside_castling_rights(chess.BLACK) else 0,
                    1 if board.has_queenside_castling_rights(chess.BLACK) else 0
                ]
                total_material = self.calculate_material(board)
                game_phase = [0, 0, 0]
                if total_material > 60:
                    game_phase[0] = 1  # Khai cuộc
                elif total_material >= 20:
                    game_phase[1] = 1  # Trung cuộc
                else:
                    game_phase[2] = 1  # Tàn cuộc

                move_info = None
                captured_piece = -1
                special_move_type = 0
                if move:
                    from_square = move.from_square
                    to_square = move.to_square
                    captured = board.piece_at(to_square)
                    captured_piece = self.piece_to_index.get(captured.symbol(), -1) if captured else -1
                    if move_uci in ['e1g1', 'e1c1', 'e8g8', 'e8c8']:
                        special_move_type = 1
                    move_info = (from_square, to_square, captured_piece, special_move_type)

                board.push_uci(move_uci)
                move_count += 1
            except Exception as e:
                logging.warning(f"Error pushing move {move_uci}: {e}")
                break

            board_matrix = self.board_to_matrix(board)
            history_trimmed = history[-self.history_length:]
            while len(history_trimmed) < self.history_length:
                history_trimmed.insert(0, None)

            is_promotion = 0
            if next_move:
                from_square, to_square = next_move
                piece = board.piece_at(from_square)
                if piece and piece.piece_type == chess.PAWN:
                    to_row = to_square // 8
                    if (piece.color == chess.WHITE and to_row == 7) or (piece.color == chess.BLACK and to_row == 0):
                        is_promotion = 1

            data_point = {
                "board": board_matrix,
                "side_to_move": side_to_move,
                "castling_rights": castling_rights,
                "history": history_trimmed,
                "move_count": move_count,
                "game_phase": game_phase
            }

            if next_move:
                X.append(data_point)
                y.append((next_move[0], next_move[1], is_promotion))

            history.append(move_info)

        return X, y

    def process_and_save_csv_data(self, csv_path: str, max_games: Optional[int] = None,
                                  games_per_file: int = 1000, output_folder: str = "data") -> None:
        """
        Đọc dữ liệu từ file CSV, xử lý và lưu dưới dạng các file .npy.

        Args:
            csv_path (str): Đường dẫn đến file CSV chứa ván cờ.
            max_games (Optional[int]): Giới hạn số lượng ván cần xử lý.
            games_per_file (int): Số lượng ván trong mỗi file .npy.
            output_folder (str): Thư mục để lưu dữ liệu đã xử lý.
        """
        print("🔄 Reading CSV file...")
        table_pgn = pd.read_csv(csv_path)
        if max_games is not None:
            table_pgn = table_pgn.head(max_games)
        total_games = len(table_pgn)
        print(f"✅ Loaded {total_games} games from CSV")

        os.makedirs(output_folder, exist_ok=True)
        all_X, all_y = [], []
        games_in_file = 0
        file_id = 0
        total_samples = 0

        for idx, row in tqdm(table_pgn.iterrows(), total=total_games, desc="Processing games"):
            X_game, y_game = self.process_game_data(row)
            if X_game and y_game:
                all_X.extend(X_game)
                all_y.extend(y_game)
                total_samples += len(X_game)
                games_in_file += 1
                if idx % 20000 == 0:
                    print(f"Processed {idx + 1} games, {total_samples} samples")

            if games_in_file >= games_per_file:
                output_path = os.path.join(output_folder, f"chess_data_{file_id}.npy")
                np.save(output_path, {"X": all_X, "y": all_y}, allow_pickle=True)
                logging.info(f"Saved file: {output_path} with {len(all_X)} samples (games {idx - games_in_file + 1} to {idx})")
                file_id += 1
                all_X, all_y = [], []
                games_in_file = 0

        if all_X and all_y:
            output_path = os.path.join(output_folder, f"chess_data_{file_id}.npy")
            np.save(output_path, {"X": all_X, "y": all_y}, allow_pickle=True)
            logging.info(f"Saved file: {output_path} with {len(all_X)} samples (final)")

        print(f"✅ Processed {total_games} games, created {total_samples} samples, saved {file_id + 1} .npy files")


def main():
    """
    Hàm chính để khởi chạy xử lý dữ liệu từ file CSV và lưu kết quả ra file .npy.
    """
    processor = ChessDataProcessor(history_length=8)
    processor.process_and_save_csv_data(
        csv_path="src/ai/data/pgn_chess_data.csv",
        max_games=None,  # Đọc toàn bộ CSV
        games_per_file=1000,
        output_folder="src/ai/data"
    )


if __name__ == "__main__":
    # Gọi hàm main nếu file này được thực thi trực tiếp
    main()
