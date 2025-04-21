import pandas as pd
import numpy as np
import chess
import os
from tqdm import tqdm
import logging
from typing import List, Tuple, Dict, Optional

# Thiết lập logging để ghi nhận các lỗi hoặc cảnh báo
logging.basicConfig(level=logging.INFO)

class ChessDataProcessor:
    """
    Class xử lý dữ liệu cờ vua từ file CSV, chuyển thành vector và lưu dưới dạng file .npy.
    """
    
    def __init__(self, history_length: int = 8):
        """
        Khởi tạo bộ xử lý dữ liệu cờ vua.

        Args:
            history_length (int): Số lượng nước đi lịch sử được lưu cho mỗi trạng thái.
        """
        self.history_length = history_length
        self.piece_to_index = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # Quân trắng
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Quân đen
        }

    def board_to_matrix(self, board: chess.Board) -> np.ndarray:
        """
        Chuyển trạng thái bàn cờ thành ma trận (8, 8, 12) dạng one-hot.

        Args:
            board (chess.Board): Đối tượng bàn cờ từ thư viện python-chess.

        Returns:
            np.ndarray: Ma trận 8x8x12 biểu diễn trạng thái bàn cờ.
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
        Chuyển nước đi dạng UCI (e.g., 'e2e4') thành tuple (from_square, to_square).

        Args:
            uci_move (str): Chuỗi nước đi dạng UCI.

        Returns:
            Optional[Tuple[int, int]]: Tuple chứa (from_square, to_square) hoặc None nếu không hợp lệ.
        """
        try:
            from_square = chess.SQUARE_NAMES.index(uci_move[:2])
            to_square = chess.SQUARE_NAMES.index(uci_move[2:4])
            return (from_square, to_square)
        except Exception as e:
            logging.warning(f"Invalid UCI move: {uci_move}, error: {e}")
            return None

    def process_game_data(self, row: pd.Series, validate_moves: bool = True) -> Tuple[List[Dict], List[Tuple[int, int, int]]]:
        """
        Xử lý một ván cờ thành danh sách (X, y) cho huấn luyện.

        Args:
            row (pd.Series): Dòng dữ liệu từ DataFrame (chứa WhiteElo, BlackElo, Moves).
            validate_moves (bool): Có kiểm tra tính hợp lệ của nước đi hay không.

        Returns:
            Tuple[List[Dict], List[Tuple[int, int, int]]]: Dữ liệu đầu vào (X) và đầu ra (y).
        """
        white_elo = row['WhiteElo']
        black_elo = row['BlackElo']
        moves = row['Moves'].split()

        if not moves:
            logging.warning(f"Game data has no valid moves: {row}")
            return [], []

        board = chess.Board()
        X, y = [], []
        history = []
        move_count = 0

        for i in range(len(moves) - 1):  # Trừ bước cuối vì không có next_move
            move_uci = moves[i]
            next_move = self.uci_to_index(moves[i+1])

            try:
                if validate_moves:
                    move = chess.Move.from_uci(move_uci)
                    if move not in board.legal_moves:
                        logging.warning(f"Invalid move {move_uci} in game: {row}")
                        break
                # Lưu thông tin trước khi đẩy nước đi
                side_to_move = 0 if board.turn == chess.WHITE else 1
                castling_rights = [
                    1 if board.has_kingside_castling_rights(chess.WHITE) else 0,
                    1 if board.has_queenside_castling_rights(chess.WHITE) else 0,
                    1 if board.has_kingside_castling_rights(chess.BLACK) else 0,
                    1 if board.has_queenside_castling_rights(chess.BLACK) else 0
                ]
                # Xác định giai đoạn ván cờ
                game_phase = [0, 0, 0]
                if move_count < 15:
                    game_phase[0] = 1  # Khai cuộc
                elif move_count < 40:
                    game_phase[1] = 1  # Trung cuộc
                else:
                    game_phase[2] = 1  # Tàn cuộc

                # Lưu thông tin lịch sử nước đi
                move_info = None
                captured_piece = -1
                special_move_type = 0
                if move:
                    from_square = move.from_square
                    to_square = move.to_square
                    captured = board.piece_at(to_square)
                    captured_piece = self.piece_to_index.get(captured.symbol(), -1) if captured else -1
                    if move_uci in ['e1g1', 'e1c1', 'e8g8', 'e8c8']:  # Nhập thành
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

            # Kiểm tra phong cấp cho nước đi tiếp theo
            is_promotion = 0
            if next_move:
                from_square, to_square = next_move
                piece = board.piece_at(from_square)
                if piece and piece.piece_type == chess.PAWN:
                    to_row = to_square // 8
                    if (piece.color == chess.WHITE and to_row == 7) or (piece.color == chess.BLACK and to_row == 0):
                        is_promotion = 1  # Tự động phong cấp thành Hậu

            data_point = {
                "board": board_matrix,
                "side_to_move": side_to_move,
                "castling_rights": castling_rights,
                "history": history_trimmed,
                "move_count": move_count,
                "game_phase": game_phase,
                "white_elo": white_elo,
                "black_elo": black_elo
            }

            if next_move:
                X.append(data_point)
                y.append((next_move[0], next_move[1], is_promotion))

            history.append(move_info)

        return X, y

    def process_and_save_csv_data(self, csv_path: str, max_games: int = 2000, games_per_file: int = 1000, output_folder: str = "data") -> None:
        """
        Xử lý dữ liệu từ file CSV và lưu dưới dạng các file .npy.

        Args:
            csv_path (str): Đường dẫn đến file CSV.
            max_games (int): Số ván cờ tối đa để xử lý (cho mục đích test).
            games_per_file (int): Số ván cờ mỗi file .npy.
            output_folder (str): Thư mục lưu kết quả.
        """
        # Đọc file CSV
        print("🔄 Reading CSV file...")
        table_pgn = pd.read_csv(csv_path)
        table_pgn = table_pgn.head(max_games)  # Giới hạn số ván cờ
        print(f"✅ Loaded {len(table_pgn)} games from CSV")

        os.makedirs(output_folder, exist_ok=True)
        all_X, all_y = [], []
        games_in_file = 0
        file_id = 0

        for idx, row in tqdm(table_pgn.iterrows(), total=len(table_pgn), desc="Processing games"):
            X_game, y_game = self.process_game_data(row)
            if X_game and y_game:
                all_X.extend(X_game)
                all_y.extend(y_game)
                games_in_file += 1

            if games_in_file >= games_per_file:
                np.save(f"{output_folder}/chess_data_{file_id}.npy", {"X": all_X, "y": all_y}, allow_pickle=True)
                logging.info(f"Saved file: chess_data_{file_id}.npy (games {idx - games_in_file + 1} to {idx})")
                file_id += 1
                all_X, all_y = [], []
                games_in_file = 0

        if all_X and all_y:
            np.save(f"{output_folder}/chess_data_{file_id}.npy", {"X": all_X, "y": all_y}, allow_pickle=True)
            logging.info(f"Saved file: chess_data_{file_id}.npy (final)")

def main():
    processor = ChessDataProcessor(history_length=8)
    processor.process_and_save_csv_data(
        csv_path="data/pgn_chess_data.csv",
        max_games=2000,  # Giới hạn 2000 ván cờ để test
        games_per_file=1000,
        output_folder="data"
    )

if __name__ == "__main__":
    main()