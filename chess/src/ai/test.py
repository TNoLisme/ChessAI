import os
import numpy as np
import tensorflow as tf
import chess
from chess_model import ChessModel
from chess_data_preprocessor import ChessDataPreprocessor
from typing import List, Tuple, Dict, Optional

def board_to_array(board: chess.Board) -> np.ndarray:
    """
    Chuyển trạng thái bàn cờ thành ma trận 8x8x12.

    Args:
        board (chess.Board): Đối tượng bàn cờ.

    Returns:
        np.ndarray: Ma trận 8x8x12 biểu diễn bàn cờ.
    """
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    board_array = np.zeros((8, 8, 12), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_idx = piece_map[piece.piece_type]
            color_offset = 0 if piece.color == chess.WHITE else 6
            rank = square // 8
            file = square % 8
            board_array[7 - rank, file, piece_idx + color_offset] = 1.0

    return board_array

def prepare_game_state(board: chess.Board, white_elo: float, black_elo: float, 
                       history: Optional[List[Tuple[int, int, int, int]]] = None, 
                       history_length: int = 8) -> Dict:
    """
    Chuẩn bị trạng thái ván cờ để đưa vào mô hình.

    Args:
        board (chess.Board): Đối tượng bàn cờ.
        white_elo (float): Elo của người chơi trắng.
        black_elo (float): Elo của người chơi đen.
        history (List[Tuple[int, int, int, int]], optional): Lịch sử nước đi.
        history_length (int): Độ dài lịch sử tối đa.

    Returns:
        Dict: Trạng thái ván cờ đã được chuẩn bị.
    """
    # Tạo ma trận bàn cờ
    board_array = board_to_array(board)

    # Lượt đi
    side_to_move = 1.0 if board.turn == chess.BLACK else 0.0

    # Quyền nhập thành
    castling_rights = np.zeros(4, dtype=np.float32)
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_rights[0] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_rights[1] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_rights[2] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_rights[3] = 1.0

    # Lịch sử nước đi
    if history is None:
        history = []
    history_array = []
    for i in range(min(len(history), history_length)):
        move = history[-(i+1)]  # Lấy từ mới nhất đến cũ nhất
        history_array.append(move)
    # Điền các giá trị 0 nếu lịch sử ngắn hơn history_length
    while len(history_array) < history_length:
        history_array.append((0, 0, 0, 0))  # Thay -1 bằng 0
    history_array = history_array[-history_length:]  # Đảm bảo đúng độ dài

    # Số nước đi
    move_count = board.fullmove_number

    # Giai đoạn ván cờ (giả lập đơn giản)
    total_pieces = len(board.piece_map())
    if total_pieces > 20:
        game_phase = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Mở đầu
    elif total_pieces > 10:
        game_phase = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Trung cuộc
    else:
        game_phase = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Tàn cuộc

    return {
        "board": board_array,
        "side_to_move": side_to_move,
        "castling_rights": castling_rights,
        "history": history_array,
        "move_count": move_count,
        "game_phase": game_phase,
        "white_elo": white_elo,
        "black_elo": black_elo
    }

def main():
    # Đường dẫn đến mô hình đã huấn luyện
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, 'ai', 'models', 'chess_model_best.h5')

    # Kiểm tra file mô hình
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Tải mô hình
    print("🔄 Loading model...")
    model = ChessModel.load(model_path)
    print("✅ Model loaded")

    # Tạo bàn cờ từ FEN (sau các nước đi e4, e5, Nf3)
    board = chess.Board("rnbqkbnr/pp1p1ppp/8/2p1p3/4P3/3P1N2/PPP2PPP/RNBQKB1R b KQkq - 1 2")
    print("\n📜 Current board state:")
    print(board)

    # Thiết lập Elo của hai người chơi
    white_elo = 2500.0
    black_elo = 2300.0

    # Lịch sử nước đi: e2e4, e7e5, g1f3
    history = [
        (52, 36, 0, 0),  # e2e4: ô 52 -> ô 36, không bắt quân, không đặc biệt
        (12, 28, 0, 0),  # e7e5: ô 12 -> ô 28, không bắt quân, không đặc biệt
        (62, 45, 0, 0)   # g1f3: ô 62 -> ô 45, không bắt quân, không đặc biệt
    ]

    # Chuẩn bị trạng thái ván cờ
    game_state = prepare_game_state(board, white_elo, black_elo, history)

    # Dự đoán nước đi
    print("🔄 Predicting move...")
    predicted_move = model.predict_move(game_state, board)

    # Lấy giá trị value (xác suất thắng) từ mô hình
    preprocessor = ChessDataPreprocessor(history_length=8)
    X_processed = preprocessor.preprocess_input([game_state])
    _, _, _, value = model.model.predict(
        [X_processed["board"], X_processed["history"], X_processed["white_elo"],
         X_processed["black_elo"], X_processed["side_to_move"], X_processed["castling_rights"],
         X_processed["move_count"], X_processed["game_phase"]], verbose=0
    )

    # Hiển thị kết quả
    print("\n📜 Current board state:")
    print(board)
    if predicted_move:
        print(f"\n✅ Predicted move: {predicted_move.uci()} (from {chess.square_name(predicted_move.from_square)} to {chess.square_name(predicted_move.to_square)})")
        if predicted_move.promotion:
            print(f"Promotion to: {chess.piece_name(predicted_move.promotion)}")
    else:
        print("\n❌ No legal move predicted.")
    print(f"\n📈 Predicted win probability (value): {value[0, 0]:.4f} (from -1 to 1)")

if __name__ == "__main__":
    main()