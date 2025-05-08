import chess
import chess.engine
import numpy as np
from chess_model import ChessModel  # Giả định bạn có class ChessModel

# Hàm prepare_game_state (lấy từ app.py)
def board_to_array(board: chess.Board) -> np.ndarray:
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    board_array = np.zeros((8, 8, 12), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            idx = piece_map[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
            rank = square // 8
            file = square % 8
            board_array[7 - rank, file, idx] = 1.0

    return board_array

def prepare_game_state(board: chess.Board, white_elo: float, black_elo: float, 
                       history: list = None, history_length: int = 8) -> dict:
    board_array = board_to_array(board)
    side_to_move = 1.0 if board.turn == chess.BLACK else 0.0

    castling_rights = np.array([
        float(board.has_kingside_castling_rights(chess.WHITE)),
        float(board.has_queenside_castling_rights(chess.WHITE)),
        float(board.has_kingside_castling_rights(chess.BLACK)),
        float(board.has_queenside_castling_rights(chess.BLACK))
    ], dtype=np.float32)

    if history is None:
        history = []

    history_array = history[-history_length:]
    while len(history_array) < history_length:
        history_array.append((0, 0, 0, 0))

    total_pieces = len(board.piece_map())
    game_phase = np.array(
        [1.0, 0.0, 0.0] if total_pieces > 20 else
        [0.0, 1.0, 0.0] if total_pieces > 10 else
        [0.0, 0.0, 1.0],
        dtype=np.float32
    )

    return {
        "board": board_array,
        "side_to_move": side_to_move,
        "castling_rights": castling_rights,
        "history": history_array,
        "move_count": board.fullmove_number,
        "game_phase": game_phase,
        "white_elo": white_elo,
        "black_elo": black_elo
    }

# Hàm tính Elo
def calculate_elo(current_elo, opponent_elo, results, k=32):
    expected_score = 1 / (1 + 10 ** ((opponent_elo - current_elo) / 400))
    actual_score = sum(results) / len(results)  # Trung bình điểm thực tế
    new_elo = current_elo + k * (actual_score - expected_score)
    return new_elo

# Đường dẫn đến Stockfish (cập nhật đường dẫn thực tế)
stockfish_path = "E:/AI Chess/chess2/src/lib/stockfish.exe"  # Thay bằng đường dẫn của bạn

# Tải mô hình của bạn
model = ChessModel.load('src/ai/models/chess_model_best.keras')

# Tải Stockfish
try:
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": 0})  # Elo ~2000 (ước lượng)
except Exception as e:
    print(f"❌ Error loading Stockfish: {e}")
    exit(1)

# Thiết lập
num_games = 20
results = []  # 1: thắng, 0.5: hòa, 0: thua
current_elo = 1000  # Elo ban đầu của AI
opponent_elo = 1100  # Elo của Stockfish (ước lượng)

# Chơi 10 ván
for game_num in range(num_games):
    board = chess.Board()
    print(f"Game {game_num + 1} started")

    while not board.is_game_over():
        if board.turn == chess.WHITE:  # AI của bạn chơi trắng
            game_state = prepare_game_state(board, 1500, 1500)
            move = model.predict_move(game_state, board)
            if move and move in board.legal_moves:
                board.push(move)
            else:
                # Nếu không tìm được nước hợp lệ, chọn ngẫu nhiên
                legal_moves = list(board.legal_moves)
                board.push(legal_moves[0])
        else:  # Stockfish chơi đen
            result = engine.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)

    # Ghi lại kết quả
    result = board.result()
    if result == "1-0":
        results.append(1)  # Thắng
        print(f"Game {game_num + 1}: AI wins")
    elif result == "0-1":
        results.append(0)  # Thua
        print(f"Game {game_num + 1}: Stockfish wins")
    else:
        results.append(0.5)  # Hòa
        print(f"Game {game_num + 1}: Draw")

# Tính Elo mới
new_elo = calculate_elo(current_elo, opponent_elo, results)
print(f"Results: {results}")
print(f"Initial Elo: {current_elo}")
print(f"Final Elo after {num_games} games: {new_elo}")

# Đóng engine
engine.quit()