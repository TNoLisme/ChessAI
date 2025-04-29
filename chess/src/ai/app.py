from flask import Flask, send_from_directory, request, jsonify, send_file
import os
import numpy as np
import chess
import csv
import re
from chess_model import ChessModel
from typing import List, Tuple, Dict, Optional

# Khởi tạo Flask, chỉ định rõ thư mục gốc chứa static & templates
app = Flask(
    __name__,
    static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../')),
    template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

# Đường dẫn đến mô hình AI đã huấn luyện
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'chess_model_best.keras')

# Load mô hình khi server khởi động
print("🔄 Loading Chess AI model...")
model = ChessModel.load(MODEL_PATH)
print("✅ Model loaded successfully")

# Đường dẫn đến file CSV lưu lịch sử ván đấu
CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'games.csv')

# Khởi tạo file CSV với tiêu đề nếu chưa tồn tại
def init_csv():
    """
    Initialize the CSV file with headers if it does not exist.
    """
    if not os.path.exists(CSV_PATH):
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        with open(CSV_PATH, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['game_id', 'history'])

# Lấy game_id tiếp theo (dựa trên số dòng trong CSV)
def get_next_game_id():
    """
    Get the next available game_id based on the current CSV file content.
    """
    if not os.path.exists(CSV_PATH):
        return 1
    with open(CSV_PATH, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        game_ids = [int(row[0]) for row in reader if row and row[0].isdigit()]
        return max(game_ids) + 1 if game_ids else 1

# ===================== HÀM HỖ TRỢ =====================

def board_to_array(board: chess.Board) -> np.ndarray:
    """
    Convert a chess.Board object into a 8x8x12 NumPy array encoding piece positions.
    Each layer corresponds to a piece type and color.
    """
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
                       history: Optional[List[Tuple[int, int, int, int]]] = None, 
                       history_length: int = 8) -> Dict:
    """
    Prepare the game state dictionary for the AI model, including:
    - Current board array
    - Side to move
    - Castling rights
    - Last 'history_length' moves
    - Game phase (opening/midgame/endgame)
    - Players' Elo ratings
    """
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

# ===================== ROUTES =====================

# Trang chính
@app.route('/')
def index():
    """
    Serve the main index.html file for the application.
    """
    index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../index.html'))
    if not os.path.exists(index_path):
        return f"❌ index.html not found at {index_path}", 404
    return send_file(index_path)

# Trả về các file tĩnh (JS, CSS, IMG, LIB, HTML, ...)
@app.route('/<path:filename>')
def static_files(filename):
    """
    Serve static files (JS, CSS, images, libraries, etc.) by their path.
    """
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', filename))
    if not os.path.exists(file_path):
        return f"❌ File {filename} not found.", 404
    return send_file(file_path)

# Dự đoán nước đi từ AI
@app.route('/get_ai_move', methods=['POST'])
def get_ai_move():
    """
    Predict the next best move from the current board state using the AI model.
    Input JSON must contain FEN string and optional move history.
    """
    try:
        data = request.get_json()
        fen = data.get('fen')
        history = data.get('history', [])

        if not fen:
            return jsonify({'error': 'No FEN provided'}), 400

        board = chess.Board(fen)
        white_elo = 2500.0
        black_elo = 2300.0

        history_formatted = [(m['from'], m['to'], m['captured'], m['special']) for m in history]
        game_state = prepare_game_state(board, white_elo, black_elo, history_formatted)

        predicted_move = model.predict_move(game_state, board)
        if predicted_move:
            move_san = board.san(predicted_move)
            return jsonify({'move': move_san})
        else:
            return jsonify({'error': 'No legal move predicted'}), 500

    except Exception as e:
        print(f"🔥 Error: {e}")
        return jsonify({'error': str(e)}), 500

# Lưu lịch sử ván đấu vào CSV
@app.route('/save_game', methods=['POST'])
def save_game():
    """
    Save the completed game's move history into a CSV file.
    Input JSON must contain a 'history' field as a string of moves.
    """
    try:
        data = request.get_json()
        history = data.get('history', '')

        if not history or not isinstance(history, str):
            print(f"🔥 Invalid history received: {history}")
            return jsonify({'error': 'Invalid or empty history'}), 400

        # Kiểm tra định dạng history (chuỗi các nước đi, ví dụ: "e2e4 e7e5")
        moves = history.split()
        if not moves:
            print(f"🔥 Empty history split: {history}")
            return jsonify({'error': 'Empty history'}), 400

        for move in moves:
            if not re.match(r'^[a-h][1-8][a-h][1-8]$', move):
                print(f"🔥 Invalid move format: {move} in history: {history}")
                return jsonify({'error': f'Invalid move format: {move}'}), 400

        # Khởi tạo CSV nếu chưa tồn tại
        init_csv()

        # Lấy game_id tiếp theo
        game_id = get_next_game_id()

        # Ghi vào CSV
        with open(CSV_PATH, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([game_id, history])

        print(f"✅ Saved game {game_id} with history: {history}")
        return jsonify({'message': 'Game saved successfully', 'game_id': game_id})

    except Exception as e:
        print(f"🔥 Error in save_game: {e}")
        return jsonify({'error': str(e)}), 500


# ===================== CHẠY SERVER =====================

if __name__ == '__main__':
    app.run(debug=True, port=5000)