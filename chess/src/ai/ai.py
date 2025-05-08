from flask import Flask, send_from_directory, request, jsonify, send_file
import os
import chess
import chess.engine
import csv
import re

# chơi với AI của stockfish
# Khởi tạo Flask
app = Flask(
    __name__,
    static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../')),
    template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

# Đường dẫn đến Stockfish (cập nhật đường dẫn thực tế)
stockfish_path = "D:/AI Chess/chess2/src/lib/stockfish.exe"
try:
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": 1})  # Elo ~1800-2000
    print("✅ Model AI loaded successfully")
except Exception as e:
    print(f"❌ Error loading model AI: {e}")
    exit(1)

# Đường dẫn đến file CSV lưu lịch sử ván đấu
CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'games.csv')

# Khởi tạo file CSV với tiêu đề nếu chưa tồn tại
def init_csv():
    if not os.path.exists(CSV_PATH):
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        with open(CSV_PATH, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['game_id', 'history'])

# Lấy game_id tiếp theo
def get_next_game_id():
    if not os.path.exists(CSV_PATH):
        return 1
    with open(CSV_PATH, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Bỏ qua tiêu đề
        game_ids = [int(row[0]) for row in reader if row and row[0].isdigit()]
        return max(game_ids) + 1 if game_ids else 1

# Routes
@app.route('/')
def index():
    index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../index.html'))
    if not os.path.exists(index_path):
        return f"❌ index.html not found at {index_path}", 404
    return send_file(index_path)

@app.route('/<path:filename>')
def static_files(filename):
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', filename))
    if not os.path.exists(file_path):
        return f"❌ File {filename} not found.", 404
    return send_file(file_path)

@app.route('/get_ai_move', methods=['POST'])
def get_ai_move():
    try:
        data = request.get_json()
        fen = data.get('fen')

        if not fen:
            return jsonify({'error': 'No FEN provided'}), 400

        # Tạo bàn cờ từ FEN
        board = chess.Board(fen)

        # Dùng Stockfish để dự đoán nước đi
        result = engine.play(board, chess.engine.Limit(time=0.1))
        move = result.move

        if move:
            move_san = board.san(move)
            return jsonify({'move': move_san})
        else:
            return jsonify({'error': 'No legal move predicted'}), 500

    except Exception as e:
        print(f"🔥 Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/save_game', methods=['POST'])
def save_game():
    try:
        data = request.get_json()
        history = data.get('history', '')

        if not history or not isinstance(history, str):
            print(f"🔥 Invalid history received: {history}")
            return jsonify({'error': 'Invalid or empty history'}), 400

        moves = history.split()
        if not moves:
            print(f"🔥 Empty history split: {history}")
            return jsonify({'error': 'Empty history'}), 400

        for move in moves:
            if not re.match(r'^[a-h][1-8][a-h][1-8]$', move):
                print(f"🔥 Invalid move format: {move} in history: {history}")
                return jsonify({'error': f'Invalid move format: {move}'}), 400

        init_csv()
        game_id = get_next_game_id()

        with open(CSV_PATH, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([game_id, history])

        print(f"✅ Saved game {game_id} with history: {history}")
        return jsonify({'message': 'Game saved successfully', 'game_id': game_id})

    except Exception as e:
        print(f"🔥 Error in save_game: {e}")
        return jsonify({'error': str(e)}), 500

# Chạy server
if __name__ == '__main__':
    try:
        app.run(debug=True, port=5000)
    finally:
        engine.quit()  