from flask import Flask, request, jsonify, send_from_directory
import chess
import random
import os

# Đường dẫn đến thư mục gốc (C:\Users\thinh\HKII24-25\Trí tuệ nhân tạo\2425\chess)
# app.py nằm trong src/ai_backend, nên cần đi lên 2 cấp để đến thư mục gốc
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Khởi tạo Flask app, đặt static_folder là thư mục gốc để phục vụ file tĩnh
app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')

# Route cho trang chính (phục vụ index.html)
@app.route('/')
def serve_index():
    # Đường dẫn đến file index.html trong thư mục gốc
    file_path = os.path.join(BASE_DIR, 'index.html')
    # Kiểm tra file có tồn tại không
    if not os.path.exists(file_path):
        return f"Error: File 'index.html' not found in {BASE_DIR}", 404
    return send_from_directory(BASE_DIR, 'index.html')

# Route cho chế độ chơi với AI (phục vụ vs_ai.html)
@app.route('/vs_ai')
def serve_vs_ai():
    # Đường dẫn đến file vs_ai.html trong thư mục gốc
    file_path = os.path.join(BASE_DIR, 'vs_ai.html')
    if not os.path.exists(file_path):
        return f"Error: File 'vs_ai.html' not found in {BASE_DIR}", 404
    return send_from_directory(BASE_DIR, 'vs_ai.html')

# Route cho chế độ hai người chơi (phục vụ two_player.html)
@app.route('/two_player')
def serve_two_player():
    # Đường dẫn đến file two_player.html trong thư mục gốc
    file_path = os.path.join(BASE_DIR, 'two_player.html')
    if not os.path.exists(file_path):
        return f"Error: File 'two_player.html' not found in {BASE_DIR}", 404
    return send_from_directory(BASE_DIR, 'two_player.html')

# Route để phục vụ các file tĩnh (CSS, JS, hình ảnh, v.v.)
@app.route('/<path:path>')
def serve_static(path):
    # Đường dẫn đến file tĩnh trong thư mục gốc
    file_path = os.path.join(BASE_DIR, path)
    if not os.path.exists(file_path):
        return f"Error: File '{path}' not found in {BASE_DIR}", 404
    return send_from_directory(BASE_DIR, path)

# Route để lấy nước đi từ AI
@app.route('/get_ai_move', methods=['POST'])
def get_ai_move():
    # Lấy dữ liệu FEN từ request
    data = request.get_json()
    fen = data.get("fen")
    
    # Kiểm tra FEN có được gửi không
    if not fen:
        return jsonify({"error": "FEN is required"}), 400

    # Tạo bàn cờ từ FEN
    try:
        board = chess.Board(fen)
    except Exception as e:
        return jsonify({"error": f"Invalid FEN: {str(e)}"}), 400

    # Lấy danh sách các nước đi hợp lệ
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return jsonify({"error": "No legal moves available"}), 400

    # AI chọn ngẫu nhiên một nước đi
    ai_move = random.choice(legal_moves)
    move_string = board.san(ai_move)  # Chuyển nước đi thành ký hiệu chuẩn (SAN)

    return jsonify({"move": move_string})

if __name__ == '__main__':
    # Chạy server trên port 5000 với debug mode
    app.run(host='0.0.0.0', port=5000, debug=True)