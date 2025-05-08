import chess
import chess.engine
import random
import csv
import os

# Tạo data
# Đường dẫn đến Stockfish
stockfish_path = "D:/AI Chess/chess2/src/lib/stockfish.exe"# Thay bằng đường dẫn của bạn
CSV_PATH = "D:/AI chess/chess2/src/ai/data/games.csv"
# Tải Stockfish
try:
    engine1 = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine2 = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine1.configure({"Skill Level": 20})  # Elo ~1800-2000
    engine2.configure({"Skill Level": 20})
    print("✅ Stockfish engines loaded successfully")
except Exception as e:
    print(f"❌ Error loading Stockfish: {e}")
    exit(1)

# Tùy chọn
USE_MULTIPV = True
MULTIPV_VALUE = 4
USE_OPENING_BOOK = True

# Danh sách các khai cuộc phổ biến (UCI format)
OPENING_MOVES = [
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],  # Italian Game
    ["e2e4", "c7c5"],                          # Sicilian Defense
    ["d2d4", "d7d5", "c2c4"],                  # Queen's Gambit
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],  # Ruy Lopez
    ["c2c4"],                                  # English Opening
    ["e2e4", "e7e6"],                          # French Defense
    ["e2e4", "c7c6"],                          # Caro-Kann Defense
    ["d2d4", "g8f6", "c2c4", "e7e6"],          # Queen's Indian Defense
    ["d2d4", "g8f6", "c2c4", "g7g6"],          # King's Indian Defense
    ["e2e4", "d7d5"],                          # Scandinavian Defense
    ["e2e4", "g8f6"],                          # Alekhine Defense
    ["d2d4", "f7f5"],                          # Dutch Defense
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "f8b4"],  # Nimzo-Indian Defense
    ["e2e4", "e7e5", "g1f3", "g8f6"],          # Petrov Defense
    ["g1f3", "d7d5", "c2c4"],                  # Reti Opening
]

# Khởi tạo file CSV nếu chưa tồn tại
def init_csv():
    if not os.path.exists(CSV_PATH):
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        with open(CSV_PATH, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['game_id', 'history'])
        print(f"✅ Created games.csv at {CSV_PATH}")

# Lấy game_id tiếp theo
def get_next_game_id():
    if not os.path.exists(CSV_PATH):
        return 1
    try:
        with open(CSV_PATH, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Bỏ qua tiêu đề
            game_ids = [int(row[0]) for row in reader if row and row[0].isdigit()]
            return max(game_ids) + 1 if game_ids else 1
    except Exception as e:
        print(f"❌ Error reading games.csv: {e}")
        return 1

# Chơi một ván
def play_game(game_num):
    board = chess.Board()
    move_history = []

    if USE_OPENING_BOOK:
        opening = OPENING_MOVES[game_num % len(OPENING_MOVES)]
        for move in opening:
            board.push_uci(move)
            move_history.append(move)

    print(f"Game {game_num + 1} started")

    while not board.is_game_over():
        engine = engine1 if board.turn == chess.WHITE else engine2
        if USE_MULTIPV:
            info = engine.analyse(board, chess.engine.Limit(time=0.1), multipv=MULTIPV_VALUE)
            moves = [entry["pv"][0] for entry in info if "pv" in entry and entry["pv"]]
            if moves:
                move = random.choice(moves)
            else:
                result = engine.play(board, chess.engine.Limit(time=0.1))
                move = result.move
        else:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            move = result.move

        board.push(move)
        move_history.append(move.uci())

    result = board.result()
    print(f"Game {game_num + 1} result: {result}")
    return move_history, result

# Khởi tạo file CSV
init_csv()

# Chạy nhiều ván và lưu vào games.csv
num_games = 1000
game_histories = []
starting_game_id = get_next_game_id()

for game_num in range(num_games):
    history, result = play_game(game_num)
    game_histories.append((history, result))

    # Lưu lịch sử ván đấu vào games.csv
    game_id = starting_game_id + game_num
    history_str = " ".join(history)
    try:
        with open(CSV_PATH, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([game_id, history_str])
        print(f"✅ Saved game {game_id} to games.csv")
    except Exception as e:
        print(f"❌ Error saving game {game_id}: {e}")

# Kiểm tra sự lặp lại
print("\nChecking for repetition in games:")
for i in range(len(game_histories)):
    for j in range(i + 1, len(game_histories)):
        history_i, result_i = game_histories[i]
        history_j, result_j = game_histories[j]
        if history_i == history_j:
            print(f"Game {i + 1} and Game {j + 1} are identical")

# Đóng engines
engine1.quit()
engine2.quit()