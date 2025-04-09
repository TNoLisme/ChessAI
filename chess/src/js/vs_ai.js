let board;
let game = new Chess();
let ai = new Worker("lib/stockfish-17.js"); // Sử dụng file cục bộ
let playerColor = "w"; // Người chơi luôn là trắng

let timePlayer = 1200; // 20 phút
let timeAI = 1200;
let timerPlayer, timerAI;

function startGame() {
  console.log("Đã bấm Bắt đầu!");

  const playerName = document.getElementById("playerInput").value || "Bạn";
  document.getElementById("playerName").textContent = `${playerName} (Trắng)`;

  document.getElementById("name-form").style.display = "none";

  document.getElementById("board").style.display = "block";
  document.querySelector(".player-left").style.display = "flex";
  document.querySelector(".player-right").style.display = "flex";

  board = Chessboard("board", {
    draggable: true,
    position: "start",
    onDrop: onDrop,
    pieceTheme: "img/chesspieces/wikipedia/{piece}.png",
    orientation: "white",
  });

  startTimer();
}

function onDrop(source, target) {
  let move = game.move({
    from: source,
    to: target,
    promotion: "q",
  });

  if (move === null) return "snapback";

  board.position(game.fen());

  if (game.game_over()) {
    showResult(getGameResult());
    return;
  }

  // Đảm bảo AI chỉ di chuyển khi đến lượt của nó (màu đen)
  if (game.turn() === "b") {
    setTimeout(makeAIMove, 250);
  }
}

function makeAIMove() {
  if (game.game_over() || game.turn() !== "b") return;

  // Gửi lệnh để Stockfish tính toán nước đi
  ai.postMessage("uci");
  ai.postMessage("ucinewgame"); // Khởi tạo ván mới để tránh lỗi
  ai.postMessage(`position fen ${game.fen()}`);
  ai.postMessage("go depth 20"); // Tăng độ sâu để AI mạnh hơn (Elo > 2400)

  // Lắng nghe phản hồi từ Stockfish
  ai.onmessage = function (event) {
    let message = event.data;
    console.log("Stockfish response:", message); // Debug để kiểm tra phản hồi

    if (message.startsWith("bestmove")) {
      const move = message.split(" ")[1];
      if (move === "(none)") return;

      // Thực hiện nước đi của AI
      game.move({
        from: move.substring(0, 2),
        to: move.substring(2, 4),
        promotion: "q",
      });

      board.position(game.fen());

      if (game.game_over()) {
        showResult(getGameResult());
      }
    }
  };
}

function getGameResult() {
  if (game.in_checkmate()) {
    return game.turn() === "w" ? "AI Jinwoo thắng!" : "Bạn thắng!";
  } else if (game.in_draw() || game.in_stalemate()) {
    return "Hòa!";
  }
  return "";
}

function showResult(message) {
  clearInterval(timerPlayer);
  clearInterval(timerAI);

  const resultContainer = document.createElement("div");
  resultContainer.className = "result-container";
  resultContainer.innerHTML = `
    <div class="result-box">
      <h2>${message}</h2>
      <button onclick="restartGame()">Chơi lại</button>
      <button onclick="goHome()">Quay lại trang chủ</button>
    </div>
  `;

  document.body.appendChild(resultContainer);
}

function goHome() {
  window.location.href = "index.html";
}

function restartGame() {
  const resultContainer = document.querySelector(".result-container");
  if (resultContainer) resultContainer.remove();

  game.reset();
  board.position("start");

  timePlayer = 1200;
  timeAI = 1200;

  updateTimer("timerPlayer", timePlayer);
  updateTimer("timerAI", timeAI);

  clearInterval(timerPlayer);
  clearInterval(timerAI);
  startTimer();
}

function startTimer() {
  timerPlayer = setInterval(() => {
    if (game.turn() === "w") {
      timePlayer--;
      updateTimer("timerPlayer", timePlayer);
      if (timePlayer <= 0) endGame("AI Jinwoo thắng do hết giờ!");
    }
  }, 1000);

  timerAI = setInterval(() => {
    if (game.turn() === "b") {
      timeAI--;
      updateTimer("timerAI", timeAI);
      if (timeAI <= 0) endGame("Bạn thắng do hết giờ!");
    }
  }, 1000);
}

function updateTimer(id, seconds) {
  const min = String(Math.floor(seconds / 60)).padStart(2, "0");
  const sec = String(seconds % 60).padStart(2, "0");
  document.getElementById(id).textContent = `${min}:${sec}`;
}

function endGame(message) {
  clearInterval(timerPlayer);
  clearInterval(timerAI);
  showResult(message);
}

// Khởi động Stockfish
ai.postMessage("uci");
ai.postMessage("isready");
