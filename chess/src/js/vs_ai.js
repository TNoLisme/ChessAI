
let board;
let game = new Chess();
let playerColor = "w";

let timePlayer = 1200;
let timeAI = 1200;
let timerPlayer, timerAI;

function startGame() {
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
    onSquareClick: onSquareClick,
    onMouseoverSquare: onMouseoverSquare,
    onMouseoutSquare: onMouseoutSquare,
    pieceTheme: "img/chesspieces/wikipedia/{piece}.png",
    orientation: "white",
  });

  startTimer();
}

function onDrop(source, target) {
  let move = game.move({ from: source, to: target, promotion: "q" });
  if (move === null) return "snapback";

  board.position(game.fen());

  if (game.game_over()) {
    showResult(getGameResult());
    return;
  }

  if (game.turn() === "b") {
    setTimeout(() => sendPositionToAI(game.history()), 250);
  }
}

function onMouseoverSquare(square, piece) {
  // Chỉ hiển thị nước đi của quân đang được điều khiển
  if ((game.turn() === 'w' && piece[0] !== 'w') ||
      (game.turn() === 'b' && piece[0] !== 'b')) return;

  const moves = game.moves({ square, verbose: true });
  if (!moves.length) return;

  const squaresToHighlight = moves.map(m => m.to);
  squaresToHighlight.push(square);
  highlightSquares(squaresToHighlight);
}

function onMouseoutSquare(/* square, piece */) {
  removeHighlightSquares();
}

// Xóa hết chấm dot
function removeHighlightSquares() {
  document.querySelectorAll('#board .dot')
    .forEach(el => el.classList.remove('dot'));
}

// Chèn chấm dot cho từng ô
function highlightSquares(squares) {
  squares.forEach(sq => {
    const el = document.querySelector(`#board .square-${sq}`);
    if (el) el.classList.add('dot');
  });
}


function sendPositionToAI(moveHistory) {
  fetch("/get_ai_move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fen: game.fen() }),
  })
    .then(response => {
      if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
      return response.json();
    })
    .then(data => {
      if (data.error) console.error("Lỗi từ server AI:", data.error);
      const aiMove = data.move;
      if (!aiMove) return;
      game.move(aiMove);
      board.position(game.fen());
      if (game.game_over()) showResult(getGameResult());
    })
    .catch(error => console.error("Lỗi từ server AI:", error));
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
  window.location.href = "/";
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
function onSquareClick(square, piece) {
  // chỉ highlight khi click đúng quân đang cầm (white/black)
  if ((game.turn() === 'w' && (!piece || piece[0] !== 'w')) ||
      (game.turn() === 'b' && (!piece || piece[0] !== 'b'))) {
    removeHighlightSquares();
    return;
  }

  const moves = game.moves({ square, verbose: true });
  if (!moves.length) {
    removeHighlightSquares();
    return;
  }

  const squaresToHighlight = moves.map(m => m.to);
  squaresToHighlight.push(square);
  highlightSquares(squaresToHighlight);
}
