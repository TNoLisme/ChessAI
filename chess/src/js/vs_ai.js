// vs_ai.js - Thêm chức năng đánh giá Elo cho AI và người chơi

let board;
let game = new Chess();
let playerColor = "w";

// Biến Elo và thời gian
let playerRating = parseInt(localStorage.getItem('playerRating')) || 1200;
let aiRating = parseInt(localStorage.getItem('aiRating')) || 1200;
let timePlayer = 1200;
let timeAI = 1200;
let timerPlayer, timerAI;

let selectedSquare = null;
let lastMove = null;

// --- Hàm Elo cơ bản ---
/**
 * Tính Elo mới sau một trận đấu
 * @param {number} R1 - Elo của Player 1 (AI)
 * @param {number} R2 - Elo của Player 2 (người)
 * @param {number} S1 - Kết quả từ góc AI (1 thắng, 0.5 hòa, 0 thua)
 * @param {number} K - Hệ số điều chỉnh
 * @returns {{newR1: number, newR2: number}}
 */
function updateElo(R1, R2, S1, K = 32) {
  const E1 = 1 / (1 + Math.pow(10, (R2 - R1) / 400));
  const E2 = 1 - E1;
  const S2 = 1 - S1;
  const newR1 = R1 + K * (S1 - E1);
  const newR2 = R2 + K * (S2 - E2);
  return { newR1: Math.round(newR1), newR2: Math.round(newR2) };
}

/**
 * Xác định K-factor dựa trên Elo hiện tại
 */
function getKFactor(currentRating) {
  if (currentRating < 2100) return 32;
  if (currentRating < 2400) return 24;
  return 16;
}

// Hàm cập nhật hiển thị Elo lên giao diện
function updateRatingDisplay() {
  let playerEl = document.getElementById('playerRating');
  let aiEl = document.getElementById('aiRating');
  if (!playerEl) {
    playerEl = document.createElement('div');
    playerEl.id = 'playerRating';
    document.querySelector('.player-left').appendChild(playerEl);
  }
  if (!aiEl) {
    aiEl = document.createElement('div');
    aiEl.id = 'aiRating';
    document.querySelector('.player-right').appendChild(aiEl);
  }
  playerEl.textContent = `Elo Bạn: ${playerRating}`;
  aiEl.textContent = `Elo AI: ${aiRating}`;
}

// Hàm xử lý cập nhật Elo sau khi game kết thúc
function processEloUpdate(resultMessage) {
  // Xác định kết quả từ góc AI
  let S1;
  if (resultMessage.includes('AI') && resultMessage.includes('thắng')) S1 = 1;
  else if (resultMessage.includes('Bạn') && resultMessage.includes('thắng')) S1 = 0;
  else S1 = 0.5; // hòa

  const K = getKFactor(aiRating);
  const { newR1, newR2 } = updateElo(aiRating, playerRating, S1, K);
  aiRating = newR1;
  playerRating = newR2;

  // Lưu vào localStorage
  localStorage.setItem('aiRating', aiRating);
  localStorage.setItem('playerRating', playerRating);

  updateRatingDisplay();
}
// --- End Elo ---

// Hàm cập nhật lịch sử nước đi
function updateMoveHistory() {
  const moveList = document.getElementById("moveList");
  moveList.innerHTML = "";
  const history = game.history();
  for (let i = 0; i < history.length; i += 2) {
    const whiteMove = history[i];
    const blackMove = history[i + 1] || '';
    const moveNumber = i / 2 + 1;
    const li = document.createElement("li");
    li.textContent = blackMove
      ? `${moveNumber}. ${whiteMove} ${blackMove}`
      : `${moveNumber}. ${whiteMove}`;
    moveList.appendChild(li);
  }
}

// Xóa đánh dấu nước đi trước
function clearLastMoveHighlight() {
  if (!lastMove) return;
  [lastMove.from, lastMove.to].forEach(sq => {
    const squareEl = document.querySelector(`#board .square-${sq}`);
    if (squareEl) squareEl.style.boxShadow = '';
  });
}

// Đánh dấu ô nguồn và ô đích của nước đi
function highlightLastMove(move) {
  clearLastMoveHighlight();
  ['from', 'to'].forEach(key => {
    const sq = move[key];
    const squareEl = document.querySelector(`#board .square-${sq}`);
    if (squareEl) squareEl.style.boxShadow = 'inset 0 0 10px 3px rgba(255,0,0,0.6)';
  });
  lastMove = move;
}

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

  updateMoveHistory();
  updateRatingDisplay();
  startTimer();
}

function onDrop(source, target) {
  const move = game.move({ from: source, to: target, promotion: "q" });
  if (move === null) return "snapback";

  board.position(game.fen());
  updateMoveHistory();
  highlightLastMove(move);

  if (game.game_over()) {
    const resultMsg = getGameResult();
    showResult(resultMsg);
    processEloUpdate(resultMsg);
    return;
  }

  if (game.turn() === "b") {
    setTimeout(() => sendPositionToAI(), 250);
  }
}

function onMouseoverSquare(square, piece) {
  if ((game.turn() === 'w' && piece[0] !== 'w') ||
      (game.turn() === 'b' && piece[0] !== 'b')) return;

  const moves = game.moves({ square, verbose: true });
  if (!moves.length) return;

  const squaresToHighlight = moves.map(m => m.to);
  squaresToHighlight.push(square);
  highlightSquares(squaresToHighlight);
}

function onMouseoutSquare() {
  removeHighlightSquares();
}

function removeHighlightSquares() {
  document.querySelectorAll('#board .dot')
    .forEach(el => el.classList.remove('dot'));
}

function highlightSquares(squares) {
  squares.forEach(sq => {
    const el = document.querySelector(`#board .square-${sq}`);
    if (el) el.classList.add('dot');
  });
}

function sendPositionToAI() {
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
      const moveObj = game.move(data.move);
      if (!moveObj) return;
      board.position(game.fen());
      updateMoveHistory();
      highlightLastMove(moveObj);
      if (game.game_over()) {
        const resultMsg = getGameResult();
        showResult(resultMsg);
        processEloUpdate(resultMsg);
      }
    })
    .catch(error => console.error("Lỗi từ server AI:", error));
}

function getGameResult() {
  if (game.in_checkmate()) {
    return game.turn() === "w" ? "AI thắng!" : "Bạn thắng!";
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
  updateMoveHistory();
  clearLastMoveHighlight();

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
      if (timePlayer <= 0) {
        const msg = "AI thắng do hết giờ!";
        endGame(msg);
        processEloUpdate(msg);
      }
    }
  }, 1000);

  timerAI = setInterval(() => {
    if (game.turn() === "b") {
      timeAI--;
      updateTimer("timerAI", timeAI);
      if (timeAI <= 0) {
        const msg = "Bạn thắng do hết giờ!";
        endGame(msg);
        processEloUpdate(msg);
      }
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
  removeHighlightSquares();

  if (selectedSquare === square) {
    selectedSquare = null;
    return;
  }

  const moves = game.moves({ square, verbose: true });
  if (!moves.length) {
    selectedSquare = null;
    return;
  }

  selectedSquare = square;
  const destSquares = moves.map(m => m.to);
  highlightSquares(destSquares, square);
}

function greySquare(square) {
  const squareEl = $('#board .square-' + square);
  let background = '#a9a9a9';
  if (squareEl.hasClass('black-3c85d')) background = '#696969';
  squareEl.css('background', background);
}

function highlightSquares(destSquares, sourceSquare) {
  greySquare(sourceSquare);
  destSquares.forEach(sq => greySquare(sq));
}

function removeHighlightSquares() {
  $('#board .square-55d63').css('background', '');
}