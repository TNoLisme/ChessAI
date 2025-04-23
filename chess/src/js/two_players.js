let board;
let game = new Chess();

let currentPlayer = "white"; // bắt đầu là trắng (player 1)

let timeWhite = 1200; // 20 phút
let timeBlack = 1200;

let timerWhite, timerBlack;
let lastMove = null; // lưu nước đi vừa thực hiện
let selectedSquare = null;


// Cập nhật lịch sử nước đi
function updateMoveHistory() {
  const moveList = document.getElementById("moveList");
  if (!moveList) return;
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

// Xóa highlight của nước đi trước
function clearLastMoveHighlight() {
  if (!lastMove) return;
  [lastMove.from, lastMove.to].forEach(sq => {
    const el = document.querySelector(`#board .square-${sq}`);
    if (el) el.style.boxShadow = '';
  });
}

// Đánh dấu ô nguồn và ô đích của nước đi cuối cùng
function highlightLastMove(move) {
  clearLastMoveHighlight();
  ['from','to'].forEach(key => {
    const sq = move[key];
    const el = document.querySelector(`#board .square-${sq}`);
    if (el) el.style.boxShadow = 'inset 0 0 10px 3px rgba(255,0,0,0.6)';
  });
  lastMove = move;
}

function startGame() {
  console.log("Đã bấm Bắt đầu!");

  const name1 = document.getElementById("input1").value || "Player 1";
  const name2 = document.getElementById("input2").value || "Player 2";

  document.getElementById("name1").textContent = name1;
  document.getElementById("name2").textContent = name2;

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
  });

  // Khởi tạo lịch sử nước đi và xóa highlight cũ
  updateMoveHistory();
  clearLastMoveHighlight();

  startTimer();
}

function onDrop(source, target) {
  const move = game.move({ from: source, to: target, promotion: "q" });
  if (move === null) return "snapback";

  setTimeout(() => board.position(game.fen()), 100);

  // Cập nhật lịch sử và đánh dấu nước đi vừa thực hiện
  updateMoveHistory();
  highlightLastMove(move);

  // Kiểm tra chiếu hết hoặc hòa
  if (game.in_checkmate()) {
    const winner = currentPlayer === "white" ? "Black" : "White";
    showResult(`${winner} thắng bằng chiếu hết!`);
    return;
  }

  if (game.in_draw()) {
    showResult("Ván đấu hòa!");
    return;
  }

  // Chuyển lượt
  currentPlayer = currentPlayer === "white" ? "black" : "white";
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

function showResult(message) {
  clearInterval(timerWhite);
  clearInterval(timerBlack);

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

  // Reset dữ liệu
  currentPlayer = "white";
  game.reset();
  board.position("start");

  // Đổi orientation
  const currOrient = board.orientation();
  board.orientation(currOrient === "white" ? "black" : "white");

  // Reset lịch sử và highlight
  updateMoveHistory();
  clearLastMoveHighlight();

  // Reset thời gian
  timeWhite = 1200; timeBlack = 1200;
  updateTimer("timer1", timeWhite);
  updateTimer("timer2", timeBlack);

  clearInterval(timerWhite);
  clearInterval(timerBlack);
  startTimer();
}

function startTimer() {
  timerWhite = setInterval(() => {
    if (currentPlayer === "white") {
      timeWhite--;
      updateTimer("timer1", timeWhite);
      if (timeWhite <= 0) endGame("Player 2 thắng do hết giờ!");
    }
  }, 1000);

  timerBlack = setInterval(() => {
    if (currentPlayer === "black") {
      timeBlack--;
      updateTimer("timer2", timeBlack);
      if (timeBlack <= 0) endGame("Player 1 thắng do hết giờ!");
    }
  }, 1000);
}

function updateTimer(id, seconds) {
  const min = String(Math.floor(seconds / 60)).padStart(2, "0");
  const sec = String(seconds % 60).padStart(2, "0");
  document.getElementById(id).textContent = `${min}:${sec}`;
}

function endGame(message) {
  clearInterval(timerWhite);
  clearInterval(timerBlack);
  alert(message);
}

/* Thêm hai hàm để add/remove class .dot */
function highlightSquares(squares) {
  squares.forEach(sq => {
    let sqEl = document.querySelector(`#board .square-${sq}`);
    if (!sqEl) return;
    const d = document.createElement("div");
    d.className = "dot";
    sqEl.appendChild(d);
  });
}

function removeHighlightSquares() {
  document.querySelectorAll("#board .dot").forEach(d => d.remove());
}

function onSquareClick(square, piece) {
  removeHighlightSquares();

  if (!piece) return;
  if ((currentPlayer === "white" && piece[0] !== "w") ||
      (currentPlayer === "black" && piece[0] !== "b")) return;

  const moves = game.moves({ square, verbose: true });
  if (!moves.length) return;

  const squaresToHighlight = moves.map(m => m.to);
  squaresToHighlight.push(square);
  highlightSquares(squaresToHighlight);
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