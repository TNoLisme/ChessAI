let board;
let game = new Chess();

let currentPlayer = "white"; // bắt đầu là trắng (player 1)

let timeWhite = 1200; // 20 phút
let timeBlack = 1200;

let timerWhite, timerBlack;

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
    draggable: true, // ✅ Cho phép kéo thả quân cờ
    position: "start",
    onDrop: onDrop, // ✅ Gắn sự kiện xử lý khi thả quân cờ
    pieceTheme: "img/chesspieces/wikipedia/{piece}.png",
  });

  startTimer();
}

function onDrop(source, target) {
  const move = game.move({
    from: source,
    to: target,
    promotion: "q",
  });

  if (move === null) {
    return "snapback";
  }

  setTimeout(() => board.position(game.fen()), 100);

  // ✅ Kiểm tra chiếu hết hoặc hòa
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
  window.location.href = "index.html"; // hoặc đường dẫn trang chủ của bạn
}

function restartGame() {
  // Xóa container kết quả nếu có
  const resultContainer = document.querySelector(".result-container");
  if (resultContainer) resultContainer.remove();

  // Đảo vai người chơi
  currentPlayer = "white"; // reset lượt đầu tiên

  // Đảo màu: nếu board hiện đang trắng bên dưới → lật
  const currentOrientation = board.orientation(); // "white" hoặc "black"
  const newOrientation = currentOrientation === "white" ? "black" : "white";
  board.orientation(newOrientation);

  // Reset trò chơi
  game.reset();
  board.position("start");

  // Đổi thời gian
  timeWhite = 1200;
  timeBlack = 1200;

  updateTimer("timer1", timeWhite);
  updateTimer("timer2", timeBlack);

  // Reset timer
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
