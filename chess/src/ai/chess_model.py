import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, regularizers
import numpy as np
import chess
from typing import Dict, Tuple, Optional

class ChessModel:
    """
    Class định nghĩa kiến trúc mô hình học máy cho AI cờ vua, dự đoán nước đi tiếp theo và giá trị vị trí.
    """
    
    def __init__(self, config: Dict = None):
        """
        Khởi tạo mô hình với các tham số cấu hình.

        Args:
            config (Dict, optional): Dictionary chứa các tham số cấu hình. Nếu None, sử dụng mặc định.
        """
        default_config = {
            'filters_conv': 64,
            'filters_res': 64,
            'num_res_blocks': 8,
            'mlp_units': 64,
            'dropout_rate': 0.3,
            'l2_reg': 1e-4,
            'history_length': 8
        }
        config = config or default_config
        self.filters_conv = config['filters_conv']
        self.filters_res = config['filters_res']
        self.num_res_blocks = config['num_res_blocks']
        self.mlp_units = config['mlp_units']
        self.dropout_rate = config['dropout_rate']
        self.l2_reg = config['l2_reg']
        self.history_length = config['history_length']
        self.num_squares = 64
        self.model = self._build_model()

    def _residual_block(self, x: tf.Tensor) -> tf.Tensor:
        """
        Tạo một residual block cho mạng CNN.

        Args:
            x (tf.Tensor): Tensor đầu vào.

        Returns:
            tf.Tensor: Tensor đã qua residual block.
        """
        shortcut = x
        x = layers.Conv2D(self.filters_res, kernel_size=(3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(self.filters_res, kernel_size=(3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([shortcut, x])
        x = layers.Activation('relu')(x)
        return x

    def _process_board(self, board_input: tf.Tensor) -> tf.Tensor:
        """
        Xử lý đầu vào bàn cờ bằng mạng CNN.

        Args:
            board_input (tf.Tensor): Tensor biểu diễn bàn cờ (batch_size, 8, 8, 12).

        Returns:
            tf.Tensor: Feature vector đã xử lý.
        """
        x = layers.Conv2D(self.filters_conv, kernel_size=(3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(self.l2_reg))(board_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        for _ in range(self.num_res_blocks):
            x = self._residual_block(x)
        x = layers.GlobalAveragePooling2D()(x)
        return x

    def _process_history(self, history_input: tf.Tensor) -> tf.Tensor:
        """
        Xử lý đầu vào lịch sử nước đi bằng GRU.

        Args:
            history_input (tf.Tensor): Tensor biểu diễn lịch sử (batch_size, history_length, 4).

        Returns:
            tf.Tensor: Feature vector đã xử lý.
        """
        x = history_input
        x = layers.GRU(32, return_sequences=False,
                       kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.Dense(16, activation='relu',
                        kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        return x

    def _process_metadata(self, white_elo: tf.Tensor, black_elo: tf.Tensor,
                         side_to_move: tf.Tensor, castling_rights: tf.Tensor,
                         move_count: tf.Tensor, game_phase: tf.Tensor) -> tf.Tensor:
        """
        Xử lý metadata bằng MLP.

        Args:
            white_elo (tf.Tensor): Tensor biểu diễn Elo người chơi trắng.
            black_elo (tf.Tensor): Tensor biểu diễn Elo người chơi đen.
            side_to_move (tf.Tensor): Tensor biểu diễn lượt đi.
            castling_rights (tf.Tensor): Tensor biểu diễn quyền nhập thành.
            move_count (tf.Tensor): Tensor biểu diễn số nước đi.
            game_phase (tf.Tensor): Tensor biểu diễn giai đoạn ván cờ.

        Returns:
            tf.Tensor: Feature vector đã xử lý.
        """
        x = layers.Concatenate()([white_elo, black_elo, side_to_move, castling_rights, move_count, game_phase])
        x = layers.Dense(16, activation='relu',
                        kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.Dense(8, activation='relu',
                        kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        return x

    def _build_model(self) -> Model:
        """
        Xây dựng kiến trúc mô hình hoàn chỉnh.

        Returns:
            Model: Đối tượng Model của Keras.
        """
        board_input = layers.Input(shape=(8, 8, 12), name='board')
        history_input = layers.Input(shape=(self.history_length, 4), name='history')
        white_elo_input = layers.Input(shape=(1,), name='white_elo')
        black_elo_input = layers.Input(shape=(1,), name='black_elo')
        side_to_move_input = layers.Input(shape=(1,), name='side_to_move')
        castling_rights_input = layers.Input(shape=(4,), name='castling_rights')
        move_count_input = layers.Input(shape=(1,), name='move_count')
        game_phase_input = layers.Input(shape=(3,), name='game_phase')

        board_features = self._process_board(board_input)
        history_features = self._process_history(history_input)
        metadata_features = self._process_metadata(white_elo_input, black_elo_input,
                                                  side_to_move_input, castling_rights_input,
                                                  move_count_input, game_phase_input)
        combined_features = layers.Concatenate()([board_features, history_features, metadata_features])

        x = layers.Dense(self.mlp_units, kernel_regularizer=regularizers.l2(self.l2_reg))(combined_features)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.mlp_units, kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Policy network
        from_square_output = layers.Dense(self.num_squares, activation='softmax', name='from_square')(x)
        to_square_output = layers.Dense(self.num_squares, activation='softmax', name='to_square')(x)
        is_promotion_output = layers.Dense(1, activation='sigmoid', name='is_promotion')(x)

        # Value network
        value_x = layers.Dense(self.mlp_units, activation='relu',
                              kernel_regularizer=regularizers.l2(self.l2_reg))(combined_features)
        value_x = layers.Dense(self.mlp_units // 2, activation='relu',
                              kernel_regularizer=regularizers.l2(self.l2_reg))(value_x)
        value_output = layers.Dense(1, activation='tanh', name='value')(value_x)

        model = Model(
            inputs=[board_input, history_input, white_elo_input, black_elo_input,
                    side_to_move_input, castling_rights_input, move_count_input, game_phase_input],
            outputs=[from_square_output, to_square_output, is_promotion_output, value_output]
        )
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss={
                'from_square': tf.keras.losses.CategoricalCrossentropy(),
                'to_square': tf.keras.losses.CategoricalCrossentropy(),
                'is_promotion': tf.keras.losses.BinaryCrossentropy(),
                'value': tf.keras.losses.MeanSquaredError()  # Thay 'mse' bằng đối tượng
            },
            metrics={
                'from_square': ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')],
                'to_square': ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')],
                'is_promotion': ['accuracy'],
                'value': [tf.keras.metrics.MeanAbsoluteError(name='mae')]  # Thay 'mae' bằng đối tượng
            }
        )
        return model

    def train(self, train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset, 
              epochs: int = 100, callbacks: list = None) -> Dict:
        """
        Huấn luyện mô hình trên dữ liệu.

        Args:
            train_dataset (tf.data.Dataset): Dataset huấn luyện.
            validation_dataset (tf.data.Dataset): Dataset kiểm định.
            epochs (int): Số epoch huấn luyện.
            callbacks (list, optional): Danh sách các callback.

        Returns:
            Dict: History của quá trình huấn luyện.
        """
        if callbacks is None:
            callbacks = []
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ))
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            'ai/models/chess_model_best.h5', monitor='val_loss', save_best_only=True, verbose=1
        ))
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
        ))

        history = self.model.fit(
            train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=callbacks
        )
        return history.history

    def predict_move(self, game_state: Dict, board: chess.Board) -> Optional[chess.Move]:
        """
        Dự đoán nước đi tốt nhất cho trạng thái ván cờ.

        Args:
            game_state (Dict): Dictionary chứa trạng thái ván cờ.
            board (chess.Board): Đối tượng bàn cờ hiện tại.

        Returns:
            Optional[chess.Move]: Nước đi dự đoán hoặc None nếu không hợp lệ.
        """
        return self.get_best_legal_move(board, game_state, top_k=5)

    def save(self, filepath: str) -> None:
        """
        Lưu mô hình vào file.

        Args:
            filepath (str): Đường dẫn lưu file mô hình.
        """
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath: str) -> 'ChessModel':
        """
        Tải mô hình từ file.

        Args:
            filepath (str): Đường dẫn đến file mô hình.

        Returns:
            ChessModel: Đối tượng ChessModel với mô hình đã tải.
        """
        instance = cls()
        instance.model = tf.keras.models.load_model(filepath)
        return instance

    def get_legal_move(self, board: chess.Board, from_square: int, to_square: int, 
                       is_promotion: float) -> Optional[chess.Move]:
        """
        Kiểm tra và trả về nước đi hợp lệ từ dự đoán.

        Args:
            board (chess.Board): Đối tượng bàn cờ.
            from_square (int): Ô bắt đầu.
            to_square (int): Ô đích.
            is_promotion (float): Xác suất phong cấp (0 hoặc 1).

        Returns:
            Optional[chess.Move]: Nước đi hợp lệ hoặc None nếu không hợp lệ.
        """
        promotion = chess.QUEEN if is_promotion > 0.5 else None
        move = chess.Move(from_square, to_square, promotion=promotion)
        if move in board.legal_moves:
            return move
        return None

    def get_best_legal_move(self, board: chess.Board, game_state: Dict, top_k: int = 5) -> Optional[chess.Move]:
        """
        Lấy nước đi tốt nhất và hợp lệ từ dự đoán.

        Args:
            board (chess.Board): Đối tượng bàn cờ.
            game_state (Dict): Trạng thái ván cờ.
            top_k (int): Số nước đi cần xem xét.

        Returns:
            Optional[chess.Move]: Nước đi tốt nhất hoặc None nếu không có.
        """
        from chess_data_preprocessor import ChessDataPreprocessor
        preprocessor = ChessDataPreprocessor(history_length=self.history_length)
        X_processed = preprocessor.preprocess_input([game_state])

        from_probs, to_probs, is_promotion_probs, value = self.model.predict(
            [X_processed["board"], X_processed["history"], X_processed["white_elo"],
             X_processed["black_elo"], X_processed["side_to_move"], X_processed["castling_rights"],
             X_processed["move_count"], X_processed["game_phase"]], verbose=0
        )

        top_from_squares = np.argsort(from_probs[0])[-top_k:][::-1]
        top_to_squares = np.argsort(to_probs[0])[-top_k:][::-1]
        is_promotion = is_promotion_probs[0, 0]

        for from_square in top_from_squares:
            for to_square in top_to_squares:
                move = self.get_legal_move(board, from_square, to_square, is_promotion)
                if move:
                    return move

        legal_moves = list(board.legal_moves)
        if legal_moves:
            return np.random.choice(legal_moves)
        return None