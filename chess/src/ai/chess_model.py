import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, regularizers
import numpy as np
import chess
from typing import Dict, Tuple, Optional
import psutil

class ChessModel:
    """
    Class định nghĩa kiến trúc mô hình học máy cho AI cờ vua, dự đoán nước đi tiếp theo.
    """
    
    def __init__(self, config: Dict = None):
        """
        Khởi tạo mô hình với các tham số cấu hình.
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
        """
        x = history_input
        x = layers.GRU(32, return_sequences=False,
                       kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.Dense(16, activation='relu',
                        kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        return x

    def _process_metadata(self, side_to_move: tf.Tensor, castling_rights: tf.Tensor,
                         move_count: tf.Tensor, game_phase: tf.Tensor) -> tf.Tensor:
        """
        Xử lý metadata bằng MLP.
        """
        x = layers.Concatenate()([side_to_move, castling_rights, move_count, game_phase])
        x = layers.Dense(16, activation='relu',
                        kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.Dense(8, activation='relu',
                        kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        return x

    def _build_model(self) -> Model:
        """
        Xây dựng kiến trúc mô hình hoàn chỉnh.
        """
        board_input = layers.Input(shape=(8, 8, 12), name='board')
        history_input = layers.Input(shape=(self.history_length, 4), name='history')
        side_to_move_input = layers.Input(shape=(1,), name='side_to_move')
        castling_rights_input = layers.Input(shape=(4,), name='castling_rights')
        move_count_input = layers.Input(shape=(1,), name='move_count')
        game_phase_input = layers.Input(shape=(3,), name='game_phase')

        board_features = self._process_board(board_input)
        history_features = self._process_history(history_input)
        metadata_features = self._process_metadata(side_to_move_input, castling_rights_input,
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

        from_square_output = layers.Dense(self.num_squares, activation='softmax', name='from_square')(x)
        to_square_output = layers.Dense(self.num_squares, activation='softmax', name='to_square')(x)
        is_promotion_output = layers.Dense(1, activation='sigmoid', name='is_promotion')(x)

        model = Model(
            inputs=[board_input, history_input, side_to_move_input, castling_rights_input,
                    move_count_input, game_phase_input],
            outputs=[from_square_output, to_square_output, is_promotion_output]
        )
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss={
                'from_square': tf.keras.losses.CategoricalCrossentropy(),
                'to_square': tf.keras.losses.CategoricalCrossentropy(),
                'is_promotion': tf.keras.losses.BinaryCrossentropy()
            },
            metrics={
                'from_square': ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')],
                'to_square': ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')],
                'is_promotion': ['accuracy']
            }
        )
        return model

    def train(self, train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset, 
              epochs: int = 100, callbacks: list = None, steps_per_epoch: Optional[int] = None,
              validation_steps: Optional[int] = None) -> Dict:
        """
        Huấn luyện mô hình trên dữ liệu.
        """
        if callbacks is None:
            callbacks = []
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ))
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            'models/chess_model_best.keras', monitor='val_loss', save_best_only=True, verbose=1
        ))
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
        ))

        # Giám sát RAM trước khi huấn luyện
        mem_info = psutil.virtual_memory()
        print(f"RAM usage before fit: {mem_info.percent}% ({mem_info.used/1024**3:.2f}GB / {mem_info.total/1024**3:.2f}GB)")

        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=1
        )

        return history.history

    def predict_move(self, game_state: Dict, board: chess.Board) -> Optional[chess.Move]:
        """
        Dự đoán nước đi tốt nhất cho trạng thái ván cờ.
        """
        return self.get_best_legal_move(board, game_state, top_k=5)

    def save(self, filepath: str) -> None:
        """
        Lưu mô hình vào file.
        """
        self.model.save(filepath, include_optimizer=True)

    @classmethod
    def load(cls, filepath: str) -> 'ChessModel':
        """
        Tải mô hình từ file.
        """
        instance = cls()
        instance.model = tf.keras.models.load_model(filepath)
        return instance

    def get_legal_move(self, board: chess.Board, from_square: int, to_square: int, 
                       is_promotion: float) -> Optional[chess.Move]:
        """
        Kiểm tra và trả về nước đi hợp lệ từ dự đoán.
        """
        promotion = chess.QUEEN if is_promotion > 0.5 else None
        move = chess.Move(from_square, to_square, promotion=promotion)
        if move in board.legal_moves:
            return move
        return None

    def get_best_legal_move(self, board: chess.Board, game_state: Dict, top_k: int = 5) -> Optional[chess.Move]:
        """
        Lấy nước đi tốt nhất và hợp lệ từ dự đoán.
        """
        from chess_data_preprocessor import ChessDataPreprocessor
        preprocessor = ChessDataPreprocessor(history_length=self.history_length)
        X_processed = preprocessor.preprocess_input([game_state])

        from_probs, to_probs, is_promotion_probs = self.model.predict(
            [X_processed["board"], X_processed["history"], X_processed["side_to_move"],
             X_processed["castling_rights"], X_processed["move_count"], X_processed["game_phase"]],
            verbose=0
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