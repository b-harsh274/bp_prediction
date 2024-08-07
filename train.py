import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import tensorflow as tf

from model import TimeseriesUNetModel
from data_preprocessing import DataPreprocessor

class Trainer:
    def __init__(self, output_path):
        self.output_path = output_path
        self.model = TimeseriesUNetModel().Timeseries_Unet()

    def train_ecg(self, signal_path, label_path, epochs=300, split_ratio=0.15, start_lr=1e-4, warmup_lr=1e-5, step_drop=0.5, epoch_step=100, batch_size=4):
        data_preprocessor = DataPreprocessor()
        x, y = data_preprocessor.ecg_data_preprocess(signal_path, label_path)

        if x is None or y is None:
            print('Data preprocessing failed, skipping execution')
            return

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=split_ratio, random_state=42)
        print(X_train.shape, y_train.shape)

        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=start_lr), loss='binary_crossentropy')

        current_date = datetime.now().strftime('%d%m%y_%H%M')
        checkpoint_path = os.path.join(self.output_path, f'feature_extraction_model_weights_{current_date}.keras')

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        )

        def step_decay(epoch):
            initial_lr = start_lr  # Initial learning rate
            drop_factor = step_drop  # Factor by which the learning rate will be reduced
            epochs_drop = epoch_step  # Number of epochs after which the learning rate will be reduced
            new_lr = initial_lr * tf.math.pow(drop_factor, tf.math.floor((1 + epoch) / epochs_drop))
            return float(new_lr)

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay, verbose=1)

        self.model.fit(X_train, y_train,
                       validation_data=(X_test, y_test),
                       epochs=epochs,
                       batch_size=batch_size, verbose=1,
                       callbacks=[model_checkpoint_callback, lr_scheduler])
