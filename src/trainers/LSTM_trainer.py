import os
from typing import Tuple

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras import optimizers

from config.core import config
from src.models.LSTM_learner import LSTMLearner


class BaseTrain(object):
    def __init__(self, model, training_data, validation_data):
        self.model = model
        self.training_data = training_data
        self.validation_data = validation_data

    def train(self):
        raise NotImplementedError


class LSTMTrainer(BaseTrain):
    def __init__(
        self,
        model: LSTMLearner,
        training_data: Tuple,
        validation_data: Tuple,
        model_checkpoint_callback: bool = True,
        early_stopping_callback: bool = True,
        tensorboard_callback: bool = True,
    ):
        super(LSTMTrainer, self).__init__(model, training_data, validation_data)
        self.callbacks = []
        self.train_loss = []
        self.val_loss = []
        self.model_checkpoint_callback = model_checkpoint_callback
        self.early_stopping_callback = early_stopping_callback
        self.tensorboard_callback = tensorboard_callback
        self.init_callbacks()

    def init_callbacks(self):
        if self.model_checkpoint_callback:
            self.callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(
                        config.model_folder,
                        "%s-{epoch:02d}-{val_loss:.2f}.hdf5" % config.experiment_name,
                    ),
                    monitor=config.model_checkpoint.monitor,
                    mode=config.model_checkpoint.mode,
                    save_best_only=config.model_checkpoint.save_best_only,
                    save_weights_only=config.model_checkpoint.save_weights_only,
                    verbose=config.verbose,
                )
            )
        if self.tensorboard_callback:
            self.callbacks.append(
                TensorBoard(
                    log_dir=config.log_folder,
                    update_freq=config.tensorboard.update_freq,
                )
            )

        if self.early_stopping_callback:
            self.callbacks.append(
                EarlyStopping(
                    monitor=config.early_stopping.monitor,
                    mode=config.early_stopping.mode,
                    min_delta=config.early_stopping.min_delta,
                    patience=config.early_stopping.patience,
                    verbose=config.verbose,
                )
            )

    def train(self):
        self.model.compile(
            optimizer=optimizers.Adam(config.learning_rate),
            loss=config.loss,
            metrics=config.metric,
        )
        history = self.model.fit(
            x=self.training_data[0],
            y=self.training_data[1],
            validation_data=self.validation_data,
            epochs=config.epochs,
            verbose=config.verbose,
            batch_size=config.batch_size,
            callbacks=self.callbacks,
        )
        return self.model

    def evaluate(self):
        scores = self.model.evaluate(
            x=self.validation_data[0], y=self.validation_data[1], return_dict=True
        )

        print(
            "{} Evaluation. Categorical Cross Entropy: {}, Categorical Accuracy: {}".format(
                "Validation", scores["loss"], scores[config.metric]
            )
        )
