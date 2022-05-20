"""
Script to train the LSTM Learner on the test and validation captions and images.
"""

from pathlib import Path
import os
import argparse

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model

from models.LSTM_learner import LSTM_learner
from data_utils.load_data import load_train, load_val


# Globals
model_folder = Path("models/")
logs_folder = Path("logs/")


def train_LSTM_learner():
    image_representations_train, captions_train, _ = load_train()
    image_representations_val, captions_val, _ = load_val()

    model = LSTM_learner()
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=1, verbose=1, mode="auto"
    )
    tensorboard_callback = TensorBoard(
        log_dir=logs_folder, update_freq=50
    )  # write every 50 batches
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    model.fit(
        [
            image_representations_train,
            captions_train[:, :-1],  # remove last stopword from each caption
        ],
        captions_train,
        validation_data=(
            [image_representations_val, captions_val[:, :-1]],
            captions_val,
        ),
        batch_size=100,
        epochs=100,
        callbacks=[early_stopping_callback, tensorboard_callback],
    )

    try:
        model.save_weights(os.path.join(model_folder, "LSTM_learner_weights.h5"))
    except:
        print("Could not save model.")

    return model


def evaluate_LSTM_learner(LSTM_model):
    # Get data for evaluation
    image_representations_train, captions_train, _ = load_train()
    image_representations_val, captions_val, _ = load_val()

    # Final Evaluation scores from the trained model.
    scores = LSTM_model.evaluate(
        [image_representations_train, captions_train[:, 1:]],
        captions_train,
        return_dict=True,
    )
    print(
        "{} Evaluation. Categorical Cross Entropy: {}, Categorical Accuracy: {}".format(
            "Train", scores["loss"], scores["sparse_categorical_accuracy"]
        )
    )
    scores = LSTM_model.evaluate(
        [image_representations_val, captions_val[:, 1:]], captions_val, return_dict=True
    )
    print(
        "{} Evaluation. Categorical Cross Entropy: {}, Categorical Accuracy: {}".format(
            "Validation", scores["loss"], scores["sparse_categorical_accuracy"]
        )
    )


# Load arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--load", help="Load from models/ folder instead of training", action="store_true"
)
args = parser.parse_args()

# Load model or train from scratch
if args.load:
    print("Loading trained model.")
    try:
        LSTM_model = load_model(os.path.join(model_folder, "LSTM_learner.h5"))
    except:
        LSTM_model = LSTM_learner()
        LSTM_model.load_weights(os.path.join(model_folder, "LSTM_learner_weights.h5"))
else:
    print("Training model from scratch.")
    LSTM_model = train_LSTM_learner()

# Final evaluation of trained model
evaluate_LSTM_learner(LSTM_model)
""" OUTPUT
Train Evaluation. Categorical Cross Entropy: 2.25, Categorical Accuracy: 0.72
Validation Evaluation. Categorical Cross Entropy: 2.34, Categorical Accuracy: 0.72
"""