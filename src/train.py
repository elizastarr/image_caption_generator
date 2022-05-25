"""
Script to train the LSTM Learner on the test and validation captions and images.
"""

import os
import argparse

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model

from models.LSTM_learner import LSTM_learner
from data_utils.save_and_load_data import load_representations_captions_images
from config.core import config


def train(model):
    (
        image_representations_train,
        captions_train,
        _,
    ) = load_representations_captions_images("train")
    image_representations_val, captions_val, _ = load_representations_captions_images(
        "val"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=1, verbose=1, mode="auto"
    )
    tensorboard_callback = TensorBoard(
        log_dir=config.log_folder, update_freq=50
    )  # write every 50 batches
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.learning_rate),
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
        batch_size=config.batch_size,
        epochs=config.epochs,
        callbacks=[early_stopping_callback, tensorboard_callback],
    )

    try:
        model.save_weights(
            os.path.join(config.model_folder, config.filenames.model_weights)
        )
    except:
        print("Could not save model.")

    return model


def evaluate_LSTM_learner(LSTM_model):
    # Get data for evaluation
    image_representations_val, captions_val, _ = load_representations_captions_images(
        "val"
    )

    # Final Evaluation scores from the trained model.
    scores = LSTM_model.evaluate(
        [image_representations_val, captions_val[:, 1:]], captions_val, return_dict=True
    )
    print(
        "{} Evaluation. Categorical Cross Entropy: {}, Categorical Accuracy: {}".format(
            "Validation", scores["loss"], scores["sparse_categorical_accuracy"]
        )
    )

    """ OUTPUT
    Validation Evaluation. Categorical Cross Entropy: 2.34, Categorical Accuracy: 0.72
    """


if __name__ == "__main__":
    # Load arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load",
        help="Load from models/ folder instead of training",
        action="store_true",
    )
    args = parser.parse_args()

    # Load model or train from scratch
    if args.load:
        print("Loading trained model.")
        try:
            LSTM_model = load_model(
                os.path.join(config.model_folder, config.filenames.model_weights)
            )
        except:
            LSTM_model = LSTM_learner()
            LSTM_model.load_weights(
                os.path.join(config.model_folder, config.filenames.model_weights)
            )
    else:
        print("Training model from scratch.")
        model = LSTM_learner()
        model = train(model)

    # Final evaluation of trained model
    evaluate_LSTM_learner(LSTM_model)
