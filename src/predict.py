"""
Script to predict captions using
    1) a decoder with the LSTM learner weights and
    2) test image representations.
Predictions are saved in the data/processed directory in word format (not integers).
"""

from pathlib import Path
import os
import argparse
import pickle

from src.data.load_data import load_test, load_idx_word_dicts, load_predictions
from src.analysis.bleu_scores import get_bleu_scores
from src.analysis.visualization_utils import (
    show_10_images_and_captions_grid,
    bleu_score_histogram,
)
from models.decoder import Decoder


model_folder = Path("models/")
data_folder = Path("data/processed")


def predict_decoder():
    idx_to_word, _ = load_idx_word_dicts()
    image_representations_test, _, _ = load_test()

    # Get decoder (with LSTM weights)
    decoder = Decoder()
    decoder.build(input_shape=(5000, 20480))
    decoder.load_weights(
        os.path.join(model_folder, "LSTM_learner.h5"), by_name=True, skip_mismatch=True
    )

    # Get predictions
    predictions_idx = decoder.predict(image_representations_test)
    predictions_word = [
        [idx_to_word.get(key) for key in prediction] for prediction in predictions_idx
    ]

    # Save predictions
    pickle.dump(
        predictions_word,
        open(os.path.join(data_folder, "predictions_word_test.pkl"), "wb"),
    )

    return predictions_word


def show_bleu_scores(predictions):
    _, captions_test, images_test = load_test()
    idx_to_word, _ = load_idx_word_dicts()
    captions_word = [
        [idx_to_word.get(key) for key in caption] for caption in captions_test
    ]

    # Calculate BLEU scores
    independent_bleu_scores = get_bleu_scores(
        captions_word, predictions, smoothing=1, independent=True
    )
    print(
        "Independent BLEU score example: {}".format(independent_bleu_scores.iloc[0, :])
    )

    bleu_score_histogram(independent_bleu_scores)


def show_prediciton_examples(predictions):
    _, _, images_test = load_test()
    show_10_images_and_captions_grid(images_test, predictions, encoded=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load",
        help="Load predictions from data/ folder instead of predicting",
        action="store_true",
    )
    args = parser.parse_args()

    if args.load:
        print("Loading predictions from data/ folder.")
        predictions = load_predictions()
    else:
        print("Retrieving predictions from decoder.")
        predictions = predict_decoder()

    show_bleu_scores(predictions)
    show_prediciton_examples(predictions)
