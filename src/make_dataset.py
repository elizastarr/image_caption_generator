# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pickle

from data_utils.image_representations import get_image_representations
from data_utils.caption_preprocessing import get_caption_dictionaries
from data_utils.split_and_format import train_test_val_split, format_as_matrix


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: Path, output_filepath: Path):
    """Runs data processing scripts to turn raw data from data/raw into processed data in data/processed.

    Parameters
    ----------
    input_filepath : Path
        Path of raw input data
    output_filepath : Path
        Path to folder to save processed data
    """

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    print(f"Loading data from {input_filepath}")
    raw_file_path = os.path.join(input_filepath, "Flickr8k_processed.pkl")
    images, captions = pickle.load(open(raw_file_path, "rb"))
    print("images shape: ", images.shape, " captions length: ", len(captions))

    print("Retreiving image representations...")
    image_representations = get_image_representations(images)

    print("Encoding and analyzing captions...")
    (
        idx_to_word,
        word_to_idx,
        max_caption_length,
        total_words,
        num_words,
    ) = get_caption_dictionaries(captions)

    print(f"Splitting data into train, test, and validation...")
    images_train, images_test, images_val = train_test_val_split(images)
    (
        image_representations_train,
        image_representations_test,
        image_representations_val,
    ) = train_test_val_split(image_representations)
    captions_train, captions_test, captions_val = train_test_val_split(captions)

    print(f"Reformatting image representations and captions...")
    image_representations_train, captions_train = format_as_matrix(
        image_representations_train, captions_train, max_caption_length, word_to_idx
    )
    image_representations_test, captions_test = format_as_matrix(
        image_representations_test, captions_test, max_caption_length, word_to_idx
    )
    image_representations_val, captions_val = format_as_matrix(
        image_representations_val, captions_val, max_caption_length, word_to_idx
    )

    print(f"Saving processed data to f{output_filepath}...")
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)

    # Dictionaries
    pickle.dump(
        idx_to_word, open(os.path.join(output_filepath, "idx_to_word.pkl"), "wb")
    )
    pickle.dump(
        word_to_idx, open(os.path.join(output_filepath, "word_to_idx.pkl"), "wb")
    )

    # Images
    pickle.dump(
        images_train, open(os.path.join(output_filepath, "images_train.pkl"), "wb")
    )
    pickle.dump(
        images_test, open(os.path.join(output_filepath, "images_test.pkl"), "wb")
    )
    pickle.dump(images_val, open(os.path.join(output_filepath, "images_val.pkl"), "wb"))

    # Image representations
    pickle.dump(
        image_representations_train,
        open(os.path.join(output_filepath, "image_representations_train.pkl"), "wb"),
    )
    pickle.dump(
        image_representations_test,
        open(os.path.join(output_filepath, "image_representations_test.pkl"), "wb"),
    )
    pickle.dump(
        image_representations_val,
        open(os.path.join(output_filepath, "image_representations_val.pkl"), "wb"),
    )

    # Captions
    pickle.dump(
        captions_train, open(os.path.join(output_filepath, "captions_train.pkl"), "wb")
    )
    pickle.dump(
        captions_test, open(os.path.join(output_filepath, "captions_test.pkl"), "wb")
    )
    pickle.dump(
        captions_val, open(os.path.join(output_filepath, "captions_val.pkl"), "wb")
    )

    print("Processing complete.")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
