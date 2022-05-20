from pathlib import Path

from data_utils.load_data import load_train, load_val, load_test, load_idx_word_dicts
from analysis.visualization_utils import (
    show_10_images_and_captions_grid,
    show_random_image_and_caption_individual,
)

if __name__ == "__main__":

    data_folder = Path("data/processed/")

    _, captions_train, images_train = load_train()
    _, captions_val, images_val = load_val()
    _, captions_test, images_test = load_test()
    idx_to_word, _ = load_idx_word_dicts()

    show_random_image_and_caption_individual(
        images_train,
        captions_train,
        idx_to_word,
        1,
        file_name="example_train_image.png",
    )
    show_10_images_and_captions_grid(
        images_test, captions_test, file_name="example_images.png"
    )
