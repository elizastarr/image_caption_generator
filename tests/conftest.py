import os

import pytest
import pickle

from src.data_utils.load_data import load_test
from src.data_utils.image_representations import get_image_representations
from src.data_utils.caption_preprocessing import get_caption_dictionaries


@pytest.fixture(scope="session")
def processed_data_100():
    image_representations, captions, images = load_test()
    return image_representations[:100], captions[:100], images[:100]


@pytest.fixture(scope="session")
def raw_data_100():
    raw_file_path = os.path.join("data/raw", "Flickr8k_processed.pkl")
    images, captions = pickle.load(open(raw_file_path, "rb"))
    return images[:100], captions[:100]


@pytest.fixture(scope="session")
def image_representations_100(raw_data_100):
    images, _ = raw_data_100
    image_representations = get_image_representations(images)
    return image_representations


@pytest.fixture(scope="session")
def captions_metadata_100(raw_data_100):
    _, captions = raw_data_100
    metadata = get_caption_dictionaries(captions)
    return captions, *metadata
