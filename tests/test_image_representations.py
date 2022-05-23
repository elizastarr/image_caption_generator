import pytest
from src.data_utils import image_representations
from src.data_utils.image_representations import get_image_representations


def test_get_image_representations_raises_error_for_shape(raw_data_100):
    # Given
    images, _ = raw_data_100
    b, h, w, c = images.shape
    images = images.reshape(b, h // 2, w * 2, c)

    # When
    with pytest.raises(ValueError) as error:
        get_image_representations(images)

    # Then
    assert "expected shape=(None, 128, 128, 3)" in str(error.value)


def test_get_image_representations_shape(raw_data_100):
    # Given
    images, _ = raw_data_100

    # When
    representations = get_image_representations(images)

    # Then
    assert representations.shape == (len(images), 20480)
