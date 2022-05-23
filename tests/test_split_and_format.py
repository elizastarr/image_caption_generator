import pytest

from src.data_utils.split_and_format import format_as_matrix


def test_format_as_matrix_outputs(image_representations_100, captions_metadata_100):
    # Given
    captions, _, word_to_idx, max_caption_length, _, _ = captions_metadata_100

    # When
    representations, labels = format_as_matrix(
        image_representations_100, captions, max_caption_length, word_to_idx
    )

    # Then
    assert representations.shape[0] == 5 * image_representations_100.shape[0]
    assert representations.shape[0] == labels.shape[0]
    assert labels.shape[1] == max_caption_length
    assert labels.dtype == "uint32"


def test_format_as_matrix_raises_error_for_num_images_captions(
    image_representations_100, captions_metadata_100
):
    # Given
    captions, _, word_to_idx, max_caption_length, _, _ = captions_metadata_100
    image_representations_99 = image_representations_100[:99]

    # When
    with pytest.raises(AssertionError) as error:
        format_as_matrix(
            image_representations_99, captions, max_caption_length, word_to_idx
        )

    # Then
    assert "Different number of representations and caption groups." in str(error.value)


def test_format_as_matrix_raises_error_for_not_5_captions(
    image_representations_100, captions_metadata_100
):
    # Given
    captions, _, word_to_idx, max_caption_length, _, _ = captions_metadata_100
    captions[0] = captions[0][:4]
    captions[1] = captions[1][:3]

    # When
    with pytest.raises(AssertionError) as error:
        format_as_matrix(
            image_representations_100, captions, max_caption_length, word_to_idx
        )

    # Then
    assert "Each image must have 5 captions" in str(error.value)
