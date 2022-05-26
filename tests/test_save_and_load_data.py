import pytest

from src.data_utils import save_and_load_data as sld


def test_load_reps_captions_images_raises_error_stage():
    # Given
    stage = "stage_not_implemented"

    # When
    with pytest.raises(ValueError) as error:
        sld.load_reps_captions_images(stage)

    # Then
    assert "Stage must be either train, val, or test" in str(error.value)


def test_save_reps_captions_images_raises_error_stage(processed_data_100):
    # Given
    image_representations, captions, images = processed_data_100
    stage = "stage_not_implemented"

    # When
    with pytest.raises(ValueError) as error:
        sld.save_reps_captions_images(image_representations, captions, images, stage)

    # Then
    assert "Stage must be either train, val, or test" in str(error.value)
