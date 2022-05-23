from src.data_utils.caption_preprocessing import get_caption_dictionaries


def test_get_caption_dictionaries_outputs(raw_data_100):
    # Given
    _, captions = raw_data_100

    # When
    (
        idx_to_word,
        word_to_idx,
        max_caption_length,
        total_words,
        num_unique_words,
    ) = get_caption_dictionaries(captions)

    # Then
    assert total_words == 5905
    assert max_caption_length == 32
    assert num_unique_words == 798
    assert len(idx_to_word) == len(word_to_idx) == num_unique_words
