from config.core import config
from src.models.LSTM_learner import LSTMLearner
from src.models.decoder import Decoder
from src.trainers.LSTM_trainer import LSTMTrainer

def test_LSTMLearner():
    # When
    model = LSTMLearner(max_caption_length = config.max_caption_length, num_unique_words=config.num_unique_words)

    # Then
    assert model

def test_Decoder():
    # When
    model = Decoder(max_caption_length = config.max_caption_length, num_unique_words=config.num_unique_words)

    # Then
    assert model

def test_LSTMTrainer(processed_data_100):
    # Given
    image_representations, captions, _ = processed_data_100
    data = (
        [
            image_representations,
            captions[:, :-1],
        ],
        captions,
    )
    model = LSTMLearner(max_caption_length = config.max_caption_length, num_unique_words=config.num_unique_words)

    # When
    trainer = LSTMTrainer(model=model, training_data=data, validation_data=data)
    
    # Then
    assert trainer