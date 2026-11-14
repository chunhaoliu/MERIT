"""Seq2Seq LSTM predictor adapter."""
from merit.modern_models import ModernPredictor

class Seq2SeqLSTM(ModernPredictor):
    def __init__(self, mean, scale, config=None):
        super().__init__("wtftp", mean, scale, config=config)

Model = Seq2SeqLSTM
