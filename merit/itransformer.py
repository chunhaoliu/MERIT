"""Inverted Transformer predictor adapter."""
from merit.modern_models import ModernPredictor, InvertedTransformerCore

class iTransformer(ModernPredictor):
    def __init__(self, mean, scale, config=None):
        super().__init__("itransformer", mean, scale, config=config)

Model = iTransformer
