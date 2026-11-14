"""DLinear decomposition-linear predictor adapter."""
from merit.modern_models import ModernPredictor, DLinearCore

class DLinear(ModernPredictor):
    def __init__(self, mean, scale, config=None):
        super().__init__("dlinear", mean, scale, config=config)

Model = DLinear
