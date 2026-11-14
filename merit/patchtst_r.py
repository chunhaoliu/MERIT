"""PatchTST predictor with a last-observation position residual."""
from merit.modern_models import ModernPredictor, PatchCore

class PatchTSTR(ModernPredictor):
    def __init__(self, mean, scale, config=None):
        super().__init__("patchtst_r", mean, scale, config=config)

Model = PatchTSTR
