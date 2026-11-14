"""WTFTP predictor with attention, Haar DWT and coefficient loss."""
from merit.modern_models import ModernPredictor, WTFTPWaveletCore, haar_dwt1d, haar_idwt1d

class WTFTP(ModernPredictor):
    def __init__(self, mean, scale, config=None):
        super().__init__("wtftp_w", mean, scale, config=config)

Model = WTFTP
