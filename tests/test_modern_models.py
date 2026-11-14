import unittest
import numpy as np
import torch

from merit.modern_models import haar_dwt1d, haar_idwt1d, ModernPredictor

class WaveletModels(unittest.TestCase):
    def test_haar_dwt_matches_orthonormal_reference(self):
        x=torch.tensor([[[1.,3.,2.,6.]]]); lo,hi=haar_dwt1d(x)
        self.assertTrue(torch.allclose(lo,torch.tensor([[[2.8284271,5.6568542]]]),atol=1e-6))
        self.assertTrue(torch.allclose(hi,torch.tensor([[[-1.4142135,-2.8284271]]]),atol=1e-6))
        self.assertTrue(torch.allclose(haar_idwt1d(lo,hi),x,atol=1e-6))

    def test_haar_dwt_matches_pywt_when_available(self):
        try: import pywt
        except ImportError: self.skipTest("pywt not installed")
        rng=np.random.RandomState(0); x=torch.as_tensor(rng.randn(2,3,10),dtype=torch.float64); lo,hi=haar_dwt1d(x)
        for b in range(2):
            for c in range(3):
                ref_lo,ref_hi=pywt.dwt(x[b,c].numpy(),"haar",mode="symmetric")
                self.assertTrue(np.allclose(lo[b,c].numpy(),ref_lo)); self.assertTrue(np.allclose(hi[b,c].numpy(),ref_hi))

    def test_haar_dwt_gradient_and_wtftp_interface(self):
        x=torch.randn(2,60,6,requires_grad=True); lo,hi=haar_dwt1d(torch.cat((x,x[:,:30]),dim=1).transpose(1,2)); haar_idwt1d(lo,hi).square().mean().backward()
        self.assertIsNotNone(x.grad); self.assertTrue(torch.isfinite(x.grad).all())
        model=ModernPredictor("wtftp_w",np.zeros(6),np.ones(6)); y,aux=model(x.detach())
        self.assertEqual(tuple(y.shape),(2,30,6)); self.assertEqual(aux.ndim,0); self.assertTrue(torch.isfinite(y).all())

if __name__ == "__main__": unittest.main()
