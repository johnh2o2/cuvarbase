"""GPU (CUDA) implementations of period-finding algorithms.

Re-exports the CUDA-backed Cython classes under a clean namespace:
    from periodfind.gpu import ConditionalEntropy, AOV, LombScargle
"""

from periodfind.aov import AOV
from periodfind.bls import BoxLeastSquares
from periodfind.ce import ConditionalEntropy
from periodfind.fpw import FPW
from periodfind.ls import LombScargle
from periodfind.mf import MatchedFilter
from periodfind.mhf import MultiHarmonicFourier
from periodfind.vn import ViterbiNarrowband

__all__ = ["ConditionalEntropy", "AOV", "LombScargle", "FPW", "BoxLeastSquares", "MatchedFilter", "MultiHarmonicFourier", "ViterbiNarrowband"]
