import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churn.monitor import psi

def test_psi_zero_for_same():
    s = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert psi(s, s) < 0.01

def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    assert psi(pd.Series(rng.normal(size=5000)), pd.Series(rng.normal(1, 1, size=5000))) > 0.25
