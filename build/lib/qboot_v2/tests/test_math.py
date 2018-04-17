from qboot_v2.utils import math as qbm
import numpy as np


def test_rotation():
    np.random.seed(1234)
    v1 = np.random.randn(3)
    v1 /= np.linalg.norm(v1)
    v2 = np.random.randn(3)
    v2 /= np.linalg.norm(v2)
    # v2 = [0, 0, 1]

    R = qbm.get_rotation(v1, v2)
    pred_v2 = np.dot(v1, R)
    
    print(pred_v2, v1, v2)
    assert abs(pred_v2 - v2).max() < 1E-4

