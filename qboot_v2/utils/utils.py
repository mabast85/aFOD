#!/usr/bin/env python
"""Useful functions."""

import numpy as np


#=========================================================================================
# Matteo Bastiani, Michiel Cottaar
# 01-06-2017, FMRIB, Oxford
#=========================================================================================

def round_bvals(bvals, verbose=False):
    # Round the bvals to the median value for each identified shell
    tol = 100   # Tolerance
    selected = np.zeros(bvals.size, dtype='bool')
    same_b = abs(bvals[:, None] - bvals[None, :]) <= tol
    res_b = bvals.copy()
    while not selected.all():
        use = np.zeros(selected.size, dtype='bool')
        use[np.where(~selected)[0][0]] = True
        nuse = 0
        while (use.sum() != nuse):
            nuse = use.sum()
            use = same_b[use].any(0)
            if (use & selected).any():
                raise ValueError('Re-using the same indices (%s)' % str(np.where(use & selected))[0])
        median_b = np.median(bvals[use])
        if verbose:
            print('found b-shell of %i orientations with b-value %f' % (nuse, median_b))
        res_b[use] = median_b
        selected[use] = True
    res_b[res_b <= tol] = 0
    return res_b

