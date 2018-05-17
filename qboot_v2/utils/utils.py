#!/usr/bin/env python
"""Useful functions."""

import numpy as np
import nibabel as nib
import os
from qboot_v2.utils import math as qbm
import scipy.optimize as opt


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


def get_peaks(sphere, fod_file, mask_file, max_order, n=3, sym=True, f=0.1, min_angle=25,
              non_lin=False, save_results=False, verbose=False):
    mask = (nib.load(mask_file)).get_data()
    fod_obj = nib.load(fod_file)
    fod = fod_obj.get_data()

    if sym:
        sh_coeff = 'even'
    else:
        sh_coeff = 'all'

    vertices = sphere[:, 0:3]
    edges = (sphere[:, 3:]).astype(int)
    vertices_sph = qbm.cart2sph(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    vertices_sh = qbm.get_sh(vertices_sph[:, 1], vertices_sph[:, 2], max_order, coeffs=sh_coeff)

    ii = np.where(mask)
    xs, ys, zs = ii

    n_coeffs, m, l, c1, c2 = qbm.get_m_l(max_order, sh_coeff)

    #INITIALIZE PEAK AND AMPLITUDE ARRAYS
    peaks = np.zeros(list(mask.shape) + [n*3], dtype=np.float)
    amplitudes = np.zeros(list(mask.shape) + [n], dtype=np.float)
    for x, y, z in zip(xs, ys, zs):
        _amplitudes = np.dot(vertices_sh, fod[x, y, z, :])
        _amplitudes[_amplitudes < 0] = 0
        #maxima = _amplitudes[_amplitudes >= f]
        #maxima_i = np.where(_amplitudes >= f)[0]
        maxima, maxima_i = find_maxima(edges, _amplitudes, f=f)
        # IF SYM, CONSIDER ONLY Z>0
        if sym:
            maxima = maxima[vertices[maxima_i, 2] > 0]
            maxima_i = maxima_i[vertices[maxima_i, 2] > 0]
            
        if maxima.size > 0:
            _peaks = vertices[maxima_i, :]
            #IF REFINE
            if non_lin:
                t = np.empty(maxima.size)
                p = np.empty(maxima.size)
                for i in range(0, maxima.size):
                    x0 = vertices_sph[maxima_i[i], 1:3]
                    res = opt.minimize(get_neg_amplitude, x0,
                                       args=(fod[x, y, z, :], max_order, sh_coeff, (n_coeffs, m, l, c1, c2)),
                                       tol=1e-4, method='Nelder-Mead')
                    t[i], p[i] = res.x
                    maxima[i] = -res.fun
                    if verbose:
                        if not res.success:
                            print('Minimun not found for peak ' + str(i) + ' in voxel ' + str([x, y, z]))
                _peaks = qbm.sph2cart(vertices_sph[maxima_i, 0], t, p)
                
            if maxima.size > 1:
                clean_maxima, clean_i = clean_peaks(maxima, _peaks, min_angle)
                sorted_i = np.argsort(clean_maxima)[::-1]
                if sorted_i.size > n:
                    sorted_i = sorted_i[0:n]
                clean_maxima = clean_maxima[sorted_i]
                clean_i = clean_i[sorted_i]
                peaks[x, y, z, 0:3*clean_i.size] = _peaks[clean_i, :].flatten()
                amplitudes[x, y, z, 0:clean_i.size] = clean_maxima
            else:
                peaks[x, y, z, 0:3] = _peaks
                amplitudes[x, y, z, 0] = maxima
            
            

    #WRITE ARRAYS TO FILE (IF NEEDED)
    if save_results:
        print('Storing peaks and amplitudes')
        d = os.path.dirname(fod_file)
        for i in range(0, n):
            nib.Nifti1Image(peaks[:, :, :, 3*i:3*(i+1)], None, fod_obj.header).to_filename(d + '/peak' + str(i+1) + '.nii.gz')
            nib.Nifti1Image(amplitudes[:, :, :, i], None, fod_obj.header).to_filename(d + '/peak' + str(i+1) + '_amplitude.nii.gz')


    #RETURN PEAK AND AMPLITUDE ARRAYS
    return peaks, amplitudes

def clean_peaks(m, v, min_angle):
    check_i = np.ones(m.size)
    
    v_dot = np.dot(v, v.T)
    v_dot[v_dot > 1] = 1
    v_dot[v_dot < -1] = -1
    v_alpha = np.arccos(v_dot)    

    min_angle_rad = np.deg2rad(min_angle)

    for i in range(0, m.size):
        #v_dot = np.dot(v[m_i[i], :], v.T)
        #v_dot[v_dot > 1] = 1
        #v_dot[v_dot < -1] = -1
        #v_alpha = np.arccos(v_dot)
        #m_neighs = a[v_alpha[m_i[i], :] < np.deg2rad(min_angle)]
        m_neighs = m[v_alpha[i, :] < min_angle_rad]
        if (m[i] < m_neighs).sum() >= 1:
            check_i[i] = 0
    
    return m[check_i > 0], np.where(check_i > 0)[0]
        
def find_maxima(edges, amplitudes, f=0):
    maxima_check = np.ones(edges.shape[0])
    maxima_check[amplitudes < f] = 0
    
    for i in np.where(maxima_check > 0)[0]:
        _edges = edges[i, edges[i, :] != -1]
        _amplitudes = amplitudes[_edges]
        if (amplitudes[i] < _amplitudes).sum() >= 1:
            maxima_check[i] = 0

    return amplitudes[maxima_check > 0], np.where(maxima_check > 0)[0]


def get_neg_amplitude(p_sph, fod, max_order, sh_coeff, nml):
    p_sh = qbm.get_sh(np.array([p_sph[0]]), np.array([p_sph[1]]), max_order, coeffs=sh_coeff, nml=nml)

    return -np.dot(p_sh, fod)