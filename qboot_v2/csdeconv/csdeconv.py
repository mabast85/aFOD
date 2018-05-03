import ctypes
import multiprocessing as mp
import numpy as np
import nibabel as nib
import itertools
from qboot_v2.utils import math as qbm
from cvxopt import matrix, spmatrix
from cvxopt.solvers import options, qp

options['show_progress'] = False  # disable cvxopt output

# number of processes to parallelise csdeconv across
nprocs = 16


# Response function class
class Response(object):
    '''Response function

    We want to have the basic response function sh coefficients, the signal
    and a method to compute it, and a method to i/o
    '''

    def __init__(self, coefficients, max_order):
        '''Creates a new response function on the
        coefficients and max order'''
        self.coefficients = coefficients
        self.max_order = max_order
        
    @classmethod
    def get_response(cls, data_file, mask_file, bvals_file, bvecs_file, max_order, dti_basename=None, normalize=False):
        '''Computes the response function coefficients and signal
        from a set of voxels in mask'''

        resp = 0

        if dti_basename is None:
            raise ValueError(dti_basename + ' does not appear to be a valid dtifit basename')
        else:
            dti_v1 = (nib.load(dti_basename + '_V1.nii.gz')).get_data()
        
        bvals = np.genfromtxt(bvals_file, dtype=float)
        bvecs = np.genfromtxt(bvecs_file, dtype=float)
        mask = (nib.load(mask_file)).get_data()
        ii = np.where(mask)
        print('Found ' + str(np.count_nonzero(mask)) + ' masked voxels')
        
        data = (nib.load(data_file)).get_data()
        for x, y, z in zip(*ii):
            R = qbm.get_rotation(dti_v1[x, y, z, :], [0, 0, 1])
            rotated_bvecs = np.dot(R.T, bvecs[:, bvals > 100])
            rotated_bvecs_sph = qbm.cart2sph(rotated_bvecs[0, :], rotated_bvecs[1, :], rotated_bvecs[2, :])
            rotated_bvecs_sh = qbm.get_sh(rotated_bvecs_sph[:, 1], rotated_bvecs_sph[:, 2], max_order)
            s = data[x, y, z, bvals >= 100]
            s0 = np.mean(data[x, y, z, bvals < 100])
            if normalize:
                resp = resp + np.linalg.lstsq(rotated_bvecs_sh, s / s0)[0]
            else:
                resp = resp + np.linalg.lstsq(rotated_bvecs_sh, s)[0]
        
        resp /= np.count_nonzero(mask)
        return cls(resp, max_order)

    def get_rh(self):
        delta = qbm.get_delta(np.array([0]), np.array([0]), self.max_order)
        return self.coefficients[np.where(delta[0, :])] / delta[np.where(delta)]

    # I/O
    @classmethod
    def read_coefficients(cls, fname):
        '''Reads SH coefficients from a text file

        :arg fname: (str) input file name
        '''
        _coefficients = np.genfromtxt(fname, dtype=float, delimiter=' ')
        max_order = 2 * (_coefficients.size - 1)
        delta = qbm.get_delta(np.array([0]), np.array([0]), max_order)
        coefficients = np.zeros((delta.size), dtype=float)
        coefficients[np.where(delta[0, :])] = _coefficients
        return cls(coefficients=coefficients, max_order=max_order)

    def write_coefficients(self, fname):
        '''Writes SH coefficients to a text file

        :arg fname: (str) output file name
        '''
        delta = qbm.get_delta(np.array([0]), np.array([0]), self.max_order)
        np.savetxt(fname, self.coefficients[np.where(delta[0, :])], fmt='%.5f', delimiter=' ')


def get_csd_matrix(bvecs, response, max_order, sym=True):
    bvecs_sph = qbm.cart2sph(bvecs[0, :], bvecs[1, :], bvecs[2, :])
    bvecs_sh = qbm.get_sh(bvecs_sph[:, 1], bvecs_sph[:, 2], max_order)
    rh = response.get_rh()
    if response.max_order < max_order:
        rh = np.append(rh, np.ones((max_order - response.max_order)/2) * 1E-16)
    m, R = np.concatenate([[(m, rh[int(l/2)]) for m in range(-l, l+1)] for l in range(0, max_order+1, 2)], axis=0).T
    R = np.diag(R)
    C = np.dot(bvecs_sh, R)
    if sym:
        return C
    else:
        m, l = np.concatenate([[(m, l) for m in range(-l, l+1)] for l in range(0, max_order+1)], axis=0).T
        a = (np.diag((np.mod(l, 2) == 0))[np.mod(l, 2) == 0, :]).astype(int)
        return np.dot(C, a)


def get_weights(vertices, sigma=40):
    neighs = np.array(list(itertools.product([-1, 0, 1], repeat=3)))
    neighs = np.delete(neighs, 13, 0)   # Remove [0, 0, 0]
    d = np.linalg.norm(neighs, ord=2, axis=1)
    deg_mat = np.arccos(np.dot(neighs / d[:, np.newaxis], vertices.T))
    weights = np.exp(-deg_mat / np.deg2rad(sigma))
    weights[deg_mat > np.deg2rad(60)] = 0   # Do not consider vertices that are not aligned with any neighbouring voxel
    weights = weights / d[:, np.newaxis]    # Account for distance
    weights = weights / np.sum(weights, axis=0)[np.newaxis, :]   # Divide by the vertex-wise weight sum
    weights[np.isnan(weights)] = 0  # Check for nans
    return weights


def sdeconv(response, data_file, mask_file, bvals_file, bvecs_file, max_order, sym=False, out_file=None):
    # Load data
    bvals = np.genfromtxt(bvals_file, dtype=np.float32)
    bvecs = np.genfromtxt(bvecs_file, dtype=np.float32)
    mask = (nib.load(mask_file)).get_data()
    data_obj = nib.load(data_file)
    data = data_obj.get_data()

    # Get convolution matrix
    C = get_csd_matrix(bvecs[:, bvals > 100], response, max_order, sym)
    
    # Initialise output fod matrix
    fod = np.zeros(list(mask.shape) + [C.shape[1]], dtype=np.float32)

    ii = np.where(mask)
    xs, ys, zs = ii
    for x, y, z in zip(xs, ys, zs):
        s = data[x, y, z, bvals >= 100]
        fod[x, y, z, :] = np.linalg.lstsq(C, s)[0]

    if out_file is not None:
        nib.Nifti1Image(fod, None, data_obj.header).to_filename(out_file)

    return fod
    

def csdeconv(response, data_file, mask_file, bvals_file, bvecs_file, max_order, sym=False, l=0.1):
    # Load data
    bvals = np.genfromtxt(bvals_file, dtype=np.float32)
    bvecs = np.genfromtxt(bvecs_file, dtype=np.float32)
    mask = (nib.load(mask_file)).get_data()
    data_obj = nib.load(data_file)
    data = data_obj.get_data()

    #mask[:, :, :40] = 0
    #mask[:, :, 50:] = 0
    mask[:, :, 0] = 0
    ii = np.where(mask)

    # Get CSD matrices
    B = np.genfromtxt('/Users/matteob/qboot_v2/qboot_v2/utils/ico_5.txt', dtype=np.float32)
    B_sph = qbm.cart2sph(B[:, 0], B[:, 1], B[:, 2])
    C = get_csd_matrix(bvecs[:, bvals > 100], response, max_order, sym)
    if sym is False:
        B_sh = qbm.get_sh(B_sph[:, 1], B_sph[:, 2], max_order, c='all')
        B_neg_sph = qbm.cart2sph(-B[:, 0], -B[:, 1], -B[:, 2])
        B_neg_sh = qbm.get_sh(B_neg_sph[:, 1], B_neg_sph[:, 2], max_order, c='all')
        l = l * C.shape[0] * (response.get_rh())[0] / B.shape[0]
        C = np.concatenate((C, l*B_sh), axis=0)
        w = get_weights(B)
        prev_fod = sdeconv(response, data_file, mask_file,  bvals_file, bvecs_file, max_order, sym=sym)
    else:
        B_sh = qbm.get_sh(B_sph[:, 1], B_sph[:, 2], max_order)
    H = np.dot(C.T, C)
    
    fod = np.zeros(list(mask.shape) + [B_sh.shape[1]], dtype=np.float32)
    
    # create shared memory arrays
    shared_fod = mp.RawArray(ctypes.c_float, fod.size)
    shared_data = mp.RawArray(ctypes.c_float, data.size)
    
    fod_ptr = np.ctypeslib.as_array(shared_fod).reshape(fod.shape)
    data_ptr = np.ctypeslib.as_array(shared_data).reshape(data.shape)

    data_ptr[:] = data
    csdeconv_fit.shared_fod = shared_fod
    csdeconv_fit.shared_data = shared_data

    if sym is False:
        shared_prev_fod = mp.RawArray(ctypes.c_float, prev_fod.size)
        prev_fod_ptr = np.ctypeslib.as_array(shared_prev_fod).reshape(prev_fod.shape)
        prev_fod_ptr[:] = prev_fod
        csdeconv_fit.shared_prev_fod = shared_prev_fod
    
    # Chunk up indices
    x, y, z = ii
    nvox = len(x)
    chunk_size = nvox / nprocs
    chunk_end = chunk_size * nprocs

    iixs = [x[i * chunk_size:i * chunk_size + chunk_size] for i in range(nprocs)] + [x[chunk_end:]]
    iiys = [y[i * chunk_size:i * chunk_size + chunk_size] for i in range(nprocs)] + [y[chunk_end:]]
    iizs = [z[i * chunk_size:i * chunk_size + chunk_size] for i in range(nprocs)] + [z[chunk_end:]]

    # create arguments for each child process
    args = [((xs, ys, zs), fod.shape, data.shape, bvals, H, C, B_sh, sym, B_neg_sh, w, l) 
            for xs, ys, zs in zip(iixs, iiys, iizs)]

    # Create child processes
    pool = mp.Pool(processes=nprocs)

    pool.starmap(csdeconv_fit, args)

    nib.Nifti1Image(fod_ptr, None, data_obj.header).to_filename('/Users/matteob/Desktop/fslcourse_update/fsl_course_data/fdt1/subj1/acsd_test.nii.gz')
    
    
def csdeconv_fit(vox_list, fod_shape, data_shape, bvals, H, C, B, sym, B_neg, w, l):
    fod = csdeconv_fit.shared_fod
    data = csdeconv_fit.shared_data
    prev_fod = csdeconv_fit.shared_prev_fod

    fod = np.ctypeslib.as_array(fod).reshape(fod_shape)
    data = np.ctypeslib.as_array(data).reshape(data_shape)
    prev_fod = np.ctypeslib.as_array(prev_fod).reshape(fod_shape)
    
    neighs = np.array(list(itertools.product([-1, 0, 1], repeat=3)))
    neighs = np.delete(neighs, 13, 0)   # Remove [0, 0, 0]

    for x, y, z in zip(*vox_list):
        s = data[x, y, z, bvals >= 100]
        if sym:
            f = np.dot(-C.T, s)
        else:
            fNeighs = prev_fod[x+neighs[:, 0], y+neighs[:, 1], z+neighs[:, 2]]
            n_fod = l * np.diag(np.dot(np.dot(B_neg, fNeighs.T), w))
            f = np.dot(-C.T, np.concatenate((s, n_fod)))
        h = matrix(np.zeros(252))
        # Using cvxopt
        args = [matrix(H), matrix(f)]  # Enforce symmetry on H
        args.extend([matrix(-B), h])
        sol = qp(*args)
        if 'optimal' not in sol['status']:
            print('Solution not found')
        fod[x, y, z, :] = np.array(sol['x']).reshape((f.shape[0],))
