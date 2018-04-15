import ctypes
import multiprocessing as mp
import numpy as np
import nibabel as nib
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
        

def csdeconv(response, data_file, mask_file, bvals_file, bvecs_file, max_order):
    # Load data
    bvals = np.genfromtxt(bvals_file, dtype=np.float32)
    bvecs = np.genfromtxt(bvecs_file, dtype=np.float32)
    mask = (nib.load(mask_file)).get_data()
    data_obj = nib.load(data_file)
    data = data_obj.get_data()

    #mask[:, :, :40] = 0
    #mask[:, :, 50:] = 0
    ii = np.where(mask)

    # Get CSD matrices
    C = get_csd_matrix(bvecs[:, bvals > 100], response, max_order)
    H = np.dot(C.T, C)
    B = np.genfromtxt('/Users/matteob/qboot_v2/qboot_v2/utils/ico_5.txt', dtype=np.float32)
    B_sph = qbm.cart2sph(B[:, 0], B[:, 1], B[:, 2])
    B_sh = qbm.get_sh(B_sph[:, 1], B_sph[:, 2], max_order)
    fod = np.zeros(list(mask.shape) + [B_sh.shape[1]], dtype=np.float32)

    # create shared memory arrays
    shared_fod = mp.RawArray(ctypes.c_float, fod.size)
    shared_data = mp.RawArray(ctypes.c_float, data.size)

    fod_ptr = np.ctypeslib.as_array(shared_fod).reshape(fod.shape)
    data_ptr = np.ctypeslib.as_array(shared_data).reshape(data.shape)

    data_ptr[:] = data
    csdeconv_fit.shared_fod = shared_fod
    csdeconv_fit.shared_data = shared_data

    # Chunk up indices
    x, y, z = ii
    nvox = len(x)
    chunk_size = nvox / nprocs
    chunk_end = chunk_size * nprocs

    iixs = [x[i * chunk_size:i * chunk_size + chunk_size] for i in range(nprocs)] + [x[chunk_end:]]
    iiys = [y[i * chunk_size:i * chunk_size + chunk_size] for i in range(nprocs)] + [y[chunk_end:]]
    iizs = [z[i * chunk_size:i * chunk_size + chunk_size] for i in range(nprocs)] + [z[chunk_end:]]

    # create arguments for each child process
    args = [((xs, ys, zs), fod.shape, data.shape, bvals, H, C, B_sh) 
            for xs, ys, zs in zip(iixs, iiys, iizs)]

    # Create child processes
    pool = mp.Pool(processes=nprocs)

    pool.starmap(csdeconv_fit, args)

    nib.Nifti1Image(fod_ptr, None, data_obj.header).to_filename('/Users/matteob/Desktop/fslcourse_update/fsl_course_data/fdt1/subj1/csd_test.nii.gz')


def csdeconv_fit(vox_list, fod_shape, data_shape, bvals, H, C, B):
    fod = csdeconv_fit.shared_fod
    data = csdeconv_fit.shared_data

    fod = np.ctypeslib.as_array(fod).reshape(fod_shape)
    data = np.ctypeslib.as_array(data).reshape(data_shape)

    for x, y, z in zip(*vox_list):
        s = data[x, y, z, bvals >= 100]
        f = np.dot(-C.T, s)
        h = matrix(np.zeros(252))
        # Using cvxopt
        args = [matrix(H), matrix(f)]  # Enforce symmetry on H
        args.extend([matrix(-B), h])
        sol = qp(*args)
        if 'optimal' not in sol['status']:
            print('Solution not found')
        fod[x, y, z, :] = np.array(sol['x']).reshape((f.shape[0],))
