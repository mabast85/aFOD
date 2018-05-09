import ctypes
import multiprocessing as mp
import numpy as np
import scipy as sp
import nibabel as nib
import itertools
from qboot_v2.utils import math as qbm
from qboot_v2.utils import utils
from cvxopt import matrix, spmatrix
from cvxopt.solvers import options, qp

options['show_progress'] = False  # disable cvxopt output
options['maxiters'] = 50    # maximum number of qp iteration
options['abstol'] = 1e-3
options['reltol'] = 1e-3
options['feastol'] = 1e-3

# number of processes to parallelise csdeconv across
nprocs = mp.cpu_count()


# Response function class
class Response(object):
    '''Response function

    We want to have the basic response function sh coefficients, the signal
    and a method to compute it, and a method to i/o
    '''

    def __init__(self, coefficients, max_order):
        '''Creates a new response function on the
        coefficients and max order
        '''
        self.coefficients = coefficients
        self.max_order = max_order
        
    @classmethod
    def get_response(cls, data_file, mask_file, bvals_file, bvecs_file, max_order, bval=None, dti_basename=None, normalize=False):
        '''Computes the response function coefficients and signal
        from a set of voxels in mask
        '''
        if dti_basename is None:
            raise ValueError(dti_basename + ' does not appear to be a valid dtifit basename')
        else:
            dti_V1 = (nib.load(dti_basename + '_V1.nii.gz')).get_data()
        
        bvals = np.genfromtxt(bvals_file, dtype=float)
        bvecs = np.genfromtxt(bvecs_file, dtype=float)
        data = (nib.load(data_file)).get_data()
        mask = (nib.load(mask_file)).get_data()
        vox_list = np.where(mask)
        print('Found ' + str(np.count_nonzero(mask)) + ' masked voxels')

        r_bvals = utils.round_bvals(bvals)
        if bval is None:
            u_bvals, counts = np.unique(r_bvals.astype(int), return_counts=True)
            print('Found ' + str(u_bvals.size) + ' shells')
        else:
            u_bvals = np.array(bval)

        count_b = 0
        n_coeffs = (max_order+1)*(max_order+2)/2
        coefficients = np.zeros((u_bvals.size, int(n_coeffs)))
        for i in np.arange(0, u_bvals.size):
            if u_bvals.size == 1:
                b = u_bvals
            else:
                b = u_bvals[i]
            if b <= 100:
                rot_bvecs_sph = qbm.cart2sph(bvecs[0, r_bvals > 100], bvecs[1, r_bvals > 100], bvecs[2, r_bvals > 100])
                rot_bvecs_sh = qbm.get_sh(rot_bvecs_sph[:, 1], rot_bvecs_sph[:, 2], max_order)
                s0 = np.mean(data[:, :, :, bvals < 100], axis=3)
                s = np.ones(rot_bvecs_sph.shape[0]) * np.mean(s0[mask > 0])
                if normalize:
                    coefficients[count_b, :] = coefficients[count_b, :] + np.linalg.lstsq(rot_bvecs_sh, s / np.mean(s0[mask > 0]))[0]
                else:
                    coefficients[count_b, :] = coefficients[count_b, :] + np.linalg.lstsq(rot_bvecs_sh, s)[0]
            else:
                count = 0
                for x, y, z in zip(*vox_list):
                    count += 1
                    R = qbm.get_rotation(dti_V1[x, y, z, :], [0, 0, 1])
                    rot_bvecs = np.dot(R.T, bvecs[:, r_bvals == b])
                    rot_bvecs_sph = qbm.cart2sph(rot_bvecs[0, :], rot_bvecs[1, :], rot_bvecs[2, :])
                    rot_bvecs_sh = qbm.get_sh(rot_bvecs_sph[:, 1], rot_bvecs_sph[:, 2], max_order)
                    s = data[x, y, z, r_bvals == b]
                    if normalize:
                        s0 = np.mean(data[x, y, z, bvals < 100])
                        coefficients[count_b, :] = coefficients[count_b, :] + np.linalg.lstsq(rot_bvecs_sh, s / s0)[0]
                    else:
                        coefficients[count_b, :] = coefficients[count_b, :] + np.linalg.lstsq(rot_bvecs_sh, s)[0]
                coefficients[count_b, :] /= count
            count_b += 1
                
        return cls(coefficients, max_order)
        
    # Rotational harmonics
    def get_rh(self):
        delta = qbm.get_delta(np.array([0]), np.array([0]), self.max_order)
        return self.coefficients[:, delta[0, :] != 0] / delta[delta != 0]

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


def get_csd_matrix(bvecs, bvals, response, max_order, sym=True):
    r_bvals = utils.round_bvals(bvals)
    u_bvals, counts = np.unique(r_bvals.astype(int), return_counts=True)
    if u_bvals.size != response.coefficients.shape[0]:
        raise ValueError('Number of shells does not appear to match the number of response functions')
    bvecs_sph = qbm.cart2sph(bvecs[0, :], bvecs[1, :], bvecs[2, :])
    bvecs_sh = qbm.get_sh(bvecs_sph[:, 1], bvecs_sph[:, 2], max_order)
    rh = response.get_rh()
    if response.max_order < max_order:
        rh = np.append(rh, np.ones((rh.shape[0], int((max_order - response.max_order)/2))) * 1E-16, axis=1)
    
    C = np.zeros(bvecs_sh.shape)
    for i in np.arange(0, u_bvals.size):
        if u_bvals.size == 1:
            b = u_bvals
        else:
            b = u_bvals[i]
        m, R = np.concatenate([[(m, rh[i, int(l/2)]) for m in range(-l, l+1)] for l in range(0, max_order+1, 2)], axis=0).T
        R = np.diag(R)
        C[r_bvals == b, :] = np.dot(bvecs_sh[r_bvals == b, :], R)
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
    C = get_csd_matrix(bvecs, bvals, response, max_order, sym)
    
    # Initialise output fod matrix
    fod = np.zeros(list(mask.shape) + [C.shape[1]], dtype=np.float32)

    ii = np.where(mask)
    xs, ys, zs = ii
    for x, y, z in zip(xs, ys, zs):
        s = data[x, y, z, :]
        fod[x, y, z, :] = np.linalg.lstsq(C, s, rcond=-1)[0]

    if out_file is not None:
        nib.Nifti1Image(fod, None, data_obj.header).to_filename(out_file)

    return fod
    

def csdeconv(response, data_file, mask_file, bvals_file, bvecs_file, max_order, sym=False, l=0.1, sigma=40, out_file=None):
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

    B = np.genfromtxt('/Users/matteob/qboot_v2/qboot_v2/utils/ico_5.txt', dtype=np.float32)
    B_sph = qbm.cart2sph(B[:, 0], B[:, 1], B[:, 2])
    # Get CSD matrices
    if isinstance(response, list):  # Multi-tissue
        C = get_csd_matrix(bvecs, bvals, response[0], max_order[0], sym)
        for i in np.arange(1, len(response)):
            C_tmp = get_csd_matrix(bvecs, bvals, response[i], max_order[i], sym)
            C = np.concatenate((C, C_tmp), axis=1)
    else:   # Single-tissue
        C = get_csd_matrix(bvecs, bvals, response, max_order, sym)
        
    if sym is False:
        B_sh = qbm.get_sh(B_sph[:, 1], B_sph[:, 2], max_order, c='all')
        B_neg_sph = qbm.cart2sph(-B[:, 0], -B[:, 1], -B[:, 2])
        B_neg_sh = qbm.get_sh(B_neg_sph[:, 1], B_neg_sph[:, 2], max_order, c='all')
        l = l * C.shape[0] * (response.get_rh())[0, 0] / B.shape[0]
        C = np.concatenate((C, l*B_sh), axis=0)
        w = get_weights(B, sigma)
        print('Running SD')
        prev_fod = sdeconv(response, data_file, mask_file, bvals_file, bvecs_file, max_order, sym=sym)
    else:
        if isinstance(response, list):  # Multi-tissue
            B_sh_list = [qbm.get_sh(B_sph[:, 1], B_sph[:, 2], max_order[i]) 
                         for i in np.arange(0, len(response))]
            B_sh = sp.linalg.block_diag(*B_sh_list)
        else:  # Single-tissue
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
    chunk_size = int(nvox / nprocs)
    chunk_end = int(chunk_size * nprocs)

    iixs = [x[i * chunk_size:i * chunk_size + chunk_size] for i in range(nprocs)] + [x[chunk_end:]]
    iiys = [y[i * chunk_size:i * chunk_size + chunk_size] for i in range(nprocs)] + [y[chunk_end:]]
    iizs = [z[i * chunk_size:i * chunk_size + chunk_size] for i in range(nprocs)] + [z[chunk_end:]]
    
    # create arguments for each child process
    if sym:
        args = [((xs, ys, zs), fod.shape, data.shape, bvals, H, C, B_sh, sym) 
                for xs, ys, zs in zip(iixs, iiys, iizs)]
    else:
        args = [((xs, ys, zs), fod.shape, data.shape, bvals, H, C, B_sh, sym, B_neg_sh.copy(), w.copy(), l.copy()) 
                for xs, ys, zs in zip(iixs, iiys, iizs)]
    
    # csdeconv_fit((x, y, z), fod.shape, data.shape, bvals, H, C, B_sh, sym, B_neg_sh, w, l)

    # Create child processes
    print('Starting multiple processes, N=', nprocs)
    pool = mp.Pool(processes=nprocs)
    
    print('Running CSD')
    pool.starmap(csdeconv_fit, args)

    if out_file is not None:
        print('Storing SH coefficients')
        nib.Nifti1Image(fod_ptr, None, data_obj.header).to_filename(out_file)

    return fod_ptr
    
    
def csdeconv_fit(vox_list, fod_shape, data_shape, bvals, H, C, B, sym, B_neg=None, w=None, l=None):
    fod = csdeconv_fit.shared_fod
    data = csdeconv_fit.shared_data
    
    fod = np.ctypeslib.as_array(fod).reshape(fod_shape)
    data = np.ctypeslib.as_array(data).reshape(data_shape)
    if sym is False:
        prev_fod = csdeconv_fit.shared_prev_fod
        prev_fod = np.ctypeslib.as_array(prev_fod).reshape(fod_shape)
        neighs = np.array(list(itertools.product([-1, 0, 1], repeat=3)))
        neighs = np.delete(neighs, 13, 0)   # Remove [0, 0, 0]

    h = matrix(np.zeros(B.shape[0]))
    args = [matrix(H), 0, matrix(-B), h]
    for x, y, z in zip(*vox_list):
        s = data[x, y, z, :]
        # if sym:
        #    f = np.dot(-C.T, s)
        if not sym:
            fNeighs = prev_fod[x+neighs[:, 0], y+neighs[:, 1], z+neighs[:, 2]]
            n_fod = l * np.diag(np.dot(np.dot(B_neg, fNeighs.T), w))
            s = np.concatenate((s, n_fod))
            # f = np.dot(-C.T, np.concatenate((s, n_fod)))
        f = np.dot(-C.T, s)
        # Using cvxopt
        # args = [matrix(H), matrix(f)]  # Enforce symmetry on H
        # args.extend([matrix(-B), h])
        args[1] = matrix(f)
        sol = qp(*args)
        if 'optimal' not in sol['status']:
            print('Solution not found')
        fod[x, y, z, :] = np.array(sol['x']).reshape((f.shape[0],))


def predict(response, fod_file, mask_file, bvals_file, bvecs_file, max_order, sym=False, out_file=None):
    # Load data
    bvals = np.genfromtxt(bvals_file, dtype=np.float32)
    bvecs = np.genfromtxt(bvecs_file, dtype=np.float32)
    mask = (nib.load(mask_file)).get_data()
    fod_obj = nib.load(fod_file)
    fod = fod_obj.get_data()

    # Get CSD matrices
    if isinstance(response, list):  # Multi-tissue
        C = get_csd_matrix(bvecs, bvals, response[0], max_order[0], sym)
        for i in np.arange(1, len(response)):
            C_tmp = get_csd_matrix(bvecs, bvals, response[i], max_order[i], sym)
            C = np.concatenate((C, C_tmp), axis=1)
    else:   # Single-tissue
        C = get_csd_matrix(bvecs, bvals, response, max_order, sym)
        
    # Initialise output fod matrix
    pred = np.zeros(list(mask.shape) + [bvecs.shape[1]], dtype=np.float32)

    ii = np.where(mask)
    xs, ys, zs = ii
    for x, y, z in zip(xs, ys, zs):
        f = fod[x, y, z, :]
        pred[x, y, z, :] = np.dot(C, f)

    if out_file is not None:
        print('Storing SH coefficients')
        nib.Nifti1Image(pred, None, fod_obj.header).to_filename(out_file)
    
    return pred
    
    