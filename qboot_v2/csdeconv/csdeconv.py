#!/usr/bin/env python

import os.path as op
import ctypes
import multiprocessing as mp
import numpy as np
import scipy as sp
import nibabel as nib
import itertools
import threading
import progressbar
from qboot_v2.utils import math as qbm
from qboot_v2.utils import utils
from cvxopt import matrix
from cvxopt.solvers import options, qp

# import osqp
# import scipy.sparse as sparse


options['show_progress'] = False  # disable cvxopt output
options['maxiters'] = 50    # maximum number of qp iteration
options['abstol'] = 1e-3
options['reltol'] = 1e-3
options['feastol'] = 1e-3

# number of processes to parallelise csdeconv across
nprocs = mp.cpu_count()


# Response function class
class Response(object):
    '''Response function class

    Contains and computes basic response function sh coefficients; provides
    methods for i/o.

    Attributes:
        coefficients: number of shells x coefficients numpy array.
        max_order: maximum SH order (must be even).

    '''

    def __init__(self, coefficients, max_order):
        '''Inits a new response function.'''
        self.coefficients = coefficients
        self.max_order = max_order

    @classmethod
    def get_response(cls, data_file, mask_file, bvals_file, bvecs_file,
                     max_order, dti_basename, bval=None, normalize=False):
        '''Computes the response function coefficients.

        This method computes the response function's coefficients up to
        max_order from a set of masked voxels; it uses the DT estimated by FSL.

        Args:
            data_file: string containing the path to the 4D nifti dMRI data file
            mask_file: string containing the path to the 3D nifti binary mask file
            bvals_file: string containing the path to the bvals file
            bvecs_file: string containing the path to the bvecs file
            max_order: integer specifying the maximum harmonic order (must be even)
            bval: list of integer specifying which bvals to use (optional)
            dti_basename: FSL's dtifit output basename
            normalize: if true, normalize the dw signal by the b0

        Returns:
            A response function class with the estimated coefficients
        '''
        # Read input files
        bvals = np.genfromtxt(bvals_file, dtype=float)
        bvecs = np.genfromtxt(bvecs_file, dtype=float)
        data = (nib.load(data_file)).get_data()
        mask = (nib.load(mask_file)).get_data()
        dti_V1 = (nib.load(dti_basename + '_V1.nii.gz')).get_data()
        vox_list = np.where(mask)
        print('Found ' + str(np.count_nonzero(mask)) + ' masked voxels')

        # Round the bvals
        r_bvals = utils.round_bvals(bvals)
        # If bval is not specified, get coefficients for all each unique shell
        if bval is None:
            u_bvals = np.unique(r_bvals.astype(int))
            print('Found ' + str(u_bvals.size) + ' shells')
        else:
            u_bvals = np.atleast_1d(bval)

        # Initialize outuput matrix
        n_coeffs = (max_order+1)*(max_order+2)/2
        coefficients = np.zeros((u_bvals.size, int(n_coeffs)))
        # Main loop through the requested shells
        for count_b, b in enumerate(u_bvals):
            # b0 coefficients
            if b <= 100:
                rot_bvecs_sph = qbm.cart2sph(bvecs[0, r_bvals > 100], bvecs[1, r_bvals > 100], bvecs[2, r_bvals > 100])
                rot_bvecs_sh = qbm.get_sh(rot_bvecs_sph[:, 1], rot_bvecs_sph[:, 2], max_order)
                s0 = np.mean(data[:, :, :, bvals < 100], axis=3)
                s = np.ones(rot_bvecs_sph.shape[0]) * np.mean(s0[mask > 0])
                if normalize:
                    coefficients[count_b, :] = coefficients[count_b, :] + np.linalg.lstsq(rot_bvecs_sh, s / np.mean(s0[mask > 0]))[0]
                else:
                    coefficients[count_b, :] = coefficients[count_b, :] + np.linalg.lstsq(rot_bvecs_sh, s)[0]
            # b>0 coefficients
            else:
                # Main loop through the masked voxels
                for x, y, z in zip(*vox_list):
                    # Rotation matrix to align V1 with the z axis
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
                coefficients[count_b, :] /= len(vox_list[0])

        return cls(coefficients, max_order)

    def get_rh(self):
        '''Gets rotational harmonics.'''
        delta = qbm.get_delta(np.array([0]), np.array([0]), self.max_order)
        return self.coefficients[:, delta[0, :] != 0] / delta[delta != 0]

    # I/O
    @classmethod
    def read_coefficients(cls, fname):
        '''Reads SH coefficients from a text file

        Args:
            cls: response function class.
            fname: string with the response function's coefficients path.

        Returns:
            A response function class with the imported coefficients.
        '''
        h = np.genfromtxt(fname, max_rows=1, dtype=int, delimiter=' ')
        _coefficients = np.genfromtxt(fname, skip_header=1, dtype=float, delimiter=' ')
        max_order = h[1]
        n_coeffs = int(1 + max_order / 2)
        print('Importing response function coefficients...')
        print(str(h[0]) + ' b-shells detected, max harmonic order=' + str(h[1]))
        _coefficients = _coefficients.reshape((h[0], n_coeffs))
        delta = qbm.get_delta(np.array([0]), np.array([0]), max_order)
        coefficients = np.zeros((h[0], (delta.size)), dtype=float)
        coefficients[:, delta[0, :] != 0] = _coefficients
        return cls(coefficients=coefficients, max_order=max_order)

    def write_coefficients(self, fname):
        '''Writes SH coefficients to a text file.

        Args:
            fname: string with the response function's coefficients path.
        '''
        delta = qbm.get_delta(np.array([0]), np.array([0]), self.max_order)
        with open(fname, 'wb') as fh:
            h = np.array([[self.coefficients.shape[0], self.max_order]])
            np.savetxt(fh, h, fmt='%d', delimiter=' ')
            np.savetxt(fh, self.coefficients[:, delta[0, :] != 0], fmt='%.5f', delimiter=' ')


def get_csd_matrix(bvecs, bvals, response, max_order, sym=True):
    '''Computes convolution matrix.

    Generates convolution matrix for each acquired orientation;
    if multi-tissue, concatenates convolution matrices for
    the different tissues.

    Args:
        bvecs: 3xN numpy array with diffusion encoding orientations.
        bvals: N numpy array with b-values.
        response: single response function object.
        max_order: single maximum harmonic order.
        sym: if true, consider only even order symmetrics SH coefficients.

    Returns:
        Convolution matrix as numpy array.
    '''
    # Round bvalues and find unique shells
    r_bvals = utils.round_bvals(bvals)
    u_bvals = np.unique(r_bvals.astype(int))
    if u_bvals.size != response.coefficients.shape[0]:
        raise ValueError('Number of shells does not appear to match the number of response functions')
    bvecs_sph = qbm.cart2sph(bvecs[0, :], bvecs[1, :], bvecs[2, :])
    bvecs_sh = qbm.get_sh(bvecs_sph[:, 1], bvecs_sph[:, 2], max_order)
    rh = response.get_rh()
    if response.max_order < max_order:
        rh = np.append(rh, np.zeros((rh.shape[0], int((max_order - response.max_order)/2))), axis=1)

    C = np.zeros(bvecs_sh.shape)
    for b, rh_shell in zip(u_bvals, rh):
        m, R = np.concatenate([[(m, rh_shell[int(l/2)]) for m in range(-l, l+1)]
                               for l in range(0, max_order+1, 2)], axis=0).T
        R = np.diag(R)
        C[r_bvals == b, :] = np.dot(bvecs_sh[r_bvals == b, :], R)
    if sym:
        return C
    else:
        m, l = np.concatenate([[(m, l) for m in range(-l, l+1)] for l in range(0, max_order+1)], axis=0).T
        a = (np.diag((np.mod(l, 2) == 0))[np.mod(l, 2) == 0, :]).astype(int)
        return np.dot(C, a)  # Zero odd components


def get_weights(vertices, sigma=40):
    '''Computes neighbouring fod weights for asymmetric CSD.

    Generates matrix that contains the weight for each point on the
    neighbouring fod based on their distance to the current voxel and
    the angle between the current fod point and the point of the
    neighbouring fod.

    Args:
        vertices: Nx3 numpy array with vertices of the unit sphere.
        sigma: cut-off angle.

    Returns:
        26xN weight matrix as numpy array.
    '''
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


def sdeconv(response, data_file, mask_file, bvals_file, bvecs_file, max_order,
            sym=False, out_file=None):
    '''Unconstrained spherical deconvolution.

    Estimates voxel-wise FOD using unconstrained spherical deconvolution.

    Args:
        response: list (for multi-tissue) or single response function object.
        data_file: string containing the path to the 4D nifti dMRI data file
        mask_file: string containing the path to the 3D nifti binary mask file
        bvals_file: string containing the path to the bvals file
        bvecs_file: string containing the path to the bvecs file
        max_order: list (for multi-tissue) or single maximum harmonic order.
        sym: if true, consider only even order symmetrics SH coefficients.
        out_file: string containing the output file name (optional).

    Returns:
        4D numpy array of SH coefficients.
    '''
    # Load data
    bvals = np.genfromtxt(bvals_file, dtype=np.float32)
    bvecs = np.genfromtxt(bvecs_file, dtype=np.float32)
    mask = (nib.load(mask_file)).get_data()
    data_obj = nib.load(data_file)
    data = data_obj.get_data()

    # Get convolution matrix
    if isinstance(response, list):  # Multi-tissue
        # Get CSD matrices
        C = get_csd_matrix(bvecs, bvals, response[0], max_order[0], sym)
        for i in np.arange(1, len(response)):
            C_tmp = get_csd_matrix(bvecs, bvals, response[i], max_order[i], sym)
            C = np.concatenate((C, C_tmp), axis=1)
    else:   # Single-tissue
        # Get CSD matrix
        C = get_csd_matrix(bvecs, bvals, response, max_order, sym)

    # Initialise output fod matrix
    fod = np.zeros(list(mask.shape) + [C.shape[1]], dtype=np.float32)

    ii = np.where(mask)
    xs, ys, zs = ii
    for x, y, z in zip(xs, ys, zs):
        s = data[x, y, z, :]
        fod[x, y, z, :] = np.linalg.lstsq(C, s, rcond=-1)[0]

    if out_file is not None:
        print('Storing FOD SH coefficients')
        nib.Nifti1Image(fod, None, data_obj.header).to_filename(out_file)

    return fod


def csdeconv(response, data_file, mask_file, bvals_file, bvecs_file, max_order,
             sym=False, l=0.1, sigma=40, out_file=None):
    '''Constrained spherical deconvolution.

    Estimates symmetric or asymmetric voxel-wise FOD using constrained
    spherical deconvolution. Naming of matrices follows the one specified in:
    Bastiani, M., Cottaar, M., Dikranian, K., Ghosh, A., Zhang, H., Alexander,
    D.C., Behrens, T.E., Jbabdi, S., Sotiropoulos, S.N., 2017. Improved
    tractography using asymmetric fibre orientation distributions. Neuroimage
    158, 205-218.

    Args:
        response: list (for multi-tissue) or single response function object.
        data_file: string containing the path to the 4D nifti dMRI data file
        mask_file: string containing the path to the 3D nifti binary mask file
        bvals_file: string containing the path to the bvals file
        bvecs_file: string containing the path to the bvecs file
        max_order: list (for multi-tissue) or single maximum harmonic order.
        sym: if true, consider only even order symmetrics SH coefficients.
        l: lambda regularization factor for asymmetric CSD.
        sigma: cut-off neighbourhood angle for asymmetric CSD.
        out_file: string containing the output file name (optional).

    Returns:
        4D numpy array of SH coefficients.
    '''
    # Load data
    bvals = np.genfromtxt(bvals_file, dtype=np.float32)
    bvecs = np.genfromtxt(bvecs_file, dtype=np.float32)
    mask = (nib.load(mask_file)).get_data()
    data_obj = nib.load(data_file)
    data = data_obj.get_data()

    # Get list of masked voxels
    mask[:, :, 0] = 0
    ii = np.where(mask)

    # If symmetric CSD, get only even SH coefficients
    if sym:
        sh_coeff = 'even'
    else:
        sh_coeff = 'all'

    # ========================
    # Get necessary matrices
    # ========================
    resource_dir = op.dirname(__file__)
    ico5 = op.join(resource_dir, 'ico_5.txt')
    # B = np.genfromtxt('/Users/matteob/qboot_v2/qboot_v2/utils/ico_5.txt', dtype=np.float32)[:, 0:3]
    B = np.genfromtxt(ico5, dtype=np.float32)[:, 0:3]
    B_sph = qbm.cart2sph(B[:, 0], B[:, 1], B[:, 2])

    if isinstance(response, list):  # Multi-tissue
        # Get CSD matrices
        C = get_csd_matrix(bvecs, bvals, response[0], max_order[0], sym)
        for i in np.arange(1, len(response)):
            C_tmp = get_csd_matrix(bvecs, bvals, response[i], max_order[i], sym)
            C = np.concatenate((C, C_tmp), axis=1)
        # Get B matrix
            B_sh_list = [qbm.get_sh(B_sph[:, 1], B_sph[:, 2], max_order[i], coeffs=sh_coeff)
                         for i in np.arange(0, len(response))]
            B_sh = sp.linalg.block_diag(*B_sh_list)
    else:   # Single-tissue
        # Get CSD matrix
        C = get_csd_matrix(bvecs, bvals, response, max_order, sym)
        # Get B matrix
        B_sh = qbm.get_sh(B_sph[:, 1], B_sph[:, 2], max_order, coeffs=sh_coeff)

    if sym is False:
        B_neg_sph = qbm.cart2sph(-B[:, 0], -B[:, 1], -B[:, 2])
        l = l * C.shape[0] / B.shape[0]
        w = get_weights(B, sigma)
        print('Running SD')
        prev_fod = sdeconv(response, data_file, mask_file, bvals_file, bvecs_file, max_order, sym=sym)
        if isinstance(response, list):  # Multi-tissue
            B_neg_sh = qbm.get_sh(B_neg_sph[:, 1], B_neg_sph[:, 2], max_order[0], coeffs=sh_coeff)
            # b0 = [0*i for i in np.arange(1, len(response))]
            # B_neg_sh = sp.linalg.block_diag(B_neg_sh, *b0)
            B_neg_sh = np.concatenate((B_neg_sh, np.zeros((B_neg_sh.shape[0], len(response)-1))), axis=1)
            l = l * (response[0].get_rh())[0, 0]
            # B_C_sh = sp.linalg.block_diag(B_sh_list[0], *b0)
            B_C_sh = np.concatenate((B_sh_list[0], np.zeros((B_sh_list[0].shape[0], len(response)-1))), axis=1)
            C = np.concatenate((C, l*B_C_sh), axis=0)
            # w = np.concatenate((w, np.zeros((w.shape[0], len(response)-1))), axis=1)
        else:
            B_neg_sh = qbm.get_sh(B_neg_sph[:, 1], B_neg_sph[:, 2], max_order, coeffs=sh_coeff)
            l = l * (response.get_rh())[0, 0]
            C = np.concatenate((C, l*B_sh), axis=0)

    H = np.dot(C.T, C)
    H = H + 1e-3*np.eye(H.shape[0])
    fod = np.zeros(list(mask.shape) + [B_sh.shape[1]], dtype=np.float32)

    # Create shared memory arrays
    shared_fod = mp.RawArray(ctypes.c_float, fod.size)
    shared_data = mp.RawArray(ctypes.c_float, data.size)
    shared_prev_fod = None

    fod_ptr = np.ctypeslib.as_array(shared_fod).reshape(fod.shape)
    data_ptr = np.ctypeslib.as_array(shared_data).reshape(data.shape)

    data_ptr[:] = data

    if sym is False:
        shared_prev_fod = mp.RawArray(ctypes.c_float, prev_fod.size)
        prev_fod_ptr = np.ctypeslib.as_array(shared_prev_fod).reshape(prev_fod.shape)
        prev_fod_ptr[:] = prev_fod

    # Chunk up indices
    x, y, z = ii
    nvox = len(x)
    chunk_size = int(nvox / nprocs)
    chunk_end = int(chunk_size * nprocs)

    iixs = [x[i * chunk_size:i * chunk_size + chunk_size] for i in range(nprocs)] + [x[chunk_end:]]
    iiys = [y[i * chunk_size:i * chunk_size + chunk_size] for i in range(nprocs)] + [y[chunk_end:]]
    iizs = [z[i * chunk_size:i * chunk_size + chunk_size] for i in range(nprocs)] + [z[chunk_end:]]

    # create a multiprocessing context
    ctx = mp.get_context('forkserver')
    progqueue = ctx.Queue()

    # create arguments for each child process
    if sym:
        args = [(shared_data, shared_fod, shared_prev_fod, (xs, ys, zs), fod.shape, data.shape, bvals, H, C, B_sh, sym, progqueue)
                for xs, ys, zs in zip(iixs, iiys, iizs)]
    else:
        args = [(shared_data, shared_fod, shared_prev_fod, (xs, ys, zs), fod.shape, data.shape, bvals, H, C, B_sh, sym, progqueue, B_neg_sh, w, l)
                for xs, ys, zs in zip(iixs, iiys, iizs)]

    # csdeconv_fit((x, y, z), fod.shape, data.shape, bvals, H, C, B_sh, sym, B_neg_sh, w, l)

    print('Running CSD (using {} processes)'.format(nprocs), flush=True)

    # Create a progress bar to show progress,
    # and a thread which receives updates
    # from the csdeconv_fit processes, and
    # updates the progress bar accordingly.
    progbar = progressbar.ProgressBar(max_value=nvox)
    progbar.start()

    def update_progress():
        while True:
            nextval = progqueue.get()
            if nextval == 'finish':
                return
            else:
                progbar.update(progbar.value + nextval)

    progthread = threading.Thread(target=update_progress)
    progthread.daemon = True
    progthread.start()

    # Create the child processes
    procs = []
    for a in args:
        p = ctx.Process(target=csdeconv_fit, args=a)
        p.start()
        procs.append(p)

    # Wait until they're finished
    for p in procs:
        p.join()

    # Send a signal to the progress bar
    # thread to tell it to finish up.
    progbar.finish()
    progqueue.put('finish')

    if out_file is not None:
        print('Storing SH coefficients')
        nib.Nifti1Image(fod_ptr, None, data_obj.header).to_filename(out_file)

    return fod_ptr


def csdeconv_fit(data, fod, prev_fod, vox_list, fod_shape, data_shape, bvals, H, C, B, sym, progqueue=None, B_neg=None, w=None, l=None):
    '''Constrained spherical deconvolution fitiing method.

    Computes FOD coefficients using quadratic programming (QP) solver and
    stores them in the shared memory numpy array.

    Args:
        data:  Diffusion data
        fod: Array to store output
        prev_fod:  Unconstrained spherical deconvolution (only used if sym is False)
        vox_list: list of masked voxels.
        fod_shape: list of FOD array dimensions.
        data_shape: list of data array dimensions.
        bvals: N numpy array of b-values.
        H: QP matrix.
        C: convolution matrix.
        B: unit sphere SH coefficients.
        sym: if true, consider only even order symmetrics SH coefficients.
        progqueue: mp.Queue to post progress updates
        B_neg: flipped unit sphere SH coefficients for asymmetric FOD fit.
        w: weights matrix for asymmetric FOD fit.
        l: lambda for asymmetric FOD fit.
    '''

    fod = np.ctypeslib.as_array(fod).reshape(fod_shape)
    data = np.ctypeslib.as_array(data).reshape(data_shape)
    if sym is False:
        prev_fod = np.ctypeslib.as_array(prev_fod).reshape(fod_shape)
        neighs = np.array(list(itertools.product([-1, 0, 1], repeat=3)))
        neighs = np.delete(neighs, 13, 0)   # Remove [0, 0, 0]

    # np.savetxt('/Users/matteob/Desktop/fslcourse_update/fsl_course_data/fdt1/subj1/H.txt',H)
    # np.savetxt('/Users/matteob/Desktop/fslcourse_update/fsl_course_data/fdt1/subj1/C.txt',C)
    h = matrix(np.zeros(B.shape[0]))
    args = [matrix(H), 0, matrix(-B), h]
    '''
    P = sparse.csc_matrix(H)
    q = np.zeros(H.shape[0])
    A = sparse.csc_matrix(-B)
    l = -np.inf*np.ones(len(np.zeros(B.shape[0])))
    u = np.zeros(B.shape[0])

    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, alpha=1.0)
    '''
    for i, (x, y, z) in enumerate(zip(*vox_list)):
        s = data[x, y, z, :]
        # if sym:
        #    f = np.dot(-C.T, s)
        if not sym:
            fNeighs = prev_fod[x+neighs[:, 0], y+neighs[:, 1], z+neighs[:, 2]]
            n_fod = l * np.diag(np.dot(np.dot(B_neg, fNeighs.T), w))
            s = np.concatenate((s, n_fod))
            # np.savetxt('/Users/matteob/Desktop/fslcourse_update/fsl_course_data/fdt1/subj1/fne.txt',fNeighs)
            # f = np.dot(-C.T, np.concatenate((s, n_fod)))
        f = np.dot(-C.T, s)
        # Using cvxopt
        # args = [matrix(H), matrix(f)]  # Enforce symmetry on H
        # args.extend([matrix(-B), h])
        args[1] = matrix(f)
        # np.savetxt('/Users/matteob/Desktop/fslcourse_update/fsl_course_data/fdt1/subj1/f.txt',f)
        # np.savetxt('/Users/matteob/Desktop/fslcourse_update/fsl_course_data/fdt1/subj1/B.txt',B)
        # np.savetxt('/Users/matteob/Desktop/fslcourse_update/fsl_course_data/fdt1/subj1/h_low.txt',h)

        sol = qp(*args)
        if 'optimal' not in sol['status']:
            print('Solution not found')
        fod[x, y, z, :] = np.array(sol['x']).reshape((f.shape[0],))

        if progqueue is not None and i > 0 and i % 100 == 0:
            progqueue.put(100)

        '''
        prob.update(Px=f)
        res = prob.solve()
        '''

def predict(response, fod_file, mask_file, bvals_file, bvecs_file, max_order, sym=False, out_file=None):
    '''Predicted signal from CSD fit.

    Computes predicted signal given a 4D array of SH coefficients.

    Args:
        response: list (for multi-tissue) or single response function object.
        fod_file: string containing the path to the 4D nifti FOD SH coefficients file.
        mask_file: string containing the path to the 3D nifti binary mask file
        bvals_file: string containing the path to the bvals file
        bvecs_file: string containing the path to the bvecs file
        max_order: list (for multi-tissue) or single maximum harmonic order.
        sym: if true, consider only even order symmetrics SH coefficients.
        out_file: string containing the output file name (optional).

    Returns:
        4D numpy array of predicted signal.
    '''
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
