import numpy as np
import nibabel as nib
from qboot_v2.utils import math as qbm
from cvxpy import Constant, Minimize, Problem, Variable, quad_form
# from quadprog import solve_qp
# from cvxopt import matrix, spmatrix
# from cvxopt.solvers import options, qp

# options['show_progress'] = False  # disable cvxopt output


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


def get_csd_matrix(bvecs, response, max_order):
    bvecs_sph = qbm.cart2sph(bvecs[0, :], bvecs[1, :], bvecs[2, :])
    bvecs_sh = qbm.get_sh(bvecs_sph[:, 1], bvecs_sph[:, 2], max_order)
    rh = response.get_rh()
    if response.max_order < max_order:
        rh = np.append(rh, np.ones((max_order - response.max_order)/2) * 1E-16)
    m, R = np.concatenate([[(m, rh[int(l/2)]) for m in range(-l, l+1)] for l in range(0, max_order+1, 2)], axis=0).T
    R = np.diag(R)
    return np.dot(bvecs_sh, R)


def csdeconv(response, data_file, mask_file, bvals_file, bvecs_file, max_order):
    bvals = np.genfromtxt(bvals_file, dtype=float)
    bvecs = np.genfromtxt(bvecs_file, dtype=float)
    mask = (nib.load(mask_file)).get_data()

    ii = np.where(mask)
    data = (nib.load(data_file)).get_data()
    C = get_csd_matrix(bvecs[:, bvals > 100], response, max_order)
    B = np.genfromtxt('/Users/matteob/qboot_v2/qboot_v2/utils/ico_5.txt', dtype=float)
    B_sph = qbm.cart2sph(B[:, 0], B[:, 1], B[:, 2])
    B_sh = qbm.get_sh(B_sph[:, 1], B_sph[:, 2], max_order)
    H = np.dot(C.T, C)
    fod = np.zeros((116, 116, 76, B_sh.shape[1]), dtype=float)
    n = B_sh.shape[1]
    _fod = Variable(n)
    H_const = Constant(H)  # see http://www.cvxpy.org/en/latest/faq/
    constraints = []
    constraints.append(-B_sh * _fod <= 0)
    n_vox = np.count_nonzero(mask)
    v_count = 0
    print('USING CVXPY')
    for x, y, z in zip(*ii):
        v_count = v_count + 1
        if np.mod(v_count, 1000) == 0:
            print(str(np.round(100 * v_count / n_vox)) + ' fitted voxels')
            
        s = data[x, y, z, bvals >= 100]
        f = np.dot(-C.T, s)

        # Using CVXPY
        
        objective = Minimize(0.5 * quad_form(_fod, H_const) + f * _fod)
        prob = Problem(objective, constraints)
        result = prob.solve(solver='SCS')
        fod[x, y, z, :] = (np.array(_fod.value)).reshape((n,))

        # Using quadprog
        # print(H.shape)
        # np.savetxt('/Users/matteob/Desktop/fslcourse_update/fsl_course_data/fdt1/subj1/H.txt', H, fmt='%.5f', delimiter=' ')
        # fod[x, y, z, :] = solve_qp(0.5 * (H + H.T), f, B_sh.T, np.zeros(252), 0)[0]

        # Using cvxopt
        # args = [matrix(0.5 * (H + H.T)), matrix(f)]
        # args.extend([matrix(-B_sh), matrix(np.zeros(252))])
        # sol = qp(*args)
        # if 'optimal' not in sol['status']:
        #     print('ARGH!')
        # fod[x, y, z, :] = np.array(sol['x']).reshape((f.shape[0],))

        # fod[x, y, z, :] = qc_solver(H, f, B_sh, np.zeros(B_sh.shape[0], dtype=float))

    nib.Nifti1Image(fod, np.eye(4)).to_filename('/Users/matteob/Desktop/fslcourse_update/fsl_course_data/fdt1/subj1/csd_test.nii.gz')
