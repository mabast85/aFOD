import numpy as np
import nibabel as nib
from qboot_v2.utils import math as qbm


class Response(object):
    '''Response function

    We want to have the basic response function sh coefficients, the signal
    and a method to compute it, and a method to i/o
    '''

    def __init__(self, coefficients, signal, max_order):
        '''Creates a new response function on the
        coefficients and max order'''
        self.coefficients = coefficients
        self.signal = signal
        self.max_order = max_order

    def get_response(self, data_file, mask_file, bvals_file, bvecs_file, dti_basename=None, normalize=False):
        '''Computes the response function coefficients and signal
        from a set of voxels in mask'''

        resp = 0

        if dti_basename is None:
            raise ValueError(dti_basename + ' does not appear to be a valid dtifit basename')
        else:
            dti_v1 = (nib.load(dti_basename + '_V1.nii.gz')).get_data()
        
        bvals = np.genfromtxt(bvals_file, dtype=float)
        bvecs = np.genfromtxt(bvecs_file, dtype=float)
        bvecs_sph = qbm.cart2sph(bvecs[0, bvals >= 100], bvecs[1, bvals >= 100], bvecs[2, bvals >= 100])
        bvecs_sh = qbm.get_sh(bvecs_sph[:, 1], bvecs_sph[:, 2], self.max_order)
        mask = (nib.load(mask_file)).get_data()
        ii = np.where(mask)
        print('Found ' + str(np.count_nonzero(mask)) + ' masked voxels')
        
        data = (nib.load(data_file)).get_data()
        for x, y, z in zip(*ii):
            R = qbm.get_rotation(dti_v1[x, y, z, :], [0, 0, 1])
            rotated_bvecs = np.dot(R.T, bvecs[:, bvals > 100])
            rotated_bvecs_sph = qbm.cart2sph(rotated_bvecs[0, :], rotated_bvecs[1, :], rotated_bvecs[2, :])
            rotated_bvecs_sh = qbm.get_sh(rotated_bvecs_sph[:, 1], rotated_bvecs_sph[:, 2], self.max_order)
            s = data[x, y, z, bvals >= 100]
            s0 = np.mean(data[x, y, z, bvals < 100])
            if normalize:
                resp = resp + np.linalg.lstsq(rotated_bvecs_sh, s / s0)[0]
            else:
                resp = resp + np.linalg.lstsq(rotated_bvecs_sh, s)[0]

        self.coefficients = resp / np.count_nonzero(mask)
        self.signal = np.dot(bvecs_sh, self.coefficients)

    # I/O
    def read_signal(self, fname):
        '''Reads signal amplitudes from a text file

        :arg fname: (str) input file name
        '''
        self.signal = np.genfromtxt(fname, dtype=float, delimiter=' ')

    def write_signal(self, fname):
        '''Writes signal amplitudes to a text file

        :arg fname: (str) output file name
        '''
        np.savetxt(fname, self.signal, fmt='%.5f', delimiter=' ')

    def read_coefficients(self, fname):
        '''Reads SH coefficients from a text file

        :arg fname: (str) input file name
        '''
        self.coefficients = np.genfromtxt(fname, dtype=float, delimiter=' ')

    def write_coefficients(self, fname):
        '''Writes SH coefficients to a text file

        :arg fname: (str) output file name
        '''
        np.savetxt(fname, self.coefficients, fmt='%.5f', delimiter=' ')