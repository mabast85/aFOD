import numpy as np
from scipy.special import lpmv
from scipy.misc import factorial


def cart2sph(x, y, z):
    
    res = np.zeros((x.size, 3), dtype=float)
    r = np.sqrt((x * x) + (y * y) + (z * z))
    res[:, 0] = r   # radius
    res[:, 1] = np.arctan2(y, x)    # theta [-pi, pi]
    res[res[:, 1] < 0, 1] = res[res[:, 1] < 0, 1] + 2 * np.pi # theta [0, 2pi]
    res[:, 2] = np.arccos(z / r)    # phi
    
    return res

def sph2cart(r, t, p):
    
    res = np.zeros((r.size, 3), dtype=float)
    r_sinp = r * np.sin(p)
    res[:, 0] = r_sinp * np.cos(t)    # x
    res[:, 1] = r_sinp * np.sin(t)    # y
    res[:, 2] = r * np.cos(p)         # z

    return res


def get_sh(theta, phi, L=0, c='even', m='sd'):

    n_points = theta.size
    if c == 'even':
        n_coeffs = (L+1)*(L+2)/2
        m, l = np.concatenate([[(m, l) for m in range(-l, l+1)] for l in range(0, L+1, 2)], axis=0).T
    elif c == 'odd':
        n_coeffs = np.sum(np.arange(1, 2*(L+1), 2)) - ((L+1)*(L+2))/2
        m, l = np.concatenate([[(m, l) for m in range(-l, l+1)] for l in range(1, L+1, 2)], axis=0).T
    elif c == 'all':
        n_coeffs = np.sum(np.arange(1, 2*(L+1), 2))
        m, l = np.concatenate([[(m, l) for m in range(-l, l+1)] for l in range(0, L+1)], axis=0).T
        
    sh = np.zeros((int(n_points), int(n_coeffs)), dtype=float)
    c1 = np.sqrt((2*l + 1) / (4 * np.pi))
    c2 = np.sqrt(factorial(l - np.abs(m)) / factorial(l + np.abs(m)))
    Lml = lpmv(np.abs(m[np.newaxis, :]), l[np.newaxis, :], np.cos(phi[:, np.newaxis]))
    Yml = c1[np.newaxis, :] * c2[np.newaxis, :] * Lml * np.exp(1j * np.abs(m[np.newaxis, :]) * theta[:, np.newaxis])

    sh[:, m < 0] = np.sqrt(2) * np.imag(Yml[:, m < 0])
    sh[:, m == 0] = np.real(Yml[:, m == 0])  
    sh[:, m > 0] = np.sqrt(2) * np.real(Yml[:, m > 0])

    return sh


def get_delta(theta, phi, order):
    return get_sh(theta, phi, order)


def get_rotation(v1, v2):
    n = np.cross(v2, v1)
    norm_n = np.linalg.norm(n)
    angle = np.arctan2(norm_n, np.dot(v2, v1))
    n = n / norm_n
    R = np.array([[0, -n[2], n[1]],
                  [n[2], 0, -n[0]],
                  [-n[1], n[0], 0]])
    R = R * np.sin(angle) + np.eye(3) * np.cos(angle) + (1 - np.cos(angle)) * np.outer(n, n)

    return R
