import numpy as np
from scipy.linalg import pinv
# from scipy.linalg import inv
from numba import njit
from . import measures as m
np.set_printoptions(suppress=True)


class Approximation():
    def __init__(self, trajectory):
        si, sij, si_sj = m.correlations(trajectory)
        self.si = si
        self.sij = sij
        self.si_sj = si_sj
        self.Cij = sij - si_sj

    # this should be a matrix equation take in trajectory?
    def nMF(self):
        P = np.identity(n=self.si.size)
        k = 1 - np.square(self.si)
        k2 = 1 - (self.si ** 2)
        # print(k, k2)
        np.fill_diagonal(P, k)
        Pinv = pinv(P)
        Cinv = pinv(self.Cij)
        JnMF = Pinv - Cinv
        np.fill_diagonal(JnMF, 0)
        # h = than-1(mi) -sidiag Jij)
        h = h_nMF(self.si.astype('float64'), JnMF)
        # h = h_TAP(self.si, JnMF)
        np.fill_diagonal(JnMF, h)
        # print(Pinv)
        # print(Cinv)
        # print(np.diag(JnMF))
        return JnMF


# can use this to approx any h from J! Check that right shapes?
@njit
def h_nMF(si, J):
    # h_i = arctanh(<s_i>) - sum_k(Jik * <s_k>)
    # np.fill_diagonal(J, 0)  # figure out how to fix this!!
    # print(si.dtype)
    # print(J.dtype)
    # h = 0
    h = np.arctanh(si) - (np.diag(si) @ J)
    return h


def h_TAP(si, J):
    # mi sum_j Jsqr (1 - m^2)
    reaction_term = si * ((J ** 2) @ np.diag((1 - (si ** 2))))
    h = h_nMF(si, J) + reaction_term
    return h
