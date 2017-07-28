import numpy as np

def get_dm(C, nel):
    D = C[:, :nel]
    D = D @ D.T
    return D


def get_fock(H, g, D):
    J = np.einsum("pqrs,rs->pq", g, D)
    K = np.einsum("prqs,rs->pq", g, D)
    return H + 2 * J - K


def diag(F, A):
    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C


def get_SCF_err(S, D, F):
    return np.sum(np.abs(S @ D @ F - F @ D @ S))
