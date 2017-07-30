import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from diis_solver import diis_solver
sys.path.pop()


def get_dm(C, nel):
    D = C[:, :nel]
    D = D @ D.T
    return D


def get_JK(is_fitted, g, D):
    if(is_fitted):
        # FINISH LATER
        X = np.einsum("Pls,ls->P", g, D)
        J = np.einsum("mnP,P->mn", np.swapaxes(g, 0, 2), X)
        Z = np.einsum("Pns,ls->Pnl", g, D)
        K = np.einsum('mlP,Pnl->mn', np.swapaxes(g, 0, 2), Z)
        return (J, K)
    else:
        J = np.einsum("pqrs,rs->pq", g, D)
        K = np.einsum("prqs,rs->pq", g, D)
        return (J, K)


def get_fock(H, g, D, opt, F_prev_list, r_prev_list):
    opt = opt.upper()
    # not accelerated
    if(opt == 'NONE' or len(F_prev_list) <= 1):
        J, K = get_JK(True, g, D)
        return H + 2 * J - K
    # DIIS
    elif(opt == 'DIIS'):
        c = diis_solver(r_prev_list) # GET THE COEFFICIENTS!!
        out = 0 * H
        for i, element in enumerate(F_prev_list):
            out += c[i] * element
        return out


def diag(F, A):
    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C


def get_SCF_err(S, D, F):
    return (np.sum(np.abs(S @ D @ F - F @ D @ S)), (S @ D @ F - F @ D @ S))
