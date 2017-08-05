import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from diis_solver import diis_solver, diis_solver_uhf
sys.path.pop()
import jk
import xform


def homo_lumo_mix(C, nocc, beta):
    """
    Mix a portion of LUMO to HOMO.
    Used when generating spin-unrestricted guess.
    """
    if beta < 0. or beta > 1.:
        raise Exception("Mixing beta must be in [0, 1]")
    Cb = C.copy()
    homo = C[:, nocc - 1]
    lumo = C[:, nocc]
    Cb[:, nocc - 1] = (1. - beta) ** 0.5 * homo + beta ** 0.5 * lumo
    return Cb


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
        #J = np.einsum("pqrs,rs->pq", g, D)
        #K = np.einsum("prqs,rs->pq", g, D)
        J, K = jk.getJK_np_Dshift(g, D - np.diag(np.diag(D) * 0.5))
        return (J, K)


def get_JK_uhf(is_fitted, g, Ds):
    """
    Ds = [Da, Db]
    """
    Da, Db = Ds[0], Ds[1]
    Dtot = Da + Db
    if (is_fitted == True):
        X = np.einsum("Pls,ls->P", g, Dtot)
        Jtot = np.einsum("mnP,P->mn", np.swapaxes(g, 0, 2), X)
        Za = np.einsum("Pns,ls->Pnl", g, Da)
        Ka = np.einsum('mlP,Pnl->mn', np.swapaxes(g, 0, 2), Za)
        Zb = np.einsum("Pns,ls->Pnl", g, Db)
        Kb = np.einsum('mlP,Pnl->mn', np.swapaxes(g, 0, 2), Zb)
        return Jtot, Ka, Kb
    else:
        Jtot = np.einsum("pqrs, rs -> pq", g, Dtot)
        Ka = np.einsum("prqs, rs -> pq", g, Da)
        Kb = np.einsum("prqs, rs -> pq", g, Db)
        return Jtot, Ka, Kb


def get_fock(H, g, D):
    J, K = get_JK(len(g.shape) == 3, g, D)
    return H + 2 * J - K


def diis_update(H, g, D, F_prev_list, r_prev_list):
    """
    DIIS update given previous Fock matrices and error vectors.
    Note that if there are less than two F's, return normal F.
    """
    if(len(F_prev_list) <= 1):
        return get_fock(H, g, D)
    else:
        c = diis_solver(r_prev_list) # GET THE COEFFICIENTS!!
        out = 0 * H
        for i, element in enumerate(F_prev_list):
            out += c[i] * element
        return out


def oda_update(dF, dD, dE):
    """
    ODA update:
        lbd = 0.5 - dE / E_deriv
    """
    E_deriv = np.sum(dF * dD)
    lbd = 0.5 * (1. - dE / E_deriv)
    if lbd < 0 or lbd > 1:
        lbd = 0.9999 if dE < 0 else 1.e-4
    return lbd


def get_fock_uhf(H, g, Ds, opt, F_prev_lists, r_prev_lists):
    """
    if opt == "NONE":
        Get uhf Fock matrices via:
            Fa = Hcore + J[Dtot] - K[Da]
            Fb = Hcore + J[Dtot] - K[Db]
        where
            Dtot = Da + Db
    else:
        diis update
    """
    opt = opt.upper()
    if(opt == 'FP' or len(F_prev_lists[0]) <= 1):
        # Fixed point update
        Jtot, Ka, Kb = get_JK_uhf(len(g.shape) == 3, g, Ds)
        return H + Jtot - Ka, H + Jtot - Kb
    elif opt == 'DIIS':
        c = diis_solver_uhf(r_prev_lists[0], r_prev_lists[1])
        Fa = 0 * H
        for i, element in enumerate(F_prev_lists[0]):
            Fa += c[i] * element
        Fb = 0 * H
        for i, element in enumerate(F_prev_lists[1]):
            Fb += c[i] * element
        return Fa, Fb


def diag(F, A):
    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C


def get_SCF_err(S, D, F):
    err_v = S @ D @ F - F @ D @ S
    err = np.mean(err_v ** 2) ** 0.5
    return err, err_v


def get_SCF_energy(ao_ints, F, D, unrestricted):
    """
    Calculates the energy.
    """
    H = ao_ints['T'] + ao_ints['V']
    if unrestricted == True:
        if type(F) is not list or type(D) is not list:
            raise Exception("For UHF, F and D must have type list.")
        Fa, Fb = F[0], F[1]
        Da, Db = D[0], D[1]
        Dtot = Da + Db
        return np.sum(Dtot * H + Da * Fa + Db * Fb) * 0.5
    else:
        return np.sum((H + F) * D)


def xform_2(H, A):
    """
    Basis xform for 2-tensor
    """
    if len(H.shape) != 2:
        raise Exception("Dimension error: arg1 should be a matrix")

    return A.T @ H @ A


def xform_4(g, A):
    """
    Basis xform for 4-tensor
    """
    if len(g.shape) != 4:
        raise Exception("""
            Dimension error: arg1 should be a four-tensor.
            Note that you should set is_fitted to be False.
        """)

    #return np.einsum("pi, qj, pqrs, rk, sl -> ijkl", A, A, g, A, A, optimize=True)
    return xform.xform_4_np(g, A)
