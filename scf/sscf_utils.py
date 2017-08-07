import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from diis_solver import diis_solver, diis_solver_uhf
from scf_utils import get_fock
sys.path.pop()
import jk
import xform


def get_fock_eff(H, g, D):
    nbas = H.shape[0]
    F = get_fock(H, g, D)
    Q = np.eye(nbas) - D
    Feff = F @ (Q - D) @ F + \
        2. * np.einsum("pqrs, qi, ij, jp -> rs", \
            g, Q, F, D, optimize=True) - \
        np.einsum("prqs, qi, ij, jp -> rs", \
            g, Q, F, D, optimize=True) + \
        2. * np.einsum("pqrs, qi, ij, jp -> rs", \
            g, D, F, Q, optimize=True) - \
        np.einsum("prqs, qi, ij, jp -> rs", \
            g, D, F, Q, optimize=True) + \
        2. * np.einsum("mqrs, njkl, qj, rk, sl -> mn", \
            g, g, Q, D, Q, optimize=True) - \
        2. * np.einsum("pmrs, inkl, pi, rk, sl -> mn", \
            g, g, D, D, Q, optimize=True) - \
        np.einsum("mqrs, njkl, ql, rk, sj -> mn", \
            g, g, Q, D, Q, optimize=True) + \
        np.einsum("pmrs, ijkn, pi, rk, sj -> mn", \
            g, g, D, D, Q, optimize=True)
    return Feff


def get_SSCF_variance(H, g, D, unrestricted):
    if unrestricted == True:
        if type(D) is not list:
            raise Exception("For USSCF, Arg4 (D) must be list.")
        Fs = list(get_fock_uhf(H, g, D))
        Ds = D
        nbas = D[0].shape[0]
        Qs = [np.eye(nbas) - D for D in Ds]
        var1 = var2 = 0.
        for i in range(len(Fs)):
            var1 += np.trace(Fs[i] @ Ds[i] @ Fs[i] @ Qs[i])
            var2 -= np.einsum(\
                "pqrs, ijkl, pi, ql, rk, sj ->", \
                g, g, Ds[i], Qs[i], Ds[i], Qs[i], optimize=True)
            for j in range(len(Fs)):
                var2 += np.einsum(\
                    "pqrs, ijkl, pi, qj, rk, sl -> ", \
                    g, g, Ds[i], Qs[i], Ds[j], Qs[j], optimize=True)
        var2 *= 0.5
    else:
        F = get_fock(H, g, D)
        nbas = D.shape[0]
        Q = np.eye(nbas) - D
        var1 = 2. * np.trace(F @ D @ F @ Q)
        var2 = - np.einsum("pqrs, ijkl, pi, ql, rk, sj ->", \
            g, g, D, Q, D, Q, optimize=True) + \
            2. * np.einsum("pqrs, ijkl, pi, qj, rk, sl ->", \
            g, g, D, Q, D, Q, optimize=True)

    return var1 + var2
