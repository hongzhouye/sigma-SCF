import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from diis_solver import diis_solver, diis_solver_uhf
from oda_utils import poly_min
from scf_utils import get_fock, get_fock_uhf, get_SCF_energy
from scf import rhf
sys.path.pop()


def get_SSCF_core_guess(ao_int, scf_params, e_nuc, mode):
    scf_params['guess'] = 'huckel'
    eps, C, D, F = rhf(ao_int, scf_params, e_nuc, "mute")
    scf_params['guess'] = 'core'
    Q = np.eye(scf_params['nbas']) - D
    Feff = F @ (Q - D) @ F
    if mode.upper() == "DIRECT":
        e_scf = get_SCF_energy(ao_int['H'], F, D, False)
        Feff += 2. * (e_scf - scf_params['omega']) * F
    return Feff


def _get_fock_eff(H, g, D, unrestricted):
    if unrestricted == True:
        if type(D) is not list:
            raise Exception("For USSCF, Arg3 (D) must be list.")
        nbas = H.shape[0]
        Ds = D
        Fs = list(get_fock_uhf(H, g, Ds))
        Qs = [np.eye(nbas) - D for D in Ds]
        Feffs = [0 * H, 0 * H]
        for i in [0, 1]:
            Feffs[i] += Fs[i] @ (Qs[i] - Ds[i]) @ Fs[i] - \
                np.einsum("prqs, qi, ij, jp -> rs", \
                g, Qs[i], Fs[i], Ds[i], optimize=True) - \
                np.einsum("prqs, qi, ij, jp -> rs", \
                g, Ds[i], Fs[i], Qs[i], optimize=True) - \
                np.einsum("mqrs, njkl, ql, rk, sj -> mn", \
                g, g, Qs[i], Ds[i], Qs[i], optimize=True) + \
                np.einsum("pmrs, ijkn, pi, rk, sj -> mn", \
                g, g, Ds[i], Ds[i], Qs[i], optimize=True)
            for j in [0, 1]:
                Feffs[i] += np.einsum("pqrs, qi, ij, jp -> rs", \
                    g, Qs[j], Fs[j], Ds[j], optimize=True) + \
                    np.einsum("pqrs, qi, ij, jp -> rs", \
                    g, Ds[j], Fs[j], Qs[j], optimize=True) + \
                    np.einsum("mqrs, njkl, qj, rk, sl -> mn", \
                    g, g, Qs[i], Ds[j], Qs[j], optimize=True) - \
                    np.einsum("pmrs, inkl, pi, rk, sl -> mn", \
                    g, g, Ds[i], Ds[j], Qs[j], optimize=True)
        return tuple(Feffs)
    else:
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


def get_fock_eff(H, g, D, unrestricted, mode, omega):
    if unrestricted == True:
        if type(D) is not list:
            raise Exception("For USSCF, Arg3 (D) must be list.")
        Feff, Fbeff = _get_fock_eff(H, g, D, True)
        if mode.upper() == "DIRECT":
            F, Fb = get_fock_uhf(H, g, D)
            e_scf = get_SCF_energy(H, [F, Fb], D, True)
            Feff += 2. * (e_scf - omega) * F
            Fbeff += 2. * (e_scf - omega) * Fb
        return Feff, Fbeff
    else:
        Feff = _get_fock_eff(H, g, D, False)
        if mode.upper() == "DIRECT":
            F = get_fock(H, g, D)
            e_scf = get_SCF_energy(H, F, D, False)
            Feff += 2. * (e_scf - omega) * F
        return Feff


def _get_SSCF_variance(H, g, D, unrestricted):
    if unrestricted == True:
        if type(D) is not list:
            raise Exception("For USSCF, Arg3 (D) must be list.")
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


def get_SSCF_variance(H, g, D, unrestricted, mode, omega):
    if unrestricted == True:
        if type(D) is not list:
            raise Exception("For USSCF, Arg3 (D) must be list.")
        var = _get_SSCF_variance(H, g, D, True)
        if mode.upper() == "DIRECT":
            F, Fb = get_fock_uhf(H, g, D)
            e_scf = get_SCF_energy(H, [F, Fb], D, True)
            var += (e_scf - omega) ** 2
    else:
        var = _get_SSCF_variance(H, g, D, False)
        if mode.upper() == "DIRECT":
            F = get_fock(H, g, D)
            e_scf = get_SCF_energy(H, F, D, False)
            var += (e_scf - omega) ** 2
    return var


def oda_update_sscf(H, g, D, Dold, var, var_old, unrestricted, \
    mode, omega, deg = 2):
    """
    Do a nth-order polynomial interpolation and then solve for its minimum.
    """
    lbd_set = np.linspace(0.1, 0.9, deg + 2)
    var_set = []
    if unrestricted == True and type(D) is not list:
        raise Exception("For USSCF, Arg3 (D) must be list.")
    for lbd in lbd_set:
        if unrestricted == True:
            Dn = [lbd * x[0] + (1. - lbd) * x[1] for x in zip(D, Dold)]
        else:
            Dn = lbd * D + (1. - lbd) * Dold
        var_set.append(get_SSCF_variance(H, g, Dn, unrestricted, mode, omega))
    var_set = np.array(var_set)
    # fit
    p_coeff = np.polyfit(lbd_set, var_set, deg = deg)
    lbd = poly_min(p_coeff)
    lbd_old = lbd
    if lbd > 1 or lbd < 0:
        lbd = 0.9999 if var_old > var else 0.0001
        #lbd = 1.0001 if var_old > var else -0.0001
    #print("%f  %f  %f" % (lbd_old, lbd, var - var_old))
    return lbd


def get_fock_eff_det(H, g, D, unrestricted, mode, omega):
    if unrestricted == True:
        if type(D) is not list:
            raise Exception("For DET-USSCF, Arg3 (D) must be list.")
        feff, fbeff = get_fock_eff(H, g, D, True, "direct", omega)
        F, Fb = get_fock_uhf(H, g, D)
        e_scf = get_SCF_energy(H, [F, Fb], D, True)
        W = get_SSCF_variance(H, g, D, unrestricted, "direct", omega)
        Feff_det = -F / W + (e_scf - omega) / W ** 2 * feff
        Fbeff_det = -Fb / W + (e_scf - omega) / W ** 2 * fbeff
        return Feff_det, Fbeff_det
    else:
        raise Exception("Currently DET-SSCF does not support spin-"
            "restricted case.")


def get_SSCF_variance_det(H, g, D, unrestricted, mode, omega):
    if unrestricted == True:
        if type(D) is not list:
            raise Exception("For DET-USSCF, Arg3 (D) must be list.")
        F, Fb = get_fock_uhf(H, g, D)
        e_scf = get_SCF_energy(H, [F, Fb], D, True)
        W = get_SSCF_variance(H, g, D, unrestricted, "direct", omega)
        return (omega - e_scf) / W
    else:
        raise Exception("Currently DET-SSCF does not support spin-"
            "restricted case.")


def oda_update_sscf_det(H, g, D, Dold, var, var_old, unrestricted, \
    mode, omega, deg = 2):
    """
    Do a nth-order polynomial interpolation and then solve for its minimum.
    """
    lbd_set = np.linspace(0.1, 0.9, deg + 2)
    var_set = []
    if unrestricted == True and type(D) is not list:
        raise Exception("For USSCF, Arg3 (D) must be list.")
    for lbd in lbd_set:
        if unrestricted == True:
            Dn = [lbd * x[0] + (1. - lbd) * x[1] for x in zip(D, Dold)]
        else:
            Dn = lbd * D + (1. - lbd) * Dold
        var_set.append(\
            get_SSCF_variance_det(H, g, Dn, unrestricted, mode, omega))
    var_set = np.array(var_set)
    # fit
    p_coeff = np.polyfit(lbd_set, var_set, deg = deg)
    lbd = poly_min(p_coeff)
    lbd_old = lbd
    if lbd > 1 or lbd < 0:
        lbd = 0.9999 if var_old > var else 0.0001
        #lbd = 1.0001 if var_old > var else -0.0001
    #print("%f  %f  %f" % (lbd_old, lbd, var - var_old))
    return lbd
