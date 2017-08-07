"""
The sigma-SCF driver.
Performs a sigma-SCF calculation on a given molecule.
"""
import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from scf_utils import *
from sscf_utils import *
sys.path.pop()
from scf import rhf, uhf
from collections import deque

np.set_printoptions(suppress=True, precision=3)


def sscf(ao_int, scf_params, e_nuc):
    """
    Solve the sigma-SCF problem given AO integrals (ao_int [dict]):
        - T: ao_kinetic
        - V: ao_potential
        - g: ao_eri
        - S: ao_overlap
        - A: S^(-1/2)
    and parameters for SCF (scf_params [dict]):
        - nel_alpha/beta
                [int]   # of electrons
                        CHECK: non-negative, even integer

        - nbas
                [int]   # of basis functions
                        CHECK: 1. positive integer
                               2. consistent w/ T, V, g, ...

        - conv
                [int]   convg threshold (10E-conv) for ||[F, D]||
                        CHECK: positive integer less than 14

        - opt
                [str]   optimizer to accelerate SCF (case insensitive)
                        CHECK: must be from ['damping', 'diis', 'direct']

        - max_nbf
                [int]   max # of basis functions (for memory concerned)

        - guess
                [str]   initial guess (currently only "core" is supported)

        - max_iter
                [int]   max # of SCF iterations
                        CHECK: positive integers

        - is_fitted
                [bool]  switch for density fitting

        - unrestricted
                [bool]  switch for breaking spin symmetry
    and parameters for sigma-SCF
        - omega
                [float] guess energy
    """
    # rhf or uhf
    if scf_params['unrestricted'] == True:
        return usscf(ao_int, scf_params, e_nuc)
    else:
        return rsscf(ao_int, scf_params, e_nuc)


def rsscf(ao_int, scf_params, e_nuc):
    """
    Spin-restricted Hartree-Fock
    """
    # unpack scf_params
    nel = scf_params['nel_alpha']
    nbas = scf_params['nbas']
    conv = 10. ** (-scf_params['conv'])
    opt = scf_params['opt']
    max_nbf = scf_params['max_nbf']
    guess = scf_params['guess']
    max_iter = scf_params['max_iter']
    is_fitted = scf_params['is_fitted']
    method = scf_params['method']
    omega = scf_params['omega']

    # unpack ao_int
    if scf_params['ortho_ao'] == False:
        raise Exception("ORTHO_AO must be set True for sigma-SCF!")
    T = ao_int['T']
    V = ao_int['V']
    g = ao_int['g3'] if is_fitted else ao_int['g4']
    S = ao_int['S']
    A = ao_int['A']

    # build Hcore (T and V are not needed in scf)
    H = T + V

    # initial guess (case insensitive)
    if guess.upper() == "CORE":
        scf_params['guess'] = 'huckel'
        eps, C, D, F = rhf(ao_int, scf_params, e_nuc, "mute")
        scf_params['guess'] = 'core'
        Q = np.eye(nbas) - D
        Feff = F @ (Q - D) @ F
    elif guess.upper() == "HUCKEL":
        eps, C, D, F = rhf(ao_int, scf_params, e_nuc, "mute")
        Q = np.eye(nbas) - D
        Heff = F @ (Q - D) @ F
        dH = np.diag(Heff)
        Feff = 1.75 * S * (dH.reshape(nbas, 1) + dH) * 0.5
    # elif guess.upper() == "RHF":
    #     F = H
    else:
        raise Exception("Keyword guess must be core.")

    eps, C = diag(Feff, A)
    D = get_dm(C, nel)
    Feff = get_fock_eff(H, g, D)

    # initialize storage of errors and previous Fs if we're doing DIIS
    max_prev_count = 1
    if(opt.upper() == 'DIIS'):
        max_prev_count = 10
    Feff_prev_list = deque([], max_prev_count)
    r_prev_list = deque([], max_prev_count)

    # SCF loop
    conv_flag = False
    for iteration in range(1,(max_iter+1)):
        # oda collect old Fock/DM/Energy
        if opt.upper() == "ODA":
            Dold, Feffold, Wold = \
                D, Feff, get_SSCF_variance(H, g, D)

        # diag and update density matrix
        eps, C = diag(Feff, A)
        D = get_dm(C, nel)

        # get F
        Feff = get_fock_eff(H, g, D)

        # oda: collect new Fock/DM/Energy
        if opt.upper() == "ODA":
            W = get_SSCF_variance(H, g, D)
            lbd = oda_update(Feff - Feffold, D - Dold, W - Wold)
            D = lbd * D + (1. - lbd) * Dold
            Feff = lbd * Feff + (1. - lbd) * Feffold

        # calculate error
        err, err_v = get_SCF_err(S, D, Feff)

        # update F_prev_list and r_prev_list
        Feff_prev_list.append(Feff)
        r_prev_list.append(err_v)

        # diis update
        if opt.upper() == "DIIS":
            Feff = diis_update(Feff_prev_list, r_prev_list) \
                if len(Feff_prev_list) > 1 else get_fock_eff(H, g, D)

        # get energy
        F = get_fock(H, g, D)
        energy = get_SCF_energy(ao_int, F, D, False) + e_nuc
        variance = get_SSCF_variance(H, g, D)

        # print iteratoin info
        print(\
            "iter: {0:2d}, etot: {1:0.8F}, vtot: {2:0.8F}, err: {3:0.5E}"\
            .format(iteration, energy, variance, err))

        # check convergence
        if err < conv:
            conv_flag = True
            print ("  ** R-SCF converges in %d iterations! **" % iteration)
            #eps, C = diag(F, A)
            #D = get_dm(C, nel)
            break


    # post process
    if conv_flag:
        return eps, C, D, get_fock(H, g, D)
    else:
        raise Exception ("  ** R-SCF fails to converge in %d iterations! **"
                         % max_iter)


def usscf(ao_int, scf_params, e_nuc):
    """
    Spin-unrestricted Hartree-Fock
    """
    # unpack scf_params
    nel = scf_params['nel_alpha']
    nelb = scf_params['nel_beta']
    nbas = scf_params['nbas']
    conv = 10. ** (-scf_params['conv'])
    opt = scf_params['opt']
    max_nbf = scf_params['max_nbf']
    guess = scf_params['guess']
    max_iter = scf_params['max_iter']
    is_fitted = scf_params['is_fitted']
    method = scf_params['method']
    unrestricted = scf_params['unrestricted']
    mixing_beta = float(scf_params['homo_lumo_mix']) / 10.

    # unpack ao_int
    T = ao_int['T']
    V = ao_int['V']
    g = ao_int['g3'] if is_fitted else ao_int['g4']
    S = ao_int['S']
    A = ao_int['A']

    # build Hcore (T and V are not needed in scf)
    H = T + V

    # initial guess (case insensitive)
    if guess.upper() == "CORE":
        F = H
    elif guess.upper() == "HUCKEL":
        dH = np.diag(H)
        F = 1.75 * S * (dH.reshape(nbas, 1) + dH) * 0.5
    elif guess.upper() == "RHF":
        scf_params['guess'] = 'huckel'
        eps, C, D, F = rhf(ao_int, scf_params, e_nuc)
        scf_params['guess'] = 'rhf'
    else:
        raise Exception("Keyword guess must be core, huckel or rhf.")

    eps, C = diag(F, A)
    D = get_dm(C, nel)
    Cb = homo_lumo_mix(C, nelb, mixing_beta)
    Db = get_dm(Cb, nelb)
    F, Fb = get_fock_uhf(H, g, [D, Db])

    # initialize storage of errors and previous Fs if we're doing DIIS
    max_prev_count = 1
    if(opt.upper() == 'DIIS'):
        max_prev_count = 10
    F_prev_list = deque([], max_prev_count)
    r_prev_list = deque([], max_prev_count)
    Fb_prev_list = deque([], max_prev_count)
    rb_prev_list = deque([], max_prev_count)

    # SCF loop
    conv_flag = False
    for iteration in range(1,(max_iter+1)):
        # oda collect old Fock/DM/Energy
        if opt.upper() == "ODA":
            Dold, Fold, Dbold, Fbold = D, F, Db, Fb
            Eold = get_SCF_energy(ao_int, [F, Fb], [D, Db], True)

        # diag and update density matrix
        eps, C = diag(F, A)
        D = get_dm(C, nel)
        epsb, Cb = diag(Fb, A)
        Db = get_dm(Cb, nelb)

        # get F
        F, Fb = get_fock_uhf(H, g, [D, Db])

        # oda: collect new Fock/DM/Energy
        if opt.upper() == "ODA":
            E = get_SCF_energy(ao_int, [F, Fb], [D, Db], True)
            lbd = oda_update_uhf(\
                [F - Fold, Fb - Fbold], [D - Dold, Db - Dbold], E - Eold)
            D = lbd * D + (1. - lbd) * Dold
            Db = lbd * Db + (1. - lbd) * Dbold
            F = lbd * F + (1. - lbd) * Fold
            Fb = lbd * Fb + (1. - lbd) * Fbold

        # calculate error
        err, err_v = get_SCF_err(S, D, F)
        errb, errb_v = get_SCF_err(S, Db, Fb)
        errtot = (err + errb) * 0.5

        # update F_prev_list and r_prev_list
        F_prev_list.append(F)
        r_prev_list.append(err_v)
        Fb_prev_list.append(Fb)
        rb_prev_list.append(errb_v)

        # diis update
        if opt.upper() == "DIIS":
            F, Fb = diis_update_uhf(H, g, [D, Db], \
                [F_prev_list, Fb_prev_list], \
                [r_prev_list, rb_prev_list])

        # get energy
        energy = get_SCF_energy(ao_int, [F, Fb], [D, Db], True) + e_nuc

        # print iteratoin info
        print("iter: {0:2d}, etot: {1:0.8F}, err: {2:0.5E}".format(\
            iteration, energy, errtot))

        # check convergence
        if errtot < conv:
            conv_flag = True
            print ("  ** U-SCF converges in %d iterations! **" % iteration)
            eps, C = diag(F, A)
            D = get_dm(C, nel)
            epsb, Cb = diag(Fb, A)
            Db = get_dm(Cb, nelb)
            break


    # post process
    if conv_flag:
        return [eps, epsb], [C, Cb], [D, Db], [F, Fb]
    else:
        raise Exception ("  ** U-SCF fails to converge in %d iterations! **"
                         % max_iter)
