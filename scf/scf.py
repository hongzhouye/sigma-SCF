"""
The Hartree-Fock driver.
Performs a Hartree-Fock calculation on a given molecule.
"""
import numpy as np
import os, sys
import logging
sys.path.append(os.path.dirname(__file__))
from scf_utils import *
sys.path.pop()
from collections import deque

np.set_printoptions(suppress=True, precision=3)


def scf(ao_int, scf_params, e_nuc, logger_level = "normal"):
    """
    Solve the SCF problem given AO integrals (ao_int [dict]):
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
    """
    # rhf or uhf
    if scf_params['unrestricted'] == True:
        return uhf(ao_int, scf_params, e_nuc, logger_level)
    else:
        return rhf(ao_int, scf_params, e_nuc, logger_level)


def rhf(ao_int, scf_params, e_nuc, logger_level = "normal"):
    """
    Spin-restricted Hartree-Fock
    """
    # setup logger
    if logger_level.upper() == "NORMAL":
        logger_verbose = logging.getLogger("normal")
        logger_concise = logging.getLogger("print")
    elif logger_level.upper() == "MUTE":
        logger_verbose = logging.getLogger("mute")
        logger_concise = logging.getLogger("mute")

    logger_verbose.info("\n\t** Module: Spin-restricted Hartree-Fock **")
    # unpack scf_params
    nel = scf_params['nel_alpha']
    nbas = scf_params['nbas']
    conv = 10. ** (-scf_params['conv'])
    opt = scf_params['opt']
    max_nbf = scf_params['max_nbf']
    guess = scf_params['guess']
    max_iter = scf_params['max_iter']
    is_fitted = scf_params['is_fitted']

    # unpack ao_int
    H = ao_int['H']
    g = ao_int['g3'] if is_fitted else ao_int['g4']
    S = ao_int['S']
    A = ao_int['A']

    # initial guess (case insensitive)
    logger_verbose.info("\n\t** RHF initial guess: %s" % guess.upper())
    if guess.upper() == "CORE":
        F = H
    elif guess.upper() == "HUCKEL":
        dH = np.diag(H)
        F = 1.75 * S * (dH.reshape(nbas, 1) + dH) * 0.5
    elif guess.upper() == "RHF":
        F = H
    else:
        raise Exception("Keyword guess must be core, huckel or rhf.")

    eps, C = diag(F, A)
    D = get_dm(C, nel)
    F = get_fock(H, g, D)
    e_scf = get_SCF_energy(H, F, D, False)

    # initialize storage of errors and previous Fs if we're doing DIIS
    max_prev_count = 1
    if(opt.upper() == 'DIIS'):
        max_prev_count = 10
    F_prev_list = deque([], max_prev_count)
    r_prev_list = deque([], max_prev_count)

    # SCF loop
    logger_verbose.info("\n\t** Starting RHF SCF loop **")
    logger_concise.info("\t\t---------------------------------")
    logger_concise.info("\t\titer   total energy   ||[D, F]||")
    logger_concise.info("\t\t---------------------------------")
    conv_flag = False
    for iteration in range(1,(max_iter+1)):
        # oda collect old Fock/DM/Energy
        if opt.upper() == "ODA":
            Dold, Fold, Eold = D, F, e_scf

        # diag and update density matrix
        eps, C = diag(F, A)
        D = get_dm(C, nel)

        # get F
        F = get_fock(H, g, D)

        # oda: collect new Fock/DM/Energy
        if opt.upper() == "ODA":
            E = get_SCF_energy(H, F, D, False)
            lbd = oda_update(F - Fold, D - Dold, E - Eold)
            D = lbd * D + (1. - lbd) * Dold
            F = lbd * F + (1. - lbd) * Fold

        # calculate error
        err, err_v = get_SCF_err(S, D, F)

        # update F_prev_list and r_prev_list
        F_prev_list.append(F)
        r_prev_list.append(err_v)

        # diis update
        if opt.upper() == "DIIS":
            F = diis_update(F_prev_list, r_prev_list) \
                if len(F_prev_list) > 1 else get_fock(H, g, D)

        # get energy
        e_scf = get_SCF_energy(H, F, D, False)
        e_tot = e_scf + e_nuc

        # print iteratoin info
        logger_concise.info(\
            "\t\t%4d   % 12.8F   %7.4E" % (iteration, e_tot, err))

        # check convergence
        if err < conv:
            conv_flag = True
            break

    logger_concise.info("\t\t---------------------------------\n")
    # check convergence
    if conv_flag:
        logger_verbose.info(\
            "\n\t** RHF converges in %d iterations! **" % iteration)
        return eps, C, D, F
    else:
        raise Exception(\
            "\t** RHF fails to converge in %d iterations! **" % max_iter)


def uhf(ao_int, scf_params, e_nuc, logger_level = "normal"):
    """
    Spin-unrestricted Hartree-Fock
    """
    # setup logger
    if logger_level.upper() == "NORMAL":
        logger_verbose = logging.getLogger("normal")
        logger_concise = logging.getLogger("print")
    elif logger_level.upper() == "MUTE":
        logger_verbose = logging.getLogger("mute")
        logger_concise = logging.getLogger("mute")
    else:
        raise Exception("Arg4 in scf must be 'normal' or 'mute'.")

    logger_verbose.info("\n\t** Module: Spin-unrestricted Hartree-Fock **")
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
    unrestricted = scf_params['unrestricted']
    mixing_beta = float(scf_params['homo_lumo_mix']) / 10.

    # unpack ao_int
    H = ao_int['H']
    g = ao_int['g3'] if is_fitted else ao_int['g4']
    S = ao_int['S']
    A = ao_int['A']

    # initial guess (case insensitive)
    logger_verbose.info("\n\t** UHF initial guess: %s" % guess.upper())
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
    e_scf = get_SCF_energy(H, [F, Fb], [D, Db], True)

    # initialize storage of errors and previous Fs if we're doing DIIS
    max_prev_count = 1
    if(opt.upper() == 'DIIS'):
        max_prev_count = 10
    F_prev_list = deque([], max_prev_count)
    r_prev_list = deque([], max_prev_count)
    Fb_prev_list = deque([], max_prev_count)
    rb_prev_list = deque([], max_prev_count)

    # SCF loop
    logger_verbose.info("\n\t** Starting UHF SCF loop **")
    logger_concise.info("\t\t---------------------------------")
    logger_concise.info("\t\titer   total energy   ||[D, F]||")
    logger_concise.info("\t\t---------------------------------")
    conv_flag = False
    for iteration in range(1,(max_iter+1)):
        # oda collect old Fock/DM/Energy
        if opt.upper() == "ODA":
            Dold, Fold, Dbold, Fbold, Eold = D, F, Db, Fb, e_scf

        # diag and update density matrix
        eps, C = diag(F, A)
        D = get_dm(C, nel)
        epsb, Cb = diag(Fb, A)
        Db = get_dm(Cb, nelb)

        # get F
        F, Fb = get_fock_uhf(H, g, [D, Db])

        # oda: collect new Fock/DM/Energy
        if opt.upper() == "ODA":
            E = get_SCF_energy(H, [F, Fb], [D, Db], True)
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
            F, Fb = diis_update_uhf(\
                [F_prev_list, Fb_prev_list], \
                [r_prev_list, rb_prev_list]) \
                if len(F_prev_list) > 1 else get_fock_uhf(H, g, [D, Db])

        # get energy
        e_scf = get_SCF_energy(H, [F, Fb], [D, Db], True)
        e_tot = e_scf + e_nuc

        # print iteratoin info
        logger_concise.info(\
            "\t\t%4d   % 12.8F   %7.4E" % (iteration, e_tot, errtot))

        # check convergence
        if errtot < conv:
            conv_flag = True
            break

    logger_concise.info("\t\t---------------------------------\n")
    # check convergence
    if conv_flag:
        logger_verbose.info(\
            "\n\t** UHF converges in %d iterations! **" % iteration)
        return [eps, epsb], [C, Cb], [D, Db], [F, Fb]
    else:
        raise Exception(\
            "\t** UHF fails to converge in %d iterations! **" % max_iter)
