"""
The sigma-SCF driver.
Performs a sigma-SCF calculation on a given molecule.
"""
import numpy as np
import os, sys
import logging
sys.path.append(os.path.dirname(__file__))
from scf_utils import *
from sscf_utils import *
sys.path.pop()
from scf import rhf, uhf
from collections import deque

np.set_printoptions(suppress=True, precision=3)


def sscf(ao_int, scf_params, e_nuc, mode = "minvar", logger_level = "normal"):
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
        return usscf(ao_int, scf_params, e_nuc, mode, logger_level)
    else:
        return rsscf(ao_int, scf_params, e_nuc, mode, logger_level)


def rsscf(ao_int, scf_params, e_nuc, mode = "minvar", logger_level = "normal"):
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

    logger_verbose.info("\n\t** Module: Spin-restricted sigma-SCF **")
    # unpack scf_params
    nel = scf_params['nel_alpha']
    nbas = scf_params['nbas']
    conv = 10. ** (-scf_params['conv'])
    opt = scf_params['opt']
    max_nbf = scf_params['max_nbf']
    guess = scf_params['guess']
    max_iter = scf_params['max_iter']
    is_fitted = scf_params['is_fitted']
    omega = scf_params['omega']

    logger_verbose.info("\n\t** MODE: %s" % (\
        "direct energy targeting" if mode.upper() == "DIRECT" \
        else "variance minimization"))

    # unpack ao_int
    if scf_params['ortho_ao'] == False:
        raise Exception("ORTHO_AO must be set True for sigma-SCF!")
    H = ao_int['H']
    g = ao_int['g3'] if is_fitted else ao_int['g4']
    S = ao_int['S']
    A = ao_int['A']

    # initial guess (case insensitive)
    logger_verbose.info("\n\t** RSSCF initial guess: %s" % guess.upper())
    if guess.upper() == "CORE":
        Feff = get_SSCF_core_guess(ao_int, scf_params, e_nuc, mode)
    elif guess.upper() == "HUCKEL":
        Heff = get_SSCF_core_guess(ao_int, scf_params, e_nuc, mode)
        dH = np.diag(Heff)
        Feff = 1.75 * S * (dH.reshape(nbas, 1) + dH) * 0.5
    # elif guess.upper() == "RHF":
    #     F = H
    else:
        raise Exception("Keyword guess must be core.")

    eps, C = diag(Feff, A)
    D = get_dm(C, nel)
    Feff = get_fock_eff(H, g, D, False, mode, omega)
    var = get_SSCF_variance(H, g, D, False, mode, omega)

    # initialize storage of errors and previous Fs if we're doing DIIS
    max_prev_count = 1
    if(opt.upper() == 'DIIS'):
        max_prev_count = 10
    Feff_prev_list = deque([], max_prev_count)
    r_prev_list = deque([], max_prev_count)

    # SCF loop
    logger_verbose.info("\n\t** Starting RSSCF SCF loop **")
    logger_concise.info("\t" * 2 + "-" * 48)
    logger_concise.info("\t\titer   total energy     variance     ||[D, F]||")
    logger_concise.info("\t" * 2 + "-" * 48)
    conv_flag = False
    for iteration in range(1,(max_iter+1)):
        # oda collect old Fock/DM/Energy
        if opt.upper() == "ODA":
            Dold, Feffold, Wold = D, Feff, var

        # diag and update density matrix
        eps, C = diag(Feff, A)
        D = get_dm(C, nel)

        # get F
        Feff = get_fock_eff(H, g, D, False, mode, omega)

        # oda: collect new Fock/DM/Energy
        if opt.upper() == "ODA":
            W = get_SSCF_variance(H, g, D, False, mode, omega)
            lbd = oda_update_sscf(\
                H, g, D, Dold, W, Wold, False, mode, omega, 2)
            D = lbd * D + (1. - lbd) * Dold
            Feff = get_fock_eff(H, g, D, False, mode, omega)

        # calculate error
        err, err_v = get_SCF_err(S, D, Feff)

        # update F_prev_list and r_prev_list
        Feff_prev_list.append(Feff)
        r_prev_list.append(err_v)

        # diis update
        if opt.upper() == "DIIS":
            Feff = diis_update(Feff_prev_list, r_prev_list) \
                if len(Feff_prev_list) > 1 \
                else get_fock_eff(H, g, D, False, mode, omega)

        # get energy
        F = get_fock(H, g, D)
        e_tot = get_SCF_energy(H, F, D, False) + e_nuc
        var = get_SSCF_variance(H, g, D, False, mode, omega)

        # print iteratoin info
        logger_concise.info(\
            "\t\t%4d   % 12.8F  %12.8F   %7.4E" % (iteration, e_tot, var, err))

        # check convergence
        if err < conv:
            conv_flag = True
            break

    logger_concise.info("\t" * 2 + "-" * 48 + "\n")
    # post process
    if conv_flag:
        logger_verbose.info(\
            "\n\t** RSSCF converges in %d iterations! **" % iteration)
        return eps, C, D, get_fock(H, g, D)
    else:
        raise Exception(\
            "\t** RSSCF fails to converge in %d iterations! **" % max_iter)


def usscf(ao_int, scf_params, e_nuc, mode = "minvar", logger_level = "normal"):
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

    logger_verbose.info("\n\t** Module: Spin-unrestricted sigma-SCF **")
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
    omega = scf_params['omega']

    # unpack ao_int
    if scf_params['ortho_ao'] == False:
        raise Exception("ORTHO_AO must be set True for sigma-SCF!")
    H = ao_int['H']
    g = ao_int['g3'] if is_fitted else ao_int['g4']
    S = ao_int['S']
    A = ao_int['A']

    # initial guess (case insensitive)
    logger_verbose.info("\n\t** USSCF initial guess: %s" % guess.upper())
    if guess.upper() == "CORE":
        Feff = get_SSCF_core_guess(ao_int, scf_params, e_nuc, mode)
    elif guess.upper() == "HUCKEL":
        Heff = get_SSCF_core_guess(ao_int, scf_params, e_nuc, mode)
        dH = np.diag(Heff)
        Feff = 1.75 * S * (dH.reshape(nbas, 1) + dH) * 0.5
    elif guess.upper() == "USER":
        C, Cb = scf_params['guess_C'], scf_params['guess_Cb']
        D, Db = get_dm(C, nel), get_dm(Cb, nelb)
    else:
        raise Exception("Keyword guess must be core, huckel.")

    if guess.upper() == "CORE" or guess.upper() == "HUCKEL":
        eps, C = diag(Feff, A)
        D = get_dm(C, nel)
        Cb = homo_lumo_mix(C, nelb, mixing_beta)
        Db = get_dm(Cb, nelb)

    Feff, Fbeff = get_fock_eff(H, g, [D, Db], True, mode, omega)
    var = get_SSCF_variance(H, g, [D, Db], True, mode, omega)

    # initialize storage of errors and previous Fs if we're doing DIIS
    max_prev_count = 1
    if 'DIIS' in opt.upper():   # opt = DIIS or ODA+DIIS
        max_prev_count = 10
    Feff_prev_list = deque([], max_prev_count)
    r_prev_list = deque([], max_prev_count)
    Fbeff_prev_list = deque([], max_prev_count)
    rb_prev_list = deque([], max_prev_count)

    # SCF loop
    logger_verbose.info("\n\t** Starting USSCF SCF loop **")
    logger_concise.info("\t" * 2 + "-" * 48)
    logger_concise.info("\t\titer   total energy     variance     ||[D, F]||")
    logger_concise.info("\t" * 2 + "-" * 48)
    conv_flag = False
    for iteration in range(1,(max_iter+1)):
        # oda collect old Fock/DM/Energy
        if opt.upper() == "ODA":
            Dold, Feffold, Dbold, Fbeffold, Wold = D, Feff, Db, Fbeff, var

        # diag and update density matrix
        eps, C = diag(Feff, A)
        D = get_dm(C, nel)
        epsb, Cb = diag(Fbeff, A)
        Db = get_dm(Cb, nelb)

        # get F
        Feff, Fbeff = get_fock_eff(H, g, [D, Db], True, mode, omega)

        # oda: collect new Fock/DM/Energy
        if opt.upper() == "ODA":
            W = get_SSCF_variance(H, g, [D, Db], True, mode, omega)
            lbd = oda_update_sscf(
                H, g, [D, Db], [Dold, Dbold], W, Wold, True, mode, omega, 2)
            D = lbd * D + (1. - lbd) * Dold
            Db = lbd * Db + (1. - lbd) * Dbold
            Feff, Fbeff = get_fock_eff(H, g, [D, Db], True, mode, omega)

        # calculate error
        err, err_v = get_SCF_err(S, D, Feff)
        errb, errb_v = get_SCF_err(S, Db, Fbeff)
        errtot = (err + errb) * 0.5

        # update F_prev_list and r_prev_list
        Feff_prev_list.append(Feff)
        r_prev_list.append(err_v)
        Fbeff_prev_list.append(Fbeff)
        rb_prev_list.append(errb_v)

        # diis update
        if opt.upper() == "DIIS":
            Feff, Fbeff = diis_update_uhf(\
                [Feff_prev_list, Fbeff_prev_list], \
                [r_prev_list, rb_prev_list]) \
                if len(Feff_prev_list) > 1 \
                else get_fock_eff(H, g, [D, Db], True, mode, omega)

        # get energy
        F, Fb = get_fock_uhf(H, g, [D, Db])
        e_tot = get_SCF_energy(H, [F, Fb], [D, Db], True) + e_nuc
        var = get_SSCF_variance(H, g, [D, Db], True, mode, omega)

        # print iteratoin info
        logger_concise.info(\
            "\t\t%4d   % 12.8F  %12.8F   %7.4E" % (iteration, e_tot, var, err))

        # check convergence
        if errtot < conv:
            conv_flag = True
            break

    logger_concise.info("\t" * 2 + "-" * 48 + "\n")
    # post process
    if conv_flag:
        logger_verbose.info(\
            "\n\t** USSCF converges in %d iterations! **" % iteration)
        return [eps, epsb], [C, Cb], [D, Db], [F, Fb], conv_flag
    else:
        #raise Exception(\
        #    "\t** U-SCF fails to converge in %d iterations! **" % max_iter)
        return [eps, epsb], [C, Cb], [D, Db], [F, Fb], conv_flag


def det_usscf(ao_int, scf_params, e_nuc, \
    mode = "direct", logger_level = "normal"):
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

    logger_verbose.info("\n\t** Module: Spin-unrestricted sigma-SCF **")
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
    omega = scf_params['omega']

    # unpack ao_int
    if scf_params['ortho_ao'] == False:
        raise Exception("ORTHO_AO must be set True for sigma-SCF!")
    H = ao_int['H']
    g = ao_int['g3'] if is_fitted else ao_int['g4']
    S = ao_int['S']
    A = ao_int['A']

    # initial guess (case insensitive)
    logger_verbose.info("\n\t** DET-USSCF initial guess: %s" % guess.upper())
    if guess.upper() == "CORE":
        Feff = get_SSCF_core_guess(ao_int, scf_params, e_nuc, mode)
    elif guess.upper() == "HUCKEL":
        Heff = get_SSCF_core_guess(ao_int, scf_params, e_nuc, mode)
        dH = np.diag(Heff)
        Feff = 1.75 * S * (dH.reshape(nbas, 1) + dH) * 0.5
    elif guess.upper() == "USER":
        C, Cb = scf_params['guess_C'], scf_params['guess_Cb']
        D, Db = get_dm(C, nel), get_dm(Cb, nelb)
    else:
        raise Exception("Keyword guess must be core, huckel.")

    if guess.upper() == "CORE" or guess.upper() == "HUCKEL":
        eps, C = diag(Feff, A)
        D = get_dm(C, nel)
        Cb = homo_lumo_mix(C, nelb, mixing_beta)
        Db = get_dm(Cb, nelb)

    Feff, Fbeff = get_fock_eff_det(H, g, [D, Db], True, mode, omega)
    var = get_SSCF_variance_det(H, g, [D, Db], True, mode, omega)

    # initialize storage of errors and previous Fs if we're doing DIIS
    max_prev_count = 1
    if 'DIIS' in opt.upper():   # opt = DIIS or ODA+DIIS
        max_prev_count = 10
    Feff_prev_list = deque([], max_prev_count)
    r_prev_list = deque([], max_prev_count)
    Fbeff_prev_list = deque([], max_prev_count)
    rb_prev_list = deque([], max_prev_count)

    # SCF loop
    logger_verbose.info("\n\t** Starting DET-USSCF SCF loop **")
    logger_concise.info("\t" * 2 + "-" * 48)
    logger_concise.info("\t\titer   total energy     variance     ||[D, F]||")
    logger_concise.info("\t" * 2 + "-" * 48)
    conv_flag = False
    for iteration in range(1,(max_iter+1)):
        # oda collect old Fock/DM/Energy
        if opt.upper() == "ODA":
            Dold, Feffold, Dbold, Fbeffold, Wold = D, Feff, Db, Fbeff, var

        # diag and update density matrix
        eps, C = diag(Feff, A)
        D = get_dm(C, nel)
        epsb, Cb = diag(Fbeff, A)
        Db = get_dm(Cb, nelb)

        # get F
        Feff, Fbeff = get_fock_eff_det(H, g, [D, Db], True, mode, omega)

        # oda: collect new Fock/DM/Energy
        if opt.upper() == "ODA":
            W = get_SSCF_variance_det(H, g, [D, Db], True, mode, omega)
            lbd = oda_update_sscf_det(
                H, g, [D, Db], [Dold, Dbold], W, Wold, True, mode, omega, 2)
            D = lbd * D + (1. - lbd) * Dold
            Db = lbd * Db + (1. - lbd) * Dbold
            Feff, Fbeff = get_fock_eff_det(H, g, [D, Db], True, mode, omega)

        # calculate error
        err, err_v = get_SCF_err(S, D, Feff)
        errb, errb_v = get_SCF_err(S, Db, Fbeff)
        errtot = (err + errb) * 0.5

        # update F_prev_list and r_prev_list
        Feff_prev_list.append(Feff)
        r_prev_list.append(err_v)
        Fbeff_prev_list.append(Fbeff)
        rb_prev_list.append(errb_v)

        # diis update
        if opt.upper() == "DIIS":
            Feff, Fbeff = diis_update_uhf(\
                [Feff_prev_list, Fbeff_prev_list], \
                [r_prev_list, rb_prev_list]) \
                if len(Feff_prev_list) > 1 \
                else get_fock_eff_det(H, g, [D, Db], True, mode, omega)

        # get energy
        F, Fb = get_fock_uhf(H, g, [D, Db])
        e_tot = get_SCF_energy(H, [F, Fb], [D, Db], True) + e_nuc
        var = get_SSCF_variance_det(H, g, [D, Db], True, mode, omega)

        # print iteratoin info
        logger_concise.info(\
            "\t\t%4d   % 12.8F  %12.8F   %7.4E" % (iteration, e_tot, var, err))

        # check convergence
        if errtot < conv:
            conv_flag = True
            break

    logger_concise.info("\t" * 2 + "-" * 48 + "\n")
    # post process
    if conv_flag:
        logger_verbose.info(\
            "\n\t** USSCF converges in %d iterations! **" % iteration)
        return [eps, epsb], [C, Cb], [D, Db], [F, Fb], conv_flag
    else:
        #raise Exception(\
        #    "\t** U-SCF fails to converge in %d iterations! **" % max_iter)
        return [eps, epsb], [C, Cb], [D, Db], [F, Fb], conv_flag
