"""
The Hartree-Fock driver.
Performs a Hartree-Fock calculation on a given molecule.
"""
import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from scf_utils import *
sys.path.pop()
from collections import deque


def scf(ao_int, scf_params):
    """
    Solve the SCF problem given AO integrals (ao_int [dict]):
        - T: ao_kinetic
        - V: ao_potential
        - g: ao_eri
        - S: ao_overlap
        - A: S^(-1/2)
    and parameters for SCF (scf_params [dict]):
        - nel   [int]   # of electrons
                        CHECK: non-negative, even integer

        - nbas  [int]   # of basis functions
                        CHECK: 1. positive integer
                               2. consistent w/ T, V, g, ...

        - conv  [int]   convg threshold (10E-conv) for ||[F, D]||
                        CHECK: positive integer less than 14

        - opt   [str]   optimizer to accelerate SCF (case insensitive)
                        CHECK: must be from ['damping', 'diis', 'direct']

        - max_nbf[int]  max # of basis functions (for memory concerned)

        - guess [str]   initial guess (currently only "core" is supported)

        - max_iter[int] max # of SCF iterations
                        CHECK: positive integers
    """
    # unpack ao_int
    T = ao_int['T']
    V = ao_int['V']
    g = ao_int['g']
    S = ao_int['S']
    A = ao_int['A']
    F = None

    # build Hcore (T and V are not needed in scf)
    H = T + V

    # unpack scf_params
    nel = scf_params['nel']
    nbas = scf_params['nbas']
    conv = 10. ** (-scf_params['conv'])
    opt = scf_params['opt']
    max_nbf = scf_params['max_nbf']
    guess = scf_params['guess']
    max_iter = scf_params['max_iter']

    # initial guess (case insensitive)
    err = None
    if guess.upper() == "CORE":
        eps, C = diag(H, A)
        D = get_dm(C, nel)
        F = get_fock(H, g, D, 'NONE', [], [])
    else:
        raise Exception("Currently only core guess is supported!")

    # initialize storage of errors and previous Fs if we're doing DIIS
    max_prev_count = 1
    if(opt.upper() == 'DIIS'):
        max_prev_count = 10
    F_prev_list = deque([], max_prev_count)
    r_prev_list = deque([], max_prev_count)

    # SCF loop
    conv_flag = False
    for iteration in range(1,(max_iter+1)):
        # diag and update density matrix
        eps, C = diag(F, A)
        D = get_dm(C, nel)

        # get F
        F = get_fock(H, g, D, 'NONE', F_prev_list, r_prev_list)

        # calculate error
        err, err_v = get_SCF_err(S, D, F)

        # update F_prev_list and r_prev_list
        F_prev_list.append(F)
        r_prev_list.append(err_v)

        # diis update
        F = get_fock(H, g, D, opt, F_prev_list, r_prev_list)

        # print iteratoin info
        print("iter: {0:2d}, err: {1:0.5E}".format(iteration, err))


        # check convergence
        if err < conv:
            conv_flag = True
            print ("  ** SCF converges in %d iterations! **" % iteration)
            break


    # post process
    if conv_flag:
        return eps, C, D, F
    else:
        raise Exception ("  ** SCF fails to converge in %d iterations! **"
                         % max_iter)


def energy():
    """
    Calculates the energy.
    """
    pass
