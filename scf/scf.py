"""
The Hartree-Fock driver.
Performs a Hartree-Fock calculation on a given molecule.
"""


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
    """
    # unpack ao_int
    T = ao_int['T']
    V = ao_int['V']
    g = ao_int['g']
    S = ao_int['S']
    A = ao_int['A']

    # build Hcore (T and V are not needed in scf)
    Hcore = T + V

    # unpack scf_params
    nel = scf_params['nel']
    nbas = scf_params['nbas']
    conv = 10. ** (-scf_params['conv'])
    opt = scf_params['opt']
    max_nbf = scf_params['mat_nbf']
    guess = scf_params['core']

    # initial guess


def energy():
    """
    Calculates the energy.
    """
    pass
