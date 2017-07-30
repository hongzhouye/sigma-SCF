import numpy as np
import psi4wrapper as p4w

test_scf_param = { 
    'nel': "", 
    'nbas': "", 
    'conv': 5,
    'opt': "None",
    'max_nbf': 120,
    'guess': "Core",
    'max_iter': 100,
    'charge': 0,
    'basis': 'sto-3g',
    'is_fitted': True,
    'geometry': ''' 
    O
    H   1   1.1
    H   1   1.1 2   104
    '''
}

def test_energies():
    """ 
    test for energies
    """
    ao_ints, e_nuclear_repulsion, nel, nbf = p4w.init(test_scf_param)

    eps, C, D, F = p4w.run(ao_ints, test_scf_param)
    H = ao_ints['T'] + ao_ints['V']
    energy = np.sum((F+H)*D) + e_ZZ_repul
    if(test_scf_param['is_fitted']):
        assert(abs(energy-(-74.9421760949686870)) < 1e-6)
    
 
