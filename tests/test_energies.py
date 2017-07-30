import numpy as np
import SuperCoolFast as scf

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
    ao_ints, test_scf_param, e_ZZ_repulsion = scf.init("test_scf_param.txt")

    eps, C, D, F = scf.scf(ao_ints, test_scf_param)
    H = ao_ints['T'] + ao_ints['V']
    energy = np.sum((F+H)*D) + e_ZZ_repulsion
    if(test_scf_param['is_fitted']):
        assert(abs(energy-(-74.9421760949686870)) < 1e-6)
    
 
