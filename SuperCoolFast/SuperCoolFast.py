from init import *

from scf import *

import sys
import numpy
if __name__ == "__main__":
    ao_ints, scf_params, e_ZZ_repul = init(sys.argv[1])
    eps, C, D, F = scf(ao_ints, scf_params)
    H = ao_ints['T'] + ao_ints['V']
    energy = np.sum((F + H) * D) + e_ZZ_repul
    print("\nFINAL SCF ENERGY: {}\n".format(energy))

    # mp2
    import mp2
    print("SCS-MP2 Correlation Energy: {}\n".format(mp2.get_mp2_energy(\
        eps, C, ao_ints['g4'], scf_params['nel'])))
