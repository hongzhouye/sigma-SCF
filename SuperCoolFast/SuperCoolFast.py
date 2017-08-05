from init import *

from scf import *

import sys
import numpy
if __name__ == "__main__":
    ao_ints, scf_params, e_ZZ_repul = init(sys.argv[1])
    eps, C, D, F = scf(ao_ints, scf_params, e_ZZ_repul)
    energy = get_SCF_energy(ao_ints, F, D, scf_params['unrestricted']) \
        + e_ZZ_repul
    print("\nFINAL SCF ENERGY: {}\n".format(energy))

    # mp2
    if(scf_params['method'] == "MP2"):
       import mp2
       print("SCS-MP2 Correlation Energy: {}\n".format(mp2.get_mp2_energy(\
           eps, C, ao_ints['g4'], scf_params['nel'])))
