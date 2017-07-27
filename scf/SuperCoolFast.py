from init import *

from scf import *

import sys
import numpy
if __name__ == "__main__":
    scf_params = None
    ao_ints = None
    e_ZZ_repul = None
    init(sys.argv[1])
    eps, C, D = scf(ao_ints, scf_params)
    print(np.sum(eps)) 


