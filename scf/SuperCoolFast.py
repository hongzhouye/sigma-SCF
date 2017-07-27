from init import *

from scf import *

import sys
import numpy
if __name__ == "__main__":
    ao_ints, scf_params, e_ZZ_repul = init(sys.argv[1])
    eps, C, D = scf(ao_ints, scf_params)


