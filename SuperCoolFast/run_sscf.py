"""
Test run for sigma-SCF
"""

import SuperCoolFast as scf
import numpy as np
import scipy.linalg as slg

def check(H, g, D, omega, nel):
    Feff_mv, Fbeff_mv = scf.get_fock_eff(H, g, D, True, "minvar", None)
    Feff_de, Fbeff_de = scf.get_fock_eff(H, g, D, True, "direct", omega)
    print("D:\n", D[0], "\n", D[1], "\n")
    eps, Ca = slg.eigh(Feff_mv)
    eps, Cb = slg.eigh(Fbeff_mv)
    C_mv = [Ca, Cb]
    print("C_mv:\n", C_mv[0], "\n", C_mv[1], "\n")
    eps, Ca = slg.eigh(Feff_de)
    eps, Cb = slg.eigh(Fbeff_de)
    C_de = [Ca, Cb]
    D_mv = [scf.get_dm(C, nel) for C in C_mv]
    print("D_mv:\n", D_mv[0], "\n", D_mv[1], "\n")
    D_de = [scf.get_dm(C, nel) for C in C_de]
    F_mv = list(scf.get_fock_uhf(H, g, D_mv))
    F_de = list(scf.get_fock_uhf(H, g, D_de))
    e_mv = scf.get_SCF_energy(H, F_mv, D_mv, True)
    e_de = scf.get_SCF_energy(H, F_de, D_de, True)
    print("minvar: %f, direct: %f" % (e_mv, e_de))

# prepare integrals
ao_ints, scf_params, e_nuc = scf.init("test.yml")
scf_params['opt'] = 'oda'
scf_params['oda_convergence'] = -np.log10(1.80e-2)
logger_level = "normal"
#for omega in np.arange(-2, 2, 0.05):
# direct energy targeting
omega = 0.3
scf_params['omega'] = omega
unrestricted = scf_params['unrestricted']
eps, C, D, F = scf.sscf(ao_ints, scf_params, e_nuc, "direct", logger_level)
Fp = list(scf.get_fock_uhf(ao_ints['H'], ao_ints['g4'], D))
print(np.allclose(Fp, F))
print("energy = ", scf.get_SCF_energy(ao_ints['H'], F, D, True))
check(ao_ints['H'], ao_ints['g4'], D, omega, scf_params['nel_alpha'])

# variance minimization
scf_params['guess'] = 'user'
scf_params['guess_C'], scf_params['guess_Cb'] = C[0], C[1]
eps, C, D, F = scf.sscf(ao_ints, scf_params, e_nuc, "minvar", logger_level)
e_scf = scf.get_SCF_energy(ao_ints['H'], F, D, unrestricted)
# output
print("%.2f  % .6f  % .6f" % (omega, e_scf, e_scf + e_nuc))
