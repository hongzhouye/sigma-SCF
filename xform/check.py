"""
Given ao integrals and a random C (orthogonal matrix), check whether
energy is the same in both AO and MO basis set.
"""


import numpy as np
import scipy.linalg as slg
import xform
import scf
import psi4
import time

# Make sure we get the same random array
np.random.seed(0)

# A hydrogen molecule
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

# Build a ERI tensor
basis = psi4.core.BasisSet.build(mol, target="cc-pvdz")
mints = psi4.core.MintsHelper(basis)
T = np.array(mints.ao_kinetic())
V = np.array(mints.ao_potential())
H = T + V
I = np.array(mints.ao_eri())
A = mints.ao_overlap()
A.power(-0.5, 1.e-14)
A = np.array(A)
nbas = I.shape[0]
nocc = 5
print("nbas = ", I.shape[0])

# diagonalize H to get C
eps, C = scf.diag(H, A)

# Density matrix
Cocc = C[:, :nocc]
D_ref = Cocc @ Cocc.T
D_our = np.zeros([nbas, nbas])
for i in range(nocc):
    D_our[i, i] = 1.

# Reference
h_ref = C.T @ H @ C
g_ref = np.einsum("pi, qj, pqrs, rk, sl -> ijkl", C, C, I, C, C, optimize=True)
J_ref, K_ref = scf.get_JK(False, I, D_ref)
F_ref = H + 2. * J_ref - K_ref
E_ref = np.sum((H + F_ref) * D_ref)

# our implementation
h_our = scf.xform_2(H, C)
g_our = scf.xform_4(I, C)
J_our, K_our = scf.get_JK(False, g_our, D_our)
F_our = h_our + 2. * J_our - K_our
E_our = np.sum((h_our + F_our) * D_our)

# Make sure your implementation is correct
print("h is correct: %s" % np.allclose(h_our, h_ref))
print("g is correct: %s" % np.allclose(g_our, g_ref))
print("E is correct: %s" % np.allclose(E_our, E_ref))
