import numpy as np
import xform
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
basis = psi4.core.BasisSet.build(mol, target="sto-3g")
mints = psi4.core.MintsHelper(basis)
g = np.array(mints.ao_eri())
A = mints.ao_overlap()
A.power(-0.5, 1.e-14)
A = np.array(A)
print("shape: ", A.shape, g.shape)
print("nbas = ", g.shape[0])

# Symmetric random density
nbf = g.shape[0]

# Reference
g_ref = np.einsum("ip, jq, pqrs, rk, sl -> ijkl", A, A, g, A, A, optimize=True)
print("ref passed")

# cpp
g_our = xform.xform_4_np(g, A)
print("our passed")

# Make sure your implementation is correct
print("g is correct: %s" % np.allclose(g_ref, g_our))
