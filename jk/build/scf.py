import numpy as np
import psi4
import jk
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
basis = psi4.core.BasisSet.build(mol, target="cc-pVDZ")
mints = psi4.core.MintsHelper(basis)
I = np.array(mints.ao_eri())

# Symmetric random density
nbf = I.shape[0]
D = np.random.rand(nbf, nbf)
D = (D + D.T) / 2

# Reference
start = time.time()
for i in range(1000):
    J_ref = np.einsum("pqrs,rs->pq", I, D)
    K_ref = np.einsum("prqs,rs->pq", I, D)
end = time.time()
print(end - start)

# Your implementation
start = time.time()
II = np.swapaxes(I, 1, 2).copy()
for i in range(1000):
    J = jk.getJ_np(I, D)
    K = jk.getJ_np(II, D)
end = time.time()
print(end - start)

# Make sure your implementation is correct
print("J is correct: %s" % np.allclose(J, J_ref))
print("K is correct: %s" % np.allclose(K, K_ref))
