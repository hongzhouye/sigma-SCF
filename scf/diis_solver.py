"""
This is a DIIS solver for finding matrix of error vectors
"""

import numpy as np


def diis_solver(errs):
    n = len(errs)

    B = np.zeros([n, n])
    for i, erra in enumerate(errs):
        for j, errb in enumerate(errs):
            B[i, j] = erra.ravel() @ errb.ravel()

    A = np.ones([n + 1, n + 1])
    A = -A
    A[:n, :n] = B
    A[n, n] = 0

    b = np.zeros(n + 1)
    b[n] = -1

    x = np.linalg.solve(A, b)

    c = x[:n]

    return c


if __name__ == "__main__":
    n = 10
    m = 4
    from collections import deque
    errs = deque([], maxlen=m)
    for i in range(m):
        errs.append(np.random.rand(n, n))

    c = diis_solver(errs)
    print(c)
