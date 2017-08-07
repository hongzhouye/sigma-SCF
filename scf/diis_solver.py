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

    x = np.linalg.pinv(A) @ b

    c = x[:n]

    return c


def diis_solver_uhf(errs, errbs):
    n = len(errs)

    B = np.zeros([n, n])
    for i, erra in enumerate(errs):
        for j, errb in enumerate(errs):
            B[i, j] = erra.ravel() @ errb.ravel()

    for i, erra in enumerate(errbs):
        for j, errb in enumerate(errbs):
            B[i, j] += erra.ravel() @ errb.ravel()

    A = np.ones([n + 1, n + 1])
    A = -A
    A[:n, :n] = B
    A[n, n] = 0

    b = np.zeros(n + 1)
    b[n] = -1

    x = np.linalg.pinv(A) @ b

    c = x[:n]

    return c
