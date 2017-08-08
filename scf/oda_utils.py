import numpy as np


def poly_min(p_coeff):
    def poly_deriv(p_coeff):
        deg = len(p_coeff) - 1
        dp_coeff = np.arange(deg, -1, -1) * p_coeff
        dp_coeff = np.delete(dp_coeff, -1, 0)
        dp = np.poly1d(dp_coeff)
        return dp, dp_coeff

    # get first- and second-order derivatives
    dp, dp_coeff = poly_deriv(p_coeff)
    ddp, ddp_coeff = poly_deriv(dp_coeff)
    # solve roots for first-order derivative
    dp_roots = np.roots(dp_coeff)
    ddp_at_dp_roots = ddp(dp_roots)
    min_ = -100
    if len(p_coeff) - 1 > 2:
        for i in range(len(dp_roots)):
            if abs(np.imag(dp_roots[i])) < 1.E-2:
                root = np.real(dp_roots[i])
                if 0. < root < 1 and np.real(ddp_at_dp_roots[i]) > 0.:
                    min_ = root
    #elif ddp_at_dp_roots[0] > 0:
    #    min_ = dp_roots[0]
    else:
        #print(dp_roots, ddp_at_dp_roots)
        return dp_roots[0] if ddp_at_dp_roots[0] > 0 else 1.1
    if np.allclose(min_, -100):
        return 1.1
    else:
        return min_
