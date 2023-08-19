import numpy as np


def cf_VG(u, t=1, mu=0, theta=-0.1, sigma=0.2, nu=0.1):
    """
    Characteristic function of a Variance Gamma random variable at time t
    mu: additional drift
    theta: Brownian motion drift
    sigma: Brownian motion diffusion
    kappa: Gamma process variance
    """
    return np.exp(t * (1j * mu * u - np.log(1 - 1j * theta * nu * u + 0.5 * nu * sigma ** 2 * u ** 2) / nu))
