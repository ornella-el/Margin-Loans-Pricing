import numpy as np
from scipy.fftpack import ifft
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import fsolve


def fft_Lewis(K, S0, r, T, cf, interp="cubic"):
    """
    K = vector of strike
    S = spot price scalar
    cf = characteristic function
    interp can be cubic or linear
    """
    N = 2 ** 15  # FFT more efficient for N power of 2
    B = 500  # integration limit
    dx = B / N
    x = np.arange(N) * dx  # the final value B is excluded

    weight = np.arange(N)  # Simpson weights
    weight = 3 + (-1) ** (weight + 1)
    weight[0] = 1
    weight[N - 1] = 1

    dk = 2 * np.pi / B
    b = N * dk / 2
    ks = -b + dk * np.arange(N)

    integrand = np.exp(- 1j * b * np.arange(N) * dx) * cf(x - 0.5j) * 1 / (x ** 2 + 0.25) * weight * dx / 3
    integral_value = np.real(ifft(integrand) * N)

    prices = 0
    if interp == "linear":
        spline_lin = interp1d(ks, integral_value, kind='linear')
        prices = S0 - np.sqrt(S0 * K) * np.exp(-r * T) / np.pi * spline_lin(np.log(S0 / K))
    elif interp == "cubic":
        spline_cub = interp1d(ks, integral_value, kind='cubic')
        prices = S0 - np.sqrt(S0 * K) * np.exp(-r * T) / np.pi * spline_cub(np.log(S0 / K))
    return prices

