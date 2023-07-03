import mpmath as mp
import numpy as np
import math
import scipy.special as ssp
import scipy.stats as ss
from scipy.integrate import quad
from functools import partial
from functions.FFT import fft_Lewis
from functions.CFs import cf_VG


class VG_pricer():
    """
        Closed Formula.
        Call(S0, K, T) = S0 exp(-rt)* psi(a1, b1, gamma) - k exp(-rt)*psi(a2, b2, gamma)
        psi(a1, b1, gamma): defined in terms of K_v(z) and Phi(γ,1-γ,1+γ; (1+u)/2, -sign(a) c(1+u))
        K_v(z): modified bessel function of the second kind
        Φ(γ,1-γ,1+γ; (1+u)/2, -sign(a) c(1+u)): integral representation Humbert
    """

    def __init__(self, S0, K, ttm, r, q, sigma, theta, nu, exercise):
        self.S0 = S0  # current STOCK price
        self.K = K  # strike
        self.ttm = ttm  # maturity in years
        self.r = r  # interest rate
        self.q = q  # dividend yield
        self.sigma = sigma  # σ: diffusion coefficient (annual volatility)
        self.theta = theta  # θ: Drift of gamma process
        self.nu = nu  # ν: variance of gamma process
        self.exercise = exercise

    def payoff_f(self, St, type_o):
        if type_o == 'call':
            payoff = np.maximum(St - self.K, 0)
        elif type_o == 'put':
            payoff = np.maximum(self.K - St, 0)
        else:
            raise ValueError('Please select "call" or "put" type.')
        return payoff

    def gamma(self):
        return self.ttm / self.nu

    def omega(self):
        return self.nu / np.log(1 - self.theta * self.nu - (self.sigma ** 2 * self.nu) / 2)

    def zeta(self, K):
        self.K = K
        return (np.log(self.S0 / self.K) + self.omega() * self.ttm) / self.sigma

    def upsilon(self):
        return 1 - self.nu * (self.theta + 0.5 * self.sigma ** 2)

    def a1(self):
        return self.zeta(self.K) * np.sqrt(self.upsilon() / self.nu)

    def a2(self):
        return self.a1() / np.sqrt(self.upsilon())

    def b1(self):
        return (self.theta + self.sigma ** 2) * np.sqrt(self.nu / self.upsilon()) / self.sigma

    def b2(self):
        return self.theta * np.sqrt(self.nu) / self.sigma

    def psi(self, a, u, c, alpha, beta, gamma):
        # hyper2d({'m+n':[a], 'm':[b], 'n':[c]}, {'m+n':[d]}, x, y)
        x = (1 + u) / 2
        y = - np.sign(a) * c * (1 + u)
        result = mp.hyper2d({'m+n': [alpha], 'm': [beta]}, {'m+n': [gamma]}, x, y)
        return result

    def bessel(self, c, x):
        return ssp.kv(x, c)

    def phi(self, a, b, gamma):
        u = b / np.sqrt(2 + b ** 2)
        c = abs(a) * np.sqrt(2 + b ** 2)

        res1 = np.power(c, gamma + 0.5) * np.exp(np.sign(a) * c) * np.power(1 + u, gamma) / (
                np.sqrt(2 * np.pi) * math.gamma(gamma) * gamma) \
               * self.bessel(c, gamma + 0.5) * self.psi(a, u, c, gamma, 1 - gamma, 1 + gamma)

        res2 = - np.sign(a) * np.power(c, gamma + 0.5) * np.exp(np.sign(a) * c) * np.power(1 + u, 1 + gamma) / (
                np.sqrt(2 * np.pi) * math.gamma(gamma) * (1 + gamma)) \
               * self.bessel(c, gamma - 0.5) * self.psi(a, u, c, 1 + gamma, 1 - gamma, 2 + gamma)

        res3 = + np.sign(a) * np.power(c, gamma + 0.5) * np.exp(np.sign(a) * c) * np.power(1 + u, gamma) / (
                np.sqrt(2 * np.pi) * math.gamma(gamma) * gamma) \
               * self.bessel(c, gamma - 0.5) * self.psi(a, u, c, gamma, 1 - gamma, 1 + gamma)

        return res1 + res2 + res3


    def closed_formula_call(self, K):
        """
        VG closed formula for call options.  Put is obtained by put/call parity.
        """

        def Psy(a, b, g):
            f = lambda u: ss.norm.cdf(a / np.sqrt(u) + b * np.sqrt(u)) * np.exp((g - 1) * np.log(u)) * np.exp(
                -u) / ssp.gamma(g)
            result = quad(f, 0, np.inf)
            return result[0]

        self.K = K

        # Ugly parameters
        xi = - self.theta / self.sigma ** 2
        s = self.sigma / np.sqrt(1 + ((self.theta / self.sigma) ** 2) * (self.nu / 2))
        alpha = xi * s

        c1 = self.nu / 2 * (alpha + s) ** 2
        c2 = self.nu / 2 * alpha ** 2
        d = 1 / s * (np.log(self.S0 / self.K) + self.r * self.ttm + self.ttm / self.nu * np.log((1 - c1) / (1 - c2)))

        # Closed formula
        call = self.S0 * Psy(d * np.sqrt((1 - c1) / self.nu), (alpha + s) * np.sqrt(self.nu / (1 - c1)),
                             self.ttm / self.nu) - self.K * np.exp(-self.r * self.ttm) * \
               Psy(d * np.sqrt((1 - c2) / self.nu), alpha * np.sqrt(self.nu / (1 - c2)), self.ttm / self.nu)

        return call

    def closed_formula_put(self, K):
        self.K = K
        return self.closed_formula_call(K) - self.S0 + self.K * np.exp(-self.r * self.ttm)

    def closed_formula_call2(self, K):
        self.K = K
        mp.dps = 30
        call = self.S0 * np.exp(-self.r * self.ttm) * self.phi(self.a1(), self.b1(), self.gamma()) \
               - self.K * np.exp(-self.r * self.ttm) * self.phi(self.a2(), self.b2(), self.gamma())
        return call

    # Via the put-call parity
    def closed_formula_put2(self, K):
        self.K = K
        return self.closed_formula_call2(K) - self.S0 + self.K * np.exp(-self.r * self.ttm)

    def FFT_call(self, K):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        K = np.array(K)
        u = np.linspace(0, self.ttm, 10000)
        cf_VG_b = partial(cf_VG(u, t=self.ttm, mu=(self.r - self.omega()), theta=self.theta, sigma=self.sigma, nu=self.nu))
        return fft_Lewis(K, self.S0, self.r, self.ttm, cf_VG_b, interp="cubic")

    def FFT_put(self, K):
        return self.FFT_call(K) - self.S0 + K * np.exp(-self.r * self.ttm)

    def closed_formula_call3(self, K):
        """
        VG closed formula for call options.  Put is obtained by put/call parity.
        """

        def Psy(a, b, g):
            limit = 100
            step = 0.001
            u_values = np.arange(0, limit, step)

            integrand_values = [ss.norm.cdf(a / (np.sqrt(u)+1e-10) + b * np.sqrt(u)) * u ** (g - 1) * np.exp(-u) / ssp.gamma(g)
                                for u in u_values]

            result = np.trapz(integrand_values, u_values)
            return result

        self.K = K

        # Ugly parameters
        xi = - self.theta / self.sigma ** 2
        s = self.sigma / np.sqrt(1 + ((self.theta / self.sigma) ** 2) * (self.nu / 2))
        alpha = xi * s

        c1 = self.nu / 2 * (alpha + s) ** 2
        c2 = self.nu / 2 * alpha ** 2
        d = 1 / s * (np.log(self.S0 / self.K) + self.r * self.ttm + self.ttm / self.nu * np.log((1 - c1) / (1 - c2)))

        # Closed formula
        call = self.S0 * Psy(d * np.sqrt((1 - c1) / self.nu), (alpha + s) * np.sqrt(self.nu / (1 - c1)),
                             self.ttm / self.nu) - self.K * np.exp(-self.r * self.ttm) * \
               Psy(d * np.sqrt((1 - c2) / self.nu), alpha * np.sqrt(self.nu / (1 - c2)), self.ttm / self.nu)

        return call

    def closed_formula_put3(self, K):
        self.K = K
        return self.closed_formula_call3(K) - self.S0 + self.K * np.exp(-self.r * self.ttm)

