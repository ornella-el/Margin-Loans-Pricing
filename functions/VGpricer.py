import mpmath as mp
import numpy as np
import math
import scipy.special as ssp
import scipy.stats as ss
from scipy.integrate import quad
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
from functions.FFT import fft_Lewis
from functions.CFs import cf_VG

# %%%%%%%%%%%%%%%%%%%%%%%       Variance Gamma Model       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
    S[t] = S[t-1] * exp((r + omega)t + theta*G(t) + sigma* sqrt(G(t))*Z(t)
    S[t]: the stock price at time t
    S[t-1]: the stock price at the previous time step
    r: risk-free rate
    sigma: volatility of the log-returns
    dt: is the time step size (ttm / days)
    G[t]: gamma process with shape parameter theta*dt and scale parameter 1
    Z: standard normal random variable
    theta: the drift of the gamma process
    nu: the variance of the gamma process
"""

# %%%%%%%%%%%%%%%%%%%%%%%           Option pricing           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
""" 
K: the strike price of the option
C0: the price of a European call option at time t = 0
ttm: time-to maturity (expiry date)
exercise: european or american
type: 'call' or 'put' 
N: number of simulated paths

    Closed Formula (CALL AND PUT):
        Call(S0, K, T) = S0 exp(-rt)* psi(a1, b1, gamma) - k exp(-rt)*psi(a2, b2, gamma)
        psi(a1, b1, gamma): defined in terms of K_v(z) and Phi(γ,1-γ,1+γ; (1+u)/2, -sign(a) c(1+u))
        K_v(z): modified bessel function of the second kind
        Φ(γ,1-γ,1+γ; (1+u)/2, -sign(a) c(1+u)): integral representation Humbert

"""


class VG_pricer():

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

        # PARAMETERS OF THE 2ND REPRESENTATION (DIFF OF GAMMAS)
        self.mu_p = 0.5 * np.sqrt(self.theta ** 2 + (2 * self.sigma ** 2 / self.nu)) + 0.5 * self.theta  # positive jump mean
        self.mu_n = 0.5 * np.sqrt(
            self.theta ** 2 + (2 * self.sigma ** 2 / self.nu)) - 0.5 * self.theta  # negative jump mean
        self.nu_p = self.mu_p ** 2 * self.nu  # positive jump variamce
        self.nu_n = self.mu_n ** 2 * self.nu  # negative jump variance

    def VarianceGammaPath1(self, days, N):
        dt = self.ttm / days
        size = (days, N)
        SVarGamma = np.zeros(size)
        SVarGamma[0] = self.S0
        omega = np.log(1 - self.theta * self.nu - 0.5 * self.nu * self.sigma ** 2) / self.nu
        for t in range(1, days):
            Z = np.random.normal(size=(N,))
            # U = np.random.uniform(0, 1, size=(N,))
            deltaG = np.random.gamma(shape=dt / self.nu, scale=self.nu, size=(N,))
            h = self.theta * deltaG + self.sigma * np.sqrt(deltaG) * Z
            # h = theta * G + sigma * np.sqrt(G) * norm.ppf(U)
            # SVarGamma[t] = SVarGamma[t - 1] * np.exp((r - 0.5 * sigma ** 2 + omega) * dt + h * dt)
            SVarGamma[t] = SVarGamma[t - 1] * np.exp((self.r + omega) * dt + h)

        return SVarGamma

    # Simulate variance gamma as the difference of two gammas
    def VarianceGammaPath2(self, days, N):
        dt = self.ttm / days
        size = (days, N)
        SVarGamma = np.zeros(size)
        SVarGamma[0] = self.S0
        for t in range(1, days):
            omega = (np.log(1 - self.theta * self.nu - 0.5 * self.nu * self.sigma ** 2)) / self.nu
            Gamma_p = np.random.gamma(dt / self.nu, self.mu_p * self.nu, size=(N,))
            Gamma_n = np.random.gamma(dt / self.nu, self.mu_n * self.nu, size=(N,))
            SVarGamma[t] = SVarGamma[t - 1] * np.exp((self.r + omega) * dt + Gamma_p - Gamma_n)
        return SVarGamma

    # plot the price paths
    def plotVGPath(self, SVG, symbol, method, ax=None):
        if ax is None:
            ax = plt.gca()
        plt.figure(figsize=(8, 6))
        ax.plot(SVG)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Price')
        if method == 'Time changed BM':
            ax.set_title(f'VG PATHS for {symbol} with {method}, theta = {round(self.theta,3)}, nu = {round(self.nu,3)}')
        else:
            ax.set_title(f'VG PATHS for {symbol} with {method}, mu_p = {round(self.mu_p,3)}, mu_n = {round(self.mu_n,3)}')
        # plt.savefig(f'VG_allpaths_{method}.png')
        return

    # plot the distribution of prices average
    @staticmethod
    def plotVGDist(SVG, symbol, ax=None):
        if ax is None:
            ax = plt.gca()
        avg_path = np.mean(SVG, axis=0)
        ax.hist(avg_path, bins=30)
        ax.set_xlabel('price')
        ax.set_ylabel('frequency')
        ax.set_title(f'Variance Gamma: Distribution of {symbol} prices')
        return

    @staticmethod
    def plotVGLogReturns(SVG, symbol, ax=None):
        if ax is None:
            ax = plt.gca()
        log_returns = np.log(SVG[1:] / SVG[:-1])
        log_returns = log_returns.flatten()
        sns.kdeplot(log_returns, label='Log Returns', ax=ax)
        ax.set_xlabel('Log Return')
        ax.set_ylabel('Density')
        ax.set_title(f'Variance Gamma')
        ax.legend()
        return

    def find_moment(self, order):
        if order == 1:
            return self.theta * (1 - self.nu * self.sigma ** 2)
        elif order == 2:
            return self.theta ** 2 * (
                    1 - 2 * self.nu * self.sigma ** 2) + 2 * self.sigma ** 2 * self.theta ** 2 * self.nu ** 2
        elif order == 3:
            return 3 * self.theta ** 3 * self.nu * (
                    1 - 4 * self.sigma ** 2 * self.nu) + 6 * self.theta ** 3 * self.sigma ** 2 * self.nu ** 2
        elif order == 4:
            return 3 * self.theta ** 4 * (
                    1 - 10 * self.nu * self.sigma ** 2 + 16 * self.nu ** 2 * self.sigma ** 4) + 12 * self.theta ** 4 * self.sigma ** 2 * self.nu * (
                           1 - 4 * self.nu * self.sigma ** 2) + 24 * self.theta ** 4 * self.sigma ** 4 * self.nu ** 2
        else:
            return 0

    @staticmethod
    def plotVGAtFixedTime(SVarGamma, time, symbol, ax):
        if ax is None:
            ax = plt.gca()
        fixed_values = SVarGamma[time, :]

        # Plotting the histogram
        hist, bins, _ = ax.hist(fixed_values, bins=30, density=True, alpha=0.9, label='Histogram')

        # Plotting the KDE approx
        sns.kdeplot(fixed_values, color='r', label='Approximation', ax=ax)

        # Calculate the mean, standard deviation, and quantile
        mean_price = np.mean(fixed_values)
        std_dev = np.std(fixed_values)
        quantile_95 = np.percentile(fixed_values, 95)

        # Display the mean, standard deviation, and quantile on the plot
        ax.axvline(mean_price, color='g', linestyle='--', label='Mean Price')
        ax.axvline(mean_price + std_dev, color='b', linestyle='--', label='Mean ± Std Dev')
        ax.axvline(mean_price - std_dev, color='b', linestyle='--', label='Mean ± Std Dev')
        ax.axvline(quantile_95, color='m', linestyle='--', label='95% Quantile')

        ax.set_xlabel(f'{symbol} price after T = {time + 1} days')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Variance Gamma Price at T ={time + 1}')
        ax.grid(True)
        ax.legend()
        return

    def payoff_vanilla(self, K, St, type_o):
        """
        Payoff of the plain vanilla options: european put and call
        """
        self.K = K
        if type_o == 'call':
            return self.payoff_call(self.K, St)
        elif type_o == 'put':
            return self.payoff_put(self.K, St)
        else:
            raise ValueError('Please select "call" or "put" type.')

    def payoff_call(self, K, St):
        self.K = K
        return np.maximum(St - self.K, 0)

    def payoff_put(self, K,  St):
        self.K = K
        return np.maximum(self.K - St, 0)

    # DONE: implement the payoff
    def payoff_otko(self, path, K1, K2):
        """
        Payoff of the One Touch Knock Out Daily Cliquet Options
        K1 = 'Knock-Out' barrier: let the option expire
        k2 = 'One-Touch' barrier: let the seller receive a payoff for the downward jump
        """
        payoff = 0
        returns = path[1:] / path[:-1]
        for Rt in returns:
            if Rt > K1:
                payoff = 0
            elif K2 < Rt <= K1:
                payoff = (K1 - Rt)
                return payoff
            elif Rt <= K2:
                payoff = (K1 - K2)
                return payoff
        return payoff

    def omega(self):  # martingale correction
        return - np.log(1 - self.theta * self.nu - (self.sigma ** 2 * self.nu) / 2) / self.nu

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

    def gamma(self):
        return self.ttm / self.nu

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

    # FAST FOURIER TRANSFORM METHODS
    def Q1(self, k, cf, right_lim):
        integrand = lambda u: np.real((np.exp(-u * k * 1j) / (u * 1j)) * cf(u - 1j) / cf(-1j))
        return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, right_lim, limit=1000)[0]

    def Q2(self, k, cf, right_lim):
        integrand = lambda u: np.real(np.exp(-u * k * 1j) / (u * 1j) * cf(u))
        return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, right_lim, limit=1000)[0]

    def cf_VG(self, u, t=1, mu=0, theta=-0.1, sigma=0.2, nu=0.1):
        return np.exp(
            t * (1j * mu * u - np.log(1 - 1j * theta * nu * u + 0.5 * nu * sigma ** 2 * u ** 2) / nu))

    #  https://www.impan.pl/swiat-matematyki/notatki-z-wyklado~/tankov2.pdf
    def closed_formula_otko(self, K1, K2):  # con incomplete gamma function integral, versione MIA (integrata)
        c = 1 / self.nu
        lamd1 = (np.sqrt(
            self.theta ** 2 + 2 * self.sigma ** 2 / self.nu) / self.sigma ** 2) + self.theta / self.sigma ** 2
        phi = ssp.gammainc(0, -np.log(K1) * lamd1) * c
        num = (1 - np.exp(-self.ttm * (self.r + phi)))
        den = self.r + phi
        Int1 = (K1 - K2) * ssp.gammainc(0, -np.log(K2) * lamd1) * c
        Int2 = c * K1 * (ssp.gammainc(0, -np.log(K1) * lamd1) - ssp.gammainc(0, -np.log(K2) * lamd1)) + \
               c * (ssp.gammainc(0, -np.log(K1) * (1 + lamd1)) - ssp.gammainc(0, -np.log(K2) * (1 + lamd1)))

        print(Int1, Int2, num, den)
        return (Int1 + Int2) * num / den

    # https: // arxiv.org / pdf / 2303.05615.pdf
    def closed_formula_otko5(self, K1, K2):  # con exponential integral, versione solo MIA (integrale)
        tol = 1e-6
        c_n = self.mu_n ** 2 / self.nu_n
        lambd_n = self.mu_n / self.nu_n
        phi = -c_n * ssp.expi(lambd_n * np.log(K1))
        den = self.r + phi
        num = 1 - np.exp(-self.ttm * den)
        Int1 = c_n * K2 * ssp.expi(lambd_n * np.log(K2 + tol)) - c_n * K1 * ssp.expi(lambd_n * np.log(K1))
        Int2 = c_n * (ssp.expi((lambd_n + 1) * np.log(K1)) - ssp.expi((lambd_n + 1) * np.log(K2+tol)))

        return (Int1 + Int2) * num / den * 100

    def closed_formula_otko7(self, K1, K2):  # con exponential integral, versione solo MIA (integrale)
        tol = 1e-6
        c = 1 / self.nu
        G = 1 / (np.sqrt(self.theta ** 2 * self.nu ** 2 / 4 + self.sigma ** 2 * self.nu / 2) - self.theta * self.nu/2)
        phi = -c * ssp.expi(G * np.log(K1))
        den = self.r + phi
        num = 1 - np.exp(-self.ttm * den)
        Int1 = c * (K2 * ssp.expi(G * np.log(K2 + tol)) - K1 * ssp.expi(G * np.log(K1)))
        Int2 = c * (ssp.expi((G + 1) * np.log(K1)) - ssp.expi((G + 1) * np.log(K2 + tol)))
        return ( Int2) * num / den * 100

    def closed_formula_otko6(self, K1, K2):  # con exponential integral, versione solo MIA (integrale)
        tol = 1e-6
        c = 1 / self.nu
        G = 1 / (np.sqrt(self.theta ** 2 * self.nu ** 2 / 4 + self.sigma ** 2 * self.nu / 2) - self.theta * self.nu/2)
        phi = -c * ssp.expi(G * np.log(K1))
        den = self.r + phi
        num = 1 - np.exp(-self.ttm * den)
        Int1 = c * (K2 * ssp.expi(G * np.log(K2 + tol)) - K1 * ssp.expi(G * np.log(K1)))
        Int2 = c * (ssp.expi((G + 1) * np.log(K1)) - ssp.expi((G + 1) * np.log(K2 + tol)))
        return (Int1 + Int2) * num / den * 100

    def closed_formula_otko8(self, K1, K2):  # con exponential integral, versione solo MIA (integrale)
        tol = 1e-6
        c = 1 / self.nu
        G = 1 / (np.sqrt(self.theta ** 2 * self.nu ** 2 / 4 + self.sigma ** 2 * self.nu / 2 - self.theta * self.nu/2))
        phi = -c * ssp.expi(G * np.log(K1))
        den = self.r + phi
        num = 1 - np.exp(-self.ttm * den)
        Int = -c * (ssp.expi((G + 1) * np.log(K1)) - ssp.expi((G + 1) * np.log(K2 + tol)))
        return Int * num / den * 100

    def FFT_call(self, K):
        K = np.array(K)
        cf_VG_b = partial(cf_VG, t=self.ttm, mu=(self.r - self.omega()), theta=self.theta, sigma=self.sigma, nu=self.nu)

        call_VG = np.zeros_like(K, dtype=float)
        for i in range(len(K)):
            k = np.log(K[i] / self.S0)
            call_VG[i] = self.S0 * self.Q1(k, cf_VG_b, np.inf) - K[i] * np.exp(-self.r * self.ttm) * \
                         self.Q2(k, cf_VG_b, np.inf)
        return call_VG

    def FFT_put(self, K):
        return self.FFT_call(K) - self.S0 + K * np.exp(-self.r * self.ttm)

    # FAST FOURIER TRANSFORM USING LEWIS METHOD
    def FFT_call2(self, K):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        K = np.array(K)
        cf_VG_b = partial(cf_VG, t=self.ttm, mu=(self.r - self.omega()), theta=self.theta, sigma=self.sigma, nu=self.nu)
        return fft_Lewis(K, self.S0, self.r, self.ttm, cf_VG_b, interp="linear")

    def FFT_put2(self, K):
        return self.FFT_call2(K) - self.S0 + K * np.exp(-self.r * self.ttm)

    # NOT PROPERLY WORKING CLOSED FORMULAS
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
