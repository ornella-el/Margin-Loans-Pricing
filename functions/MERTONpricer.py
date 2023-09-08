import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import math
from scipy.integrate import quad
import matplotlib.pyplot as plt
from BSpricer import BS_Pricer
from math import factorial

# %%%%%%%%%%%%%%%%%%%%%%%       Merton Jump Diffusion Model       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
""" S = (mu-0.5*sigma^2)*t + sigma*W(t) + sum_{i=1}^{N(t)} J_i

    S: stock price at time t
    mu: expected return of the stock --> r
    sigma: volatility of the stock returns
    t: time to maturity
    W(t): standard brownian Motion
    N(t): Poisson process with parameter lambda
    lambda: intensity of jumps during the unit time
    J_i: iid random variables from a lognormal dist. with parameters delta (mean) and epsilon (std)
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
    V_BS = Value of option using Black-Scholes Formula
    V_MJD = Value of option using Merton Jump Diffusion Model
    V_MJD = SUM_k=0^inf (exp(-mλT)(mλT)^k) / k! * V_BS(S,K,T,r_n,σ_k)

"""


class Merton_pricer():

    def __init__(self, S0, K, ttm, r, q, sigma, lambd, meanJ, stdJ, exercise):
        self.S0 = S0  # current STOCK price
        self.K = None  # strike
        self.ttm = ttm  # maturity in years
        self.q = 0
        self.r = r  # interest rate
        self.sigma = sigma  # σ: diffusion coefficient (annual volatility)
        self.lambd = lambd  # λ: Num of jumps per year
        self.meanJ = meanJ  # m: Mean of jump size
        self.stdJ = stdJ  # v: St. dev. of jump size
        self.exercise = None

    def MertonPath(self, days, N):
        dt = self.ttm / days
        size = (days, N)
        SMerton = np.zeros(size)
        SMerton[0] = self.S0
        for t in range(1, days):
            mean = np.exp(self.meanJ + self.stdJ ** 2 / 2)
            Z = np.random.normal(size=(N,))  # Brownian motion, diffusion component

            Nj = np.random.poisson(lam=self.lambd * dt, size=(N,))
            # J = np.random.normal(self.meanJ - self.lambd * self.stdJ ** 2 / 2, self.stdJ, size=(N,))
            J = np.random.normal(self.meanJ, self.stdJ, size=(N,))
            jump_component = J * Nj
            # jump_component = (np.exp(J) - 1) * Nj
            drift_component = (self.r - self.lambd * (
                        mean - 1) - 0.5 * self.sigma ** 2) * dt  # added risk-neutral adjustment

            diffusion_component = self.sigma * np.sqrt(dt) * Z
            # New prices computation
            SMerton[t] = SMerton[t - 1] * np.exp(drift_component + diffusion_component + jump_component)
        return SMerton

    # plot the price paths
    def plotMertonPath(self, SMerton, symbol, ax=None):
        if ax is None:
            ax = plt.gca()
        plt.figure(figsize=(10, 6))
        ax.plot(SMerton)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Price')
        ax.set_title(f'Merton Jump diffusion Price Paths for {symbol}')
        return

    # plot the distribution of prices average
    def plotMertonDist(self, SMerton, symbol, ax=None):
        if ax is None:
            ax = plt.gca()
        avg_path = np.mean(SMerton, axis=0)
        ax.hist(avg_path, bins='auto')
        ax.set_xlabel('price')
        ax.set_ylabel('frequency')
        ax.set_title(f'Merton Jump diffusion: Distribution of {symbol} prices')
        return

    def plotMertonLogReturns(self, SMerton, symbol, ax=None):
        if ax is None:
            ax = plt.gca()
        log_returns = np.log(SMerton[1:] / SMerton[:-1])
        log_returns = log_returns.flatten()
        sns.kdeplot(log_returns, label='Log Returns', ax=ax)
        ax.set_xlabel('Log Return')
        ax.set_ylabel('Density')
        ax.set_title('Merton')
        ax.legend()
        return

    @staticmethod
    def plotMertonAtFixedTime(SMerton, time, symbol, ax):
        if ax is None:
            ax = plt.gca()
        fixed_values = SMerton[time, :]

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
        ax.set_title(f'Merton Price at T ={time + 1}')
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
        K1 = K1
        K2 = K2
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

    def r_k(self, k):
        return self.r - self.lambd * (self.meanJ - 1) + (k * np.log(self.meanJ)) / self.ttm

    def sigma_k(self, k):
        return np.sqrt(self.sigma ** 2 + (k * self.stdJ ** 2) / self.ttm)

    # https://www.codearmo.com/python-tutorial/merton-jump-diffusion-model-python
    def closed_formula_call(self, K):
        self.K = K
        V = 0
        mean = np.exp(self.meanJ + self.stdJ ** 2 / 2)
        for k in range(40):
            r_k = self.r - self.lambd * (mean - 1) + (k * np.log(mean)) / self.ttm
            sigma_k = np.sqrt(self.sigma ** 2 + (k * self.stdJ ** 2) / self.ttm)
            k_fact = factorial(k)
            V += (np.exp(-mean * self.lambd * self.ttm) * np.power(mean * self.lambd * self.ttm, k)) / k_fact * \
                 BS_Pricer.BlackScholes(type_o='call', S0=self.S0, K=self.K, ttm=self.ttm,
                                        r=r_k, q=self.q, sigma=sigma_k)
        return V

    def closed_formula_put(self, K):
        self.K = K
        V = 0
        for k in range(40):
            mean = np.exp(self.meanJ + self.stdJ ** 2 / 2)
            r_k = self.r - self.lambd * (mean - 1) + (k * np.log(mean)) / self.ttm
            sigma_k = np.sqrt(self.sigma ** 2 + (k * self.stdJ ** 2) / self.ttm)
            k_fact = factorial(k)
            V += (np.exp(-mean * self.lambd * self.ttm) * np.power(mean * self.lambd * self.ttm, k)) / k_fact * \
                 BS_Pricer.BlackScholes(type_o='put', S0=self.S0, K=self.K, ttm=self.ttm,
                                        r=r_k, q=self.q, sigma=sigma_k)
        return V

    # QUELLA DI DANILO
    def closed_formula_otko(self, K1, K2):
        tol = 1e-6
        phi1 = ss.norm.cdf(np.log(K1), self.meanJ, self.stdJ)
        phi2 = ss.norm.cdf(np.log(K2 + tol), self.meanJ, self.stdJ)
        phi4 = ss.norm.cdf(np.log(K1) - self.stdJ ** 2, self.meanJ, self.stdJ)
        phi5 = ss.norm.cdf(np.log(K2 + tol) - self.stdJ ** 2, self.meanJ, self.stdJ)
        den = self.r + self.lambd * phi1
        num = (1 - np.exp(-self.ttm * den))
        Int = self.lambd * (K1 * phi1 - K2 * phi2 - (np.exp(self.meanJ + self.stdJ ** 2 / 2) * (phi4 - phi5)))
        return Int * num / den * 100

    def compute_integral(self, K1, K2):
        def integrand(x):
            return (K2 - np.exp(x)) * self.lambd * ss.norm.pdf(x, self.meanJ, self.stdJ)

        if K2 != 0:
            result = quad(integrand, np.log(K2), np.log(K1))
            return result[0]
        result = quad(integrand, 0, np.log(K1))
        return result[0]

    # quella mia integrata
    def closed_formula_otko2(self, K1, K2):

        # for me K1 = high strike, K2 = low strike
        # phi1 = phi3 so I'll use phi1
        phi1 = ss.norm.cdf(np.log(K1), self.meanJ, self.stdJ)
        if K2 != 0:
            I_go = (K1 - K2) * self.lambd * ss.norm.cdf(np.log(K2), self.meanJ, self.stdJ) + self.compute_integral(K1,
                                                                                                                   K2)
        else:
            I_go = self.compute_integral(K1, K2)
        num = (1 - np.exp(-self.ttm * (self.r + self.lambd * phi1)))
        den = self.r + self.lambd * phi1
        # print(I_go, num, den)

        return I_go * num / den

    def pdf_mjd(self, x, days):
        dt = self.ttm / days
        # Risk-free paramaters
        mean = np.exp(self.meanJ + self.stdJ ** 2 / 2)
        mu = self.r - self.lambd*(mean - 1)
        sum = 0

        for k in range(171):  # Using 171 to approximate the loop in R
            numerator = (self.lambd * dt) ** k * np.exp(-((x - mu * dt - k * self.meanJ) ** 2) / (2 * (self.sigma ** 2 * dt + k * self.stdJ ** 2)))
            denominator = math.factorial(k) * np.sqrt(2 * np.pi * (self.sigma ** 2 * dt + k * self.stdJ ** 2))
            sum += numerator / denominator

        return sum * np.exp(-self.lambd * dt)

        # phi2 = ss.norm.cdf(np.log(K2), self.meanJ, self.stdJ)
        # phi4 = ss.norm.cdf((np.log(K1) - self.meanJ) / self.stdJ - self.stdJ)
        # phi5 = ss.norm.cdf((np.log(K2) - self.meanJ) / self.stdJ - self.stdJ)
        # I_go = self.lambd * (K1 * phi1 - K2 * phi2 - np.exp(self.meanJ + self.stdJ ** 2 / 2) * (phi4 - phi5))

# REFERENCES: https://www.codearmo.com/python-tutorial/merton-jump-diffusion-model-python
#             https://github.com/Robin-Guilliou/Option-Pricing/blob/main/European%20Options/Merton%20Jump%20Diffusion/merton_jump_analytical.ipynb
#             https://github.com/cantaro86/Financial-Models-Numerical-Methods/blob/master/functions/Merton_pricer.py
#             https://github.com/QGoGithub/Merton-Jump-Diffusion-CPP/blob/master/MJDclosed.cpp
# FROM HULL BOOK: OPTIONS DERIVATIVES AND....
