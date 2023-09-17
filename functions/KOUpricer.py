import numpy as np
import pandas as pd
import scipy.stats as scs
import scipy.special as ssp
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy.optimize import newton

# %%%%%%%%%%%%%%%%%%%%%%%             Kou Model            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
    S[t] = S[t - 1] * exp((mu - 0.5 * sigma ^ 2) * dt + sigma * W[t]) * prod_{i=1}^{N[t]} (Vi-1)
    dS = mu*S*dt + sigma*S*dW(t) + d(sum_{i=1}^{N(t)} (Vi-1))

    S: stock price at time t
    mu: expected return of the stock --> r
    sigma: volatility of the stock returns
    t: time to maturity
    W(t): standard brownian Motion
    N(t): Poisson process with parameter lambda
    lambda: intensity of jumps during the unit time
    V_i: iid random variables from a double exponential dist. with parameters p, q=1-p, eta1, eta2
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
        V_KJD_1 = S(0) * Yfunction(r + 0.5*sigma^2 - lambda*zeta, sigma, lambda', p', eta1', eta2'; log(K/S(0)), T)
        V_KJD_2 = K exp(-rT) * Yfunction(r - 0.5*sigma^2 - lambda*zeta, sigma, lambda, p, eta1, eta2; log(K/S(0)), T)

        V_KJD_call =      V_KJD_1 - V_KJD_2

        V_KJD_put = V_KJD_call + (K*exp(-rT) - S(0))   put-call parity

"""

# %%%%%%%%%%%%%%%%%%%%%%%           REFERENCES           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
> RISK NEUTRAL COEFF. https://core.ac.uk/download/pdf/48539543.pdf
"""
class Kou_pricer():

    def __init__(self, S0, K, ttm, r, sigma, lambd, p, eta1, eta2, exercise):
        self.S0 = S0  # current STOCK price
        self.K = None  # strike
        self.T = ttm  # maturity in years
        self.r = r  # interest rate
        self.sigma = sigma  # σ: diffusion coefficient (annual volatility)
        self.lambd = lambd  # λ: Num of jumps per year
        self.p = p  # p: probability of upward jumps
        self.q = 1 - self.p  # q: probability of downward jumps
        self.eta1 = eta1  # η_1: rate of exponential r.v. ξ_1 (1/η_1 mean)
        self.eta2 = eta2  # η_2: rate of exponential r.v. ξ_2 (1/η_2 mean)
        self.exercise = None

    def KouPath(self, days, N):
        dt = self.T / days
        size = (days, N)
        SKou = np.zeros(size)
        SKou[0] = self.S0
        for t in range(1, days):

            # find risk-neutral parameters
            zeta = self.q * self.eta2 / (self.eta2 + 1) + self.p * self.eta1 / (self.eta1 - 1) - 1

            # Random numbers generation
            Z = np.random.normal(size=(N,))
            Nj = np.random.poisson(lam=self.lambd * dt, size=(N,))

            # Generate jump sizes J
            U = np.random.uniform(0, 1, size=(N,))  # Generate uniform random variables
            J = np.zeros_like(U)  # Initialize jump sizes
            for i in range(N):
                if U[i] >= self.p:
                    J[i] = (-1/self.eta1) * np.log((1-U[i]) / self.p)
                else:
                    J[i] = (1 / self.eta2) * np.log(U[i] / self.q)

            # J[U <= self.p] = 1/self.eta2 * np.log((1 - U[U < self.p]) / self.q)   # Negative jumps
            # J[U > self.p] = (-1/self.eta1) * np.log(U[U >= self.p] / self.p)  # Positive jumps

            # J = np.where(U <= self.p, -np.log(1 - self.eta1 * U) / self.eta1, np.log(1 - self.eta2 * (U - self.p) / self.q) / self.eta2)  # Use the same jump size calculation as before
            # Step 2: Calculate J using inverse transform sampling
            # J = np.where(U <= self.p, -np.log(1 - 1/self.eta1 * U) * self.eta1, np.log(1 - 1/self.eta2 * (U - self.p) / self.q) * self.eta2)

            # Find components
            jump_component = J * Nj
            drift_component = (self.r - 0.5 * self.sigma ** 2 - self.lambd*zeta) * dt
            # drift_component = (self.r - 0.5 * self.sigma ** 2) * dt
            diffusion_component = self.sigma * np.sqrt(dt) * Z

            # New prices computation
            SKou[t] = SKou[t - 1] * np.exp(drift_component + diffusion_component + jump_component)
        return SKou

    # plot the price paths
    @staticmethod
    def plotKouPath(SKou, symbol, ax=None):
        if ax is None:
            ax = plt.gca()
        plt.figure(figsize=(10, 6))
        ax.plot(SKou)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Price')
        ax.set_title(f'Kou Jump Diffusion Price Paths for {symbol}')
        return

    @staticmethod
    def plotKouDist(SKou, symbol, ax=None):
        if ax is None:
            ax = plt.gca()
        avg_path = np.mean(SKou, axis=0)
        ax.hist(avg_path, bins='auto')
        ax.set_xlabel('price')
        ax.set_ylabel('frequency')
        ax.set_title(f'Kou Jump Diffusion: Distribution of {symbol} prices')
        return

    @staticmethod
    def plotKouLogReturns(SKou, symbol, ax=None):
        if ax is None:
            ax = plt.gca()
        log_returns = np.log(SKou[1:] / SKou[:-1])
        log_returns = log_returns.flatten()
        sns.kdeplot(log_returns, label='Log Returns', ax=ax)
        ax.set_xlabel('Log Return')
        ax.set_ylabel('Density')
        ax.set_title(f'Kou')
        ax.legend()
        return

    # plot the distribution of prices average
    @staticmethod
    def plotKouAtFixedTime(SKou, time, symbol, ax):
        if ax is None:
            ax = plt.gca()
        fixed_values = SKou[time, :]

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
        ax.set_title(f'Kou Price at T ={time + 1}')
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

    # Three term recursion (Abramowitz and Stegun 197num2)
    def Hh(self, n, x):
        if n < -1:
            return 0
        elif n == -1:
            return np.exp(-x ** 2 / 2)
        elif n == 0:
            return np.sqrt(2 * np.pi) * scs.norm.cdf(-x)
        else:
            return (self.Hh(n - 2, x) - x * self.Hh(n - 1, x)) / n

    # Pfunction from KOU 2002 (Appendix B)
    def P(self, n, k, eta1, eta2, p):
        q = 1 - p
        if k < 1 or n < 1:
            return 0
        elif n == k:
            return p ** n
        else:
            sum_p = 0
            i = k
            while i <= n - 1:
                sum_p = sum_p + ssp.binom(n - k - 1, i - k) * ssp.binom(n, i) * (eta1 / (eta1 + eta2)) \
                        ** (i - k) * (eta2 / (eta1 + eta2)) ** (n - i) * p ** i * q ** (n - i)
                i += 1
            return sum_p

    # Qfunction from KOU 2002 (Appendix B)
    def Q(self, n, k, eta1, eta2, p):
        q = 1 - p
        if k < 1 or n < 1:
            return 0
        elif n == k:
            return q ** n
        else:
            sum_q = 0
            i = k
            while i <= n - 1:
                sum_q = sum_q + ssp.binom(n - k - 1, i - k) * ssp.binom(n, i) * (eta1 / (eta1 + eta2)) \
                        ** (n - i) * (eta2 / (eta1 + eta2)) ** (i - k) * p ** (n - i) * q ** i
                i += 1
        return sum_q

    def I(self, n, c, alpha, beta, delta):
        if beta < 0 and alpha < 0:
            sum_i = 0
            i = 0
            while i <= n:
                sum_i = sum_i + (beta / alpha) ** (n - i) * self.Hh(i, beta * c - delta)
                i += 1
            return -np.exp(alpha * c) / alpha * sum_i - (beta / alpha) ** (n + 1) * (np.sqrt(2 * np.pi) / beta) \
                   * np.exp((alpha * delta / beta) + (alpha ** 2 / (2 * beta ** 2))) * scs.norm.cdf(
                beta * c - delta - alpha / beta)
        elif beta > 0 and alpha != 0:
            sum_i = 0
            i = 0
            while i <= n:
                sum_i = sum_i + (beta / alpha) ** (n - i) * self.Hh(i, beta * c - delta)
                i += 1
            return -np.exp(alpha * c) / alpha * sum_i + (beta / alpha) ** (n + 1) * (np.sqrt(2 * np.pi) / beta) \
                   * np.exp((alpha * delta / beta) + (alpha ** 2 / (2 * beta ** 2))) * scs.norm.cdf(
                -beta * c + delta + alpha / beta)
        else:
            return 0

    def Pi(self, n, lambd):
        return np.exp(-lambd * self.T) * (lambd * self.T) ** n / math.factorial(n)

    def Yfunction(self, mu, sigma, lambd, p, eta1, eta2, a, T):
        bound = 10
        sump1 = 0
        sumq1 = 0

        for n in range(1, bound + 1):
            sump1_n = 0
            sumq1_n = 0
            for k in range(1, n + 1):
                sump2_k = self.P(n, k, eta1, eta2, p) * (sigma * np.sqrt(T) * eta1) ** k * \
                          self.I(k - 1, a - mu * T, -eta1, -1 / (sigma * np.sqrt(T)), -sigma * eta1 * np.sqrt(T))

                sumq2_k = self.Q(n, k, eta1, eta2, p) * (sigma * np.sqrt(T) * eta2) ** k * \
                          self.I(k - 1, a - mu * T, eta2, 1 / (sigma * np.sqrt(T)), -sigma * eta2 * np.sqrt(T))

                sump1_n += sump2_k
                sumq1_n += sumq2_k

            sump1 += self.Pi(n, lambd) * sump1_n
            sumq1 += self.Pi(n, lambd) * sumq1_n

        Y1 = np.exp((sigma * eta1) ** 2 * T / 2) / (sigma * np.sqrt(2 * np.pi * T)) * sump1
        Y2 = np.exp((sigma * eta2) ** 2 * T / 2) / (sigma * np.sqrt(2 * np.pi * T)) * sumq1
        Y3 = self.Pi(0, lambd) * scs.norm.cdf(-(a - mu * T) / (sigma * np.sqrt(T)))

        return Y1 + Y2 + Y3

    def closed_formula_call(self, K):
        self.K = K
        zeta = self.p * self.eta1 / (self.eta1 - 1) + (self.q * self.eta2) / (self.eta2 + 1) - 1
        lambd2 = self.lambd * (zeta + 1)
        eta1_2 = self.eta1 - 1
        eta2_2 = self.eta2 + 1
        p2 = self.p / (1 + zeta) * self.eta1 / (self.eta1 - 1)
        vkjd_1 = self.S0 * self.Yfunction(self.r + 1 / 2 * self.sigma ** 2 - self.lambd * zeta, self.sigma, lambd2, p2,
                                          eta1_2, eta2_2, np.log(self.K / self.S0), self.T)
        vkjd_2 = self.K * np.exp(-self.r * self.T) * self.Yfunction(
            self.r - 1 / 2 * self.sigma ** 2 - self.lambd * zeta,
            self.sigma, self.lambd, self.p, self.eta1, self.eta2,
            np.log(self.K / self.S0), self.T)

        return vkjd_1 - vkjd_2

    def closed_formula_put(self, K):
        self.K = K
        return self.closed_formula_call(self.K) + (self.K * np.exp(-self.r * self.T) - self.S0)

    def Yfunction_old(self, mu, sigma, lambd, p, eta1, eta2, a,
                      T):  # passing values again because they are modified in vkjd_1
        bound = 10
        sump1 = np.zeros(bound)
        sumq1 = np.zeros(bound)

        for n in range(1, bound + 1):
            sump2 = np.zeros(n)
            sumq2 = np.zeros(n)
            for k in range(1, n + 1):
                sump2[k - 1] = self.P(n, k, eta1, eta2, p) * (sigma * np.sqrt(T) * eta1) ** k * \
                               self.I(k - 1, a - mu * T, -eta1, -1 / (sigma * np.sqrt(T)), -sigma * eta1 * np.sqrt(T))

                sumq2[k - 1] = self.Q(n, k, eta1, eta2, p) * (sigma * np.sqrt(T) * eta2) ** k * \
                               self.I(k - 1, a - mu * T, eta2, 1 / (sigma * np.sqrt(T)), -sigma * eta2 * np.sqrt(T))

            sump1[n - 1] = self.Pi(n, lambd) * np.sum(sump2)
            sumq1[n - 1] = self.Pi(n, lambd) * np.sum(sumq2)
            # sump1[n - 1] = self.Pi(n-1, lambd) * np.sum(sump2)
            # sumq1[n - 1] = self.Pi(n-1, lambd) * np.sum(sumq2)

        Y1 = np.exp((sigma * eta1) ** 2 * T / 2) / (sigma * np.sqrt(2 * np.pi * T)) * np.sum(sump1)
        Y2 = np.exp((sigma * eta2) ** 2 * T / 2) / (sigma * np.sqrt(2 * np.pi * T)) * np.sum(sumq1)
        Y3 = self.Pi(0, lambd) * scs.norm.cdf(-(a - mu * T) / (sigma * np.sqrt(T)))

        return Y1 + Y2 + Y3

    def closed_formula_otko(self, K1, K2):
        beta = np.log(K1)
        phi = self.lambd * self.q * np.exp(beta*self.eta2)
        den = self.r + phi
        num = (1 - np.exp(-self.T * den))
        Int = self.lambd*self.q / (1 + self.eta2) * (K1**(1+self.eta2) - K2**(1+self.eta2))
        return Int * num / den * 100

    def closed_formula_otko2(self, K1, K2):
        beta = np.log(K1)
        phi = self.lambd * self.q * np.exp(beta* self.eta2)
        den = self.r + phi
        num = (1 - np.exp(-self.T * den))
        Int = self.lambd*self.q / (1 + self.eta2) * (K1**(1 + self.eta2) - K2**(1+ self.eta2))
        return Int * num / den * 100

    # REF "Kou, 2002. A Jump-Diffusion Model for Option Pricing.pdf"
    def pdf_kjd(self, x, days):
        dt = self.T / days

        # find risk-neutral parameters
        zeta = self.q * self.eta2 / (self.eta2 + 1) + self.p * self.eta1 / (self.eta1 - 1) - 1
        mu = self.r - self.lambd*zeta

        # find the sum in {} brackets
        c1 = self.p*self.eta1*np.exp(self.sigma**2*self.eta1**2*dt/2)*np.exp(-(x-mu*dt)*self.eta1)* \
                scs.norm.cdf((x-mu*dt-self.sigma**2*self.eta1*dt)/(self.sigma*np.sqrt(dt)))
        c2 = self.q*self.eta2*np.exp(self.sigma**2*self.eta2**2*dt/2)*np.exp((x-mu*dt)*self.eta2) * \
             scs.norm.cdf(- (x - mu * dt - self.sigma ** 2 * self.eta2 * dt) / (self.sigma * np.sqrt(dt)))

        # find the KDEJD density
        g_x = (1-self.lambd*dt)/(self.sigma*np.sqrt(dt))*scs.norm.pdf((x-mu*dt)/self.sigma*np.sqrt(dt)) + self.lambd*dt*(c1+c2)
        return g_x

    # REF: https: // www.researchgate.net / publication / 5143311_Risk - Neutral_and_Actual_Default_Probabilities_with_an_Endogenous_Bankruptcy_Jump - Diffusion_Model
    def Esscher_measure(self):
        def func(h):
            return self.r - h*self.sigma**2 - self.lambd*((self.p*self.eta1/(self.eta1-h)) + (self.q*self.eta2/(self.eta2+h)) -
                                                          self.p*self.eta1/(self.eta1-1-h) - (self.q * self.eta2 / self.eta2 + 1 +h))
        return newton(func, x0=0)

# REFERENCES: S. G. Kou, (2002) A Jump-Diffusion Model for Option Pricing. Management Science 48(8):1086-1101.
