import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt

# %%%%%%%%%%%%%%%%%%%%%%%       Black and Scholes Model       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""S_t = S_0 exp((r-0.5*sigma^2)*t + sigma*sqr(t)Z*(t)
        
        S0: the current price of the underlying asset
        S_t: the spot price of the underlying asset at time t
        q: dividend yield
        r: the risk-free interest rate
        sigma: the implied volatility of returns of the underlying asset (diffusion coefficient)
        t: current time in years        
"""

# %%%%%%%%%%%%%%%%%%%%%%%           Option pricing           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
""" 
K: the strike price of the option
C0: the price of a European call option at time t = 0
ttm: time-to maturity (expiry date)
exercise: european or american
type: 'call' or 'put' 
N: number of simulated paths
"""


class BS_Pricer:

    def __init__(self, S0, r, q, sigma, ttm, exercise, K):
        self.S0 = S0  # current price
        self.r = r  # interest rate
        self.sigma = sigma  # diffusion coefficient
        self.ttm = ttm  # maturity in years
        self.q = 0  # dividend yield
        # self.price = 0
        self.exercise = None
        self.K = None  # strike
        # self.type_o = type_o if type_o is not None else 'no_type'

    def BlackScholesPath(self, days, N):
        S = np.zeros((days, N))
        S[0] = self.S0
        dt = self.ttm / days
        for t in range(1, days):
            Z = np.random.normal(size=N)
            S[t] = S[t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z)
        return S

    # plot the price paths
    def plotBSPath(self, Spaths, symbol, ax=None):
        if ax is None:
            ax = plt.gca()
        plt.figure(figsize=(10, 6))
        ax.plot(Spaths)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Price')
        ax.set_title(f'B&S Model Price Paths for {symbol}')
        return

    # plot the distribution of prices average
    def plotBSDist(self, Spaths, symbol, ax=None):
        if ax is None:
            ax = plt.gca()
        avg_path = np.mean(Spaths, axis=0)
        ax.hist(avg_path, bins='auto')
        ax.set_xlabel('price')
        ax.set_ylabel('frequency')
        ax.set_title(f'B&S model: Lognormal distribution of {symbol} prices')
        return

    def plotBSLogReturns(self, Spaths, symbol, ax=None):
        if ax is None:
            ax = plt.gca()
        log_returns = np.log(Spaths[1:] / Spaths[:-1])
        log_returns = log_returns.flatten()
        sns.kdeplot(log_returns, label='Log Returns', ax=ax)
        ax.set_xlabel('Log Return')
        ax.set_ylabel('Density')
        ax.set_title(f'Black and Scholes')
        ax.legend()
        return

    def plotBSAtFixedTime(self, SBlackScholes, time, symbol, ax):
        if ax is None:
            ax = plt.gca()
        fixed_values = SBlackScholes[time, :]

        # Plotting the histogram
        hist, bins, _ = ax.hist(fixed_values, bins=30, density=True, alpha=0.7, label='Histogram')

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
        ax.set_title(f'Black-Scholes Price at T ={time + 1}')
        ax.grid(True)
        ax.legend()
        return

    def payoff_vanilla(self, St, type_o):
        """
        Payoff of the plain vanilla options: european put and call
        """
        if type_o == 'call':
            return self.payoff_call(St)
        elif type_o == 'put':
            return self.payoff_put(St)
        else:
            raise ValueError('Please select "call" or "put" type.')

    def payoff_call(self, St):
        return np.maximum(St - self.K, 0)

    def payoff_put(self, St):
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
        K1 = K1
        K2 = K2
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


    def d1_f(self):
        return (np.log(self.S0 / self.K) + ((self.r + 0.5 * self.sigma ** 2) * self.ttm)) / (
                self.sigma * np.sqrt(self.ttm))

    def d2_f(self):
        return (np.log(self.S0 / self.K) + ((self.r - 0.5 * self.sigma ** 2) * self.ttm)) / (
                self.sigma * np.sqrt(self.ttm))

    @staticmethod
    def vega(S0, K, r, q, sigma, t):
        """ BS vega: derivative of the price with respect to the volatility """
        d1 = (np.log(S0 / K) + (r - q + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
        return S0 * np.sqrt(t) * ss.norm.pdf(d1)

    @staticmethod
    def BlackScholes(type_o, S0, K, ttm, r, q, sigma):
        """ Black Scholes closed formula:
            type_o: call or put.
            S0: float.    initial stock/index level.
            K: float strike price.
            ttm: float time-to-maturity (in year fractions).
            r: float constant risk-free short rate.
            sigma: volatility factor in diffusion term. """

        d1 = (np.log(S0 / K) + (r - q + sigma ** 2 / 2) * ttm) / (sigma * np.sqrt(ttm))
        d2 = (np.log(S0 / K) + (r - q - sigma ** 2 / 2) * ttm) / (sigma * np.sqrt(ttm))

        if type_o == "call":
            return S0 * np.exp(-q * ttm) * ss.norm.cdf(d1) - K * np.exp(-r * ttm) * ss.norm.cdf(d2)
        elif type_o == "put":
            return K * np.exp(-r * ttm) * ss.norm.cdf(-d2) - S0 * np.exp(-q * ttm) * ss.norm.cdf(-d1)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def closed_formula_call(self, K):
        """
          Black Scholes closed formula for call options
        """
        self.K = K
        d1 = (np.log(self.S0 / self.K) + (self.r - self.q + self.sigma ** 2 / 2) * self.ttm) / (
                self.sigma * np.sqrt(self.ttm))
        d2 = (np.log(self.S0 / self.K) + (self.r - self.q - self.sigma ** 2 / 2) * self.ttm) / (
                self.sigma * np.sqrt(self.ttm))

        return self.S0 * np.exp(-self.q * self.ttm) * ss.norm.cdf(d1) - (
                self.K * np.exp(-self.r * self.ttm) * ss.norm.cdf(d2))

    def closed_formula_put(self, K):
        """
          Black Scholes closed formula for put options
        """
        self.K = K
        d1 = (np.log(self.S0 / self.K) + (self.r - self.q + self.sigma ** 2 / 2) * self.ttm) / (self.sigma * np.sqrt(self.ttm))
        d2 = (np.log(self.S0 / self.K) + (self.r - self.q - self.sigma ** 2 / 2) * self.ttm) / (self.sigma * np.sqrt(self.ttm))

        return self.K * np.exp(-self.r * self.ttm) * ss.norm.cdf(-d2) - self.S0 * np.exp(
            -self.q * self.ttm) * ss.norm.cdf(-d1)

    def MonteCarlo_Call(self, K, time, days, N):
        self.K = K
        payoffs = []
        SBlackScholes = self.BlackScholesPath(days, N)
        paths_at_t = SBlackScholes[time, :]
        for i in range(len(paths_at_t)):
            payoffs = self.payoff_call(paths_at_t[i])
        return np.mean(payoffs) * np.exp(-self.r * self.ttm)

    def MonteCarlo_Put(self, K, time, days, N):
        self.K = K
        payoffs = []
        SBlackScholes = self.BlackScholesPath(days, N)
        paths_at_t = SBlackScholes[time, :]
        for i in range(len(paths_at_t)):
            payoffs = self.payoff_put(paths_at_t[i])  # requires paths at a specific time t
        return np.mean(payoffs) * np.exp(-self.r * self.ttm)
