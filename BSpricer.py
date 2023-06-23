import numpy as np
import pandas as pd
import scipy.stats as ss

""" 
PARAMETERS
S0: the current price of the underlying asset
r: the risk-free interest rate
sigma: the diffusion coefficient
K: the strike price of the option
ttm: time-to maturity
exercise: 
type: 'call' or 'put'
????option_price: the call (or put) price given by the column 'Last Price' in df 
"""


class BS_pricer():
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference Black-Scholes PDE:
     df/dt + r df/dx + 1/2 sigma^2 d^f/dx^2 -rf = 0
    """

    def __init__(self, S0, r, sigma, K, ttm, exercise, type_o):
        self.S0 = S0  # current price
        self.r = r  # interest rate
        self.sigma = sigma  # diffusion coefficient
        self.K = K  # strike
        self.ttm = ttm  # maturity in years
        # self.price = 0
        self.exercise = exercise
        self.type_o = type_o

    def payoff_f(self, St):
        if self.type_o == 'call':
            payoff = np.maximum(St - self.K, 0)
        elif self.type_o == 'put':
            payoff = np.maximum(self.K - St, 0)
        else:
            raise ValueError('Please select "call" or "put" type.')
        return payoff

    def d1_f(self):
        return (np.log(self.S0 / self.K) + ((self.r + 0.5 * self.sigma ** 2) * self.ttm)) / (
                self.sigma * np.sqrt(self.ttm) + 1e-6)

    def d2_f(self):
        return self.d1_f - self.sigma * np.sqrt(self.ttm)

    def vega(self):
        """ BS vega: derivative of the price with respect to the volatility """
        return self.S0 * np.sqrt(self.ttm) * ss.norm.pdf(self.d1_f)

    def closed_formula(self):
        """
        Black Scholes closed formula:
        """
        if self.type_o == "call":
            return self.S0 * ss.norm.cdf(self.d1_f) - self.K * np.exp(-self.r * self.ttm) * ss.norm.cdf(self.d2_f)
        elif self.type_o == "put":
            return self.K * np.exp(-self.r * self.ttm) * ss.norm.cdf(-self.d2_f) - self.S0 * ss.norm.cdf(- self.d1_f)
        else:
            raise ValueError('Please select "call" or "put" type.')
