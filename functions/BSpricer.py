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


class BS_pricer:
    """
    Closed Formula.
    """

    def __init__(self, S0, r, sigma,  ttm, exercise, K):
        self.S0 = S0  # current price
        self.r = r  # interest rate
        self.sigma = sigma  # diffusion coefficient
        self.K = None  # strike
        self.ttm = ttm  # maturity in years
        # self.price = 0
        self.exercise = exercise
        # self.type_o = type_o if type_o is not None else 'no_type'

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
        return self.d1_f() - self.sigma * np.sqrt(self.ttm)

    @staticmethod
    def vega(self):
        """ BS vega: derivative of the price with respect to the volatility """
        return self.S0 * np.sqrt(self.ttm) * ss.norm.pdf(self.d1_f())

    @staticmethod
    def BlackScholes(type_o='call', S0=100., K=100., ttm=1., r=0.1, sigma=0.2):
        """ Black Scholes closed formula:
            type_o: call or put.
            S0: float.    initial stock/index level.
            K: float strike price.
            ttm: float time-to-maturity (in year fractions).
            r: float constant risk-free short rate.
            sigma: volatility factor in diffusion term. """

        d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * ttm) / (sigma * np.sqrt(ttm))
        d2 = d1 - (sigma * np.sqrt(ttm))

        if type_o == "call":
            return S0 * ss.norm.cdf(d1) - K * np.exp(-r * ttm) * ss.norm.cdf(d2)
        elif type_o == "put":
            return K * np.exp(-r * ttm) * ss.norm.cdf(-d2) - S0 * ss.norm.cdf(-d1)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def closed_formula_call(self, K):
        """
          Black Scholes closed formula for call options
        """
        self.K = K
        return self.S0 * ss.norm.cdf(self.d1_f()) - (self.K * np.exp(-self.r * self.ttm) * ss.norm.cdf(self.d2_f()))

    def closed_formula_put(self, K):
        """
          Black Scholes closed formula for put options
        """
        self.K = K
        return self.K * np.exp(-self.r * self.ttm) * ss.norm.cdf(- self.d2_f()) - self.S0 * ss.norm.cdf(-self.d1_f())
