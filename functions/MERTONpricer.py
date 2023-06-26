import numpy as np
import pandas as pd
import scipy.stats as ss
from functions.BSpricer import BS_pricer
from math import factorial


class Merton_pricer():
    """
        Closed Formula.
        V_BS = Value of option using Black-Scholes Formula
        V_MJD = Value of option using Merton Jump Diffusion Model

        V_MJD = SUM_k=0^inf (exp(-mλT)(mλT)^k) / k! * V_BS(S,K,T,r_n,σ_k)

    """

    def __init__(self, S0, K, ttm, r, sigma, lambd, meanJ, stdJ, exercise, type_o):
        self.S0 = S0  # current STOCK price
        self.K = None  # strike
        self.ttm = ttm  # maturity in years

        self.r = r  # interest rate
        self.sigma = sigma  # σ: diffusion coefficient (annual volatility)
        self.lambd = lambd  # λ: Num of jumps per year
        self.meanJ = meanJ  # m: Mean of jump size
        self.stdJ = stdJ  # v: St. dev. of jump size

        self.exercise = exercise
        self.type_o = type_o if type_o is not None else 'no_type'

    def payoff_f(self, St):
        if self.type_o == 'call':
            payoff = np.maximum(St - self.K, 0)
        elif self.type_o == 'put':
            payoff = np.maximum(self.K - St, 0)
        else:
            raise ValueError('Please select "call" or "put" type.')
        return payoff

    def closed_formula(self, K):
        """
        Merton closed formula.
        """
        self.K = K
        tot = 0
        for k in range(40):
            # m = self.lambd * (np.exp(self.meanJ + self.stdJ**2)/2)-1
            r_k = self.r - self.lambd*(self.meanJ-1) + (k * np.log(self.meanJ)) / self.ttm
            sigma_k = np.sqrt(self.sigma ** 2 + (k * self.stdJ ** 2) / self.ttm)

            tot += (np.exp(-self.meanJ * self.lambd * self.ttm) * (self.meanJ * self.lambd * self.ttm) ** k / factorial(
                k)) * BS_pricer.BlackScholes(self.type_o, self.S0, self.K, self.ttm, r_k, sigma_k)

        return tot

# REFERENCES: https://www.codearmo.com/python-tutorial/merton-jump-diffusion-model-python
