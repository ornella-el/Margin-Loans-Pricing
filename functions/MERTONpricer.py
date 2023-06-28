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
        self.exercise = exercise

    def payoff_f(self, St, type_o):
        if type_o == 'call':
            payoff = np.maximum(St - self.K, 0)
        elif type_o == 'put':
            payoff = np.maximum(self.K - St, 0)
        else:
            raise ValueError('Please select "call" or "put" type.')
        return payoff

    def r_k(self, k):
        return self.r - self.lambd * (self.meanJ - 1) + (k * np.log(self.meanJ)) / self.ttm

    def sigma_k(self, k):
        return np.sqrt(self.sigma ** 2 + (k * self.stdJ ** 2) / self.ttm)

    # https://www.codearmo.com/python-tutorial/merton-jump-diffusion-model-python
    def closed_formula_call(self, K):
        self.K = K
        V = 0
        for k in range(40):
            mean = np.exp(self.meanJ + self.stdJ ** 2 / 2)
            r_k = self.r - self.lambd * (mean - 1) + (k * np.log(mean)) / self.ttm
            sigma_k = np.sqrt(self.sigma ** 2 + (k * self.stdJ ** 2) / self.ttm)
            k_fact = factorial(k)
            V += (np.exp(-mean * self.lambd * self.ttm) * (mean * self.lambd * self.ttm) ** k) / k_fact * \
                 BS_pricer.BlackScholes(type_o='call', S0=self.S0, K=self.K, ttm=self.ttm,
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
            V += (np.exp(-mean * self.lambd * self.ttm) * (mean * self.lambd * self.ttm) ** k) / k_fact * \
                 BS_pricer.BlackScholes(type_o='put', S0=self.S0, K=self.K, ttm=self.ttm,
                                        r=r_k, q=self.q, sigma=sigma_k)
        return V



    # https://github.com/QGoGithub/Merton-Jump-Diffusion-CPP/blob/master/MJDclosed.cpp
    def closed_form_call_old4(self, K):
        self.K = K
        gamma = np.exp(self.meanJ + 0.5 * self.stdJ ** 2) - 1
        V = 0.0
        for k in range(40):
            sigma_k = np.sqrt(self.sigma ** 2 + k * self.stdJ ** 2 / self.ttm)
            r_k = self.r - self.lambd * gamma + k * np.log(1 + gamma) / self.ttm
            Vbs = BS_pricer.BlackScholes(type_o='call', S0=self.S0, K=K, ttm=self.ttm,
                                         r=r_k, q=self.q, sigma=sigma_k)
            lambd2 = self.lambd * (1 + gamma)
            prob = np.exp(lambd2 * self.ttm) * np.power(lambd2 * self.ttm, float(k)) / factorial(k)
            V += prob * Vbs

        return V

    def closed_form_put_old4(self, K):
        self.K = K
        gamma = np.exp(self.meanJ + 0.5 * self.stdJ ** 2) - 1
        V = 0.0
        for k in range(40):
            sigma_k = np.sqrt(self.sigma ** 2 + k * self.stdJ ** 2 / self.ttm)
            r_k = self.r - self.lambd * gamma + k * np.log(1 + gamma) / self.ttm
            Vbs = BS_pricer.BlackScholes(type_o='put', S0=self.S0, K=K, ttm=self.ttm,
                                         r=r_k, q=self.q, sigma=sigma_k)
            lambd2 = self.lambd * (1 + gamma)
            prob = np.exp(lambd2 * self.ttm) * np.power(lambd2 * self.ttm, float(k)) / factorial(k)
            V += prob * Vbs

        return V

    # FROM HULL BOOK: OPTIONS DERIVATIVES AND....
    def closed_formula_call_old2(self, K):
        lam_prime = self.lambd * self.meanJ
        gamma = np.log(1 + self.meanJ)
        tot = 0
        for n in range(40):
            tot += (np.exp(- lam_prime * self.ttm) * (lam_prime * self.ttm) ** n) / factorial(n) \
                   * BS_pricer.BlackScholes(type_o='call', S0=self.S0, K=K, ttm=self.ttm,
                                            r=(self.r - self.lambd * self.meanJ + n * gamma / self.ttm),
                                            sigma=np.sqrt(self.sigma ** 2 + (n * self.stdJ ** 2 / self.ttm)))
        return tot

    def closed_formula_put_old2(self, K):
        lam_prime = self.lambd * (1 + self.meanJ)
        gamma = np.log(1 + self.meanJ)
        tot = 0
        for n in range(40):
            tot += (np.exp(- lam_prime * self.ttm) * (lam_prime * self.ttm) ** n) / factorial(n) \
                   * BS_pricer.BlackScholes(type_o='put', S0=self.S0, K=K, ttm=self.ttm,
                                            r=(self.r - self.lambd * self.meanJ + n * gamma / self.ttm),
                                            sigma=(self.sigma ** 2 + (n * self.stdJ ** 2 / self.ttm)))
        return tot

    # https://github.com/cantaro86/Financial-Models-Numerical-Methods/blob/master/functions/Merton_pricer.py
    def closed_formula_call_old(self, K):
        """
        Merton closed formula for call options
        """
        self.K = K

        m = self.lambd * (np.exp(self.meanJ + (self.stdJ ** 2) / 2) - 1)  # coefficient m
        lam2 = self.lambd * np.exp(self.meanJ + (self.stdJ ** 2) / 2)
        tot = 0
        for k in range(18):
            tot += (np.exp(-lam2 * self.ttm) * (lam2 * self.ttm) ** k / factorial(k)) \
                   * BS_pricer.BlackScholes('call', self.S0, self.K, self.ttm,
                                            self.r - m + k * (self.meanJ + 0.5 * self.stdJ ** 2) / self.ttm,
                                            np.sqrt(self.sigma ** 2 + (k * self.stdJ ** 2) / self.ttm))
        return tot

    def closed_formula_put_old(self, K):
        """
        Merton closed formula for put options
        """
        self.K = K

        m = self.lambd * (np.exp(self.meanJ + (self.stdJ ** 2) / 2) - 1)  # coefficient m
        lam2 = self.lambd * np.exp(self.meanJ + (self.stdJ ** 2) / 2)
        tot = 0
        for k in range(18):
            tot += (np.exp(-lam2 * self.ttm) * (lam2 * self.ttm) ** k / factorial(k)) \
                   * BS_pricer.BlackScholes('put', self.S0, self.K, self.ttm,
                                            self.r - m + k * (self.meanJ + 0.5 * self.stdJ ** 2) / self.ttm,
                                            np.sqrt(self.sigma ** 2 + (k * self.stdJ ** 2) / self.ttm))
        return tot

    # https://github.com/Robin-Guilliou/Option-Pricing/blob/main/European%20Options/Merton%20Jump%20Diffusion/merton_jump_analytical.ipynb
    def closed_formula_call_old3(self, K):
        gamma = self.meanJ + 0.5 * self.stdJ ** 2
        V = 0
        for k in range(40):
            # parameters to plug into the BS method
            r_k = self.r - self.lambd * (np.exp(gamma) - 1) + (k * gamma / self.ttm)
            sigma_k = np.sqrt(self.sigma ** 2 + (k * self.stdJ ** 2) / self.ttm)

            sum_k = (np.exp(-(np.exp(gamma)) * self.lambd * self.ttm) * (
                    (np.exp(gamma)) * self.lambd * self.ttm) ** k / (factorial(k))) \
                    * BS_pricer.BlackScholes(type_o='call', S0=self.S0, K=K, ttm=self.ttm,
                                             r=r_k, q=self.q, sigma=sigma_k)
            V += sum_k
        return V

    def closed_formula_put_old3(self, K):
        gamma = self.meanJ + 0.5 * self.stdJ ** 2
        V = 0
        for k in range(100):
            # parameters to plug into the BS method
            r_k = self.r - self.lambd * (np.exp(gamma) - 1) + (k * gamma / self.ttm)
            sigma_k = np.sqrt(self.sigma ** 2 + (k * self.stdJ ** 2) / self.ttm)

            sum_k = (np.exp(-(np.exp(gamma)) * self.lambd * self.ttm) * (
                    (np.exp(gamma)) * self.lambd * self.ttm) ** k / (factorial(k))) \
                    * BS_pricer.BlackScholes(type_o='put', S0=self.S0, K=K, ttm=self.ttm,
                                             r=r_k, q=self.q, sigma=sigma_k)
            V += sum_k
        return V
# REFERENCES: https://www.codearmo.com/python-tutorial/merton-jump-diffusion-model-python
