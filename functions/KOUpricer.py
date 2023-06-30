import numpy as np
import pandas as pd
import scipy.stats as scs
import scipy.special as ssp
import math


class Kou_pricer():
    """
        Closed Formula.
        V_KJD_1 = S(0) * Yfucntion(r + 0.5*sigma^2 - lambda*zeta, sigma, lambda', p', eta1', eta2'; log(K/S(0)), T)
        V_KJD_2 = K exp(-rT) * Yfucntion(r - 0.5*sigma^2 - lambda*zeta, sigma, lambda, p, eta1, eta2; log(K/S(0)), T)

        V_KJD_call =      V_KJD_1 - V_KJD_2

        V_KJD_put = V_KJD_call + (Kexp(-rT) - S(0))   put-call parity
    """

    def __init__(self, S0, K, ttm, r, sigma, lambd, p, eta1, eta2, exercise):
        self.S0 = S0  # current STOCK price
        self.K = K  # strike
        self.T = ttm  # maturity in years

        self.r = r  # interest rate
        self.sigma = sigma  # σ: diffusion coefficient (annual volatility)
        self.lambd = lambd  # λ: Num of jumps per year
        self.p = p  # p: probability of upward jumps
        self.q = 1 - self.p  # q: probability of downward jumps
        self.eta1 = eta1  # η_1: rate of exponential r.v. ξ_1 (1/η_1 mean)
        self.eta2 = eta2  # η_2: rate of exponential r.v. ξ_2 (1/η_2 mean)

        self.exercise = exercise

    def payoff_f(self, St, type_o):
        if type_o == 'call':
            payoff = np.maximum(St - self.K, 0)
        elif type_o == 'put':
            payoff = np.maximum(self.K - St, 0)
        else:
            raise ValueError('Please select "call" or "put" type.')
        return payoff

    # Three term recursion (Abramowitz and Stegun 1972)
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

# REFERENCES: S. G. Kou, (2002) A Jump-Diffusion Model for Option Pricing. Management Science 48(8):1086-1101.
