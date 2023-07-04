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

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
from scipy.stats import gamma, norm


def VarianceGammaPath1(T, days, N, sigma, r, nu, theta, S0):
    dt = T / days
    size = (days, N)
    SVarGamma = np.zeros(size)
    SVarGamma[0] = S0
    for t in range(1, days):
        Z = np.random.normal(size=(N,))
        # U = np.random.uniform(0, 1, size=(N,))
        deltaG = np.random.gamma(shape=dt / nu, scale=nu, size=(N,))
        h = theta * deltaG + sigma * np.sqrt(deltaG) * Z
        # h = theta * G + sigma * np.sqrt(G) * norm.ppf(U)
        omega = (np.log(1 - nu * theta - 0.5 * nu * pow(sigma, 2))) / nu
        # SVarGamma[t] = SVarGamma[t - 1] * np.exp((r - 0.5 * sigma ** 2 + omega) * dt + h * dt)
        SVarGamma[t] = SVarGamma[t - 1] * np.exp((r - 0.5 * sigma ** 2 + omega) * dt + h)

    return SVarGamma


# Simulate variance gamma as the difference of two gammas
def VarianceGammaPath2(T, days, N, sigma, r, nu, theta, S0):
    dt = T / days
    size = (days, N)
    SVarGamma = np.zeros(size)
    SVarGamma[0] = S0
    for t in range(1, days):
        mu_p = 0.5 * np.sqrt(theta ** 2 + (2 * sigma ** 2 / nu)) + 0.5 * theta  # positive jump mean
        mu_n = 0.5 * np.sqrt(theta ** 2 + (2 * sigma ** 2 / nu)) - 0.5 * theta  # negative jump mean
        nu_p = mu_p ** 2 * nu  # positive jump variamce
        nu_n = mu_n ** 2 * nu  # negative jump variance
        omega = (np.log(1 - theta * nu - 0.5 * nu * sigma ** 2)) / nu
        Gamma_p = np.random.gamma(dt / nu, mu_p * nu, size=(N,))
        Gamma_n = np.random.gamma(dt / nu, mu_n * nu, size=(N,))
        SVarGamma[t] = SVarGamma[t - 1] * np.exp((r + omega) * dt + Gamma_p - Gamma_n)
    return SVarGamma


# plot the price paths
def plotVGPath(SVG, symbol, method, ax=None):
    if ax is None:
        ax = plt.gca()
    plt.figure(figsize=(8, 6))
    ax.plot(SVG)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Price')
    ax.set_title(f'Variance Gamma Simulated  Paths for {symbol} with {method}')
    # plt.savefig(f'VG_allpaths_{method}.png')
    return


# plot the distribution of prices average
def plotVGDist(SVG, symbol, ax=None):
    if ax is None:
        ax = plt.gca()
    avg_path = np.mean(SVG, axis=0)
    ax.hist(avg_path, bins=30)
    ax.set_xlabel('price')
    ax.set_ylabel('frequency')
    ax.set_title(f'Variance Gamma: Distribution of {symbol} prices')
    return


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


def find_moment(order, theta, nu, sigma):
    if order == 1:
        return theta * (1 - nu * sigma ** 2)
    elif order == 2:
        return theta ** 2 * (1 - 2 * nu * sigma ** 2) + 2 * sigma ** 2 * theta ** 2 * nu ** 2
    elif order == 3:
        return 3 * theta ** 3 * nu * (1 - 4 * sigma ** 2 * nu) + 6 * theta ** 3 * sigma ** 2 * nu ** 2
    elif order == 4:
        return 3 * theta ** 4 * (1 - 10 * nu * sigma ** 2 + 16 * nu ** 2 * sigma ** 4) + 12 * theta ** 4 * sigma ** 2 * nu * (1 - 4 * nu * sigma ** 2) + 24 * theta ** 4 * sigma ** 4 * nu ** 2
    else:
        return 0
