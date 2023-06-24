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

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def MertonPath(T, days, N, sigma, r, lambd, jump_mean, jump_std, S0):
    dt = T / days
    size = (days, N)
    SMerton = np.zeros(size)
    SMerton[0] = S0
    for t in range(1, days):
        # Random numbers generation
        Z = np.random.normal(size=(N,))
        Nj = np.random.poisson(lam=lambd * dt, size=(N,))
        J = np.random.normal(jump_mean, jump_std, size=(N,))
        jump_component = (np.exp(J) - 1) * Nj
        drift_component = (r - 0.5 * sigma ** 2) * dt
        diffusion_component = sigma * np.sqrt(dt) * Z
        # New prices computation
        SMerton[t] = SMerton[t - 1] * np.exp(drift_component + diffusion_component + jump_component)
    return SMerton


# plot the price paths
def plotMertonPath(SMerton, symbol):
    plt.figure(figsize=(10, 6))
    plt.plot(SMerton)
    plt.xlabel('Time (days)')
    plt.ylabel('Price')
    plt.title(f'Merton Jump diffusion Price Paths for {symbol}')
    plt.show()
    return


# plot the distribution of prices average
def plotMertonDist(SMerton, symbol):
    avg_path = np.mean(SMerton, axis=0)
    plt.hist(avg_path, bins='auto')
    plt.xlabel('price')
    plt.ylabel('frequency')
    plt.title(f'Merton Jump diffusion: Distribution of {symbol} prices')
    plt.show()
    return


def plotMertonLogReturns(SMerton, symbol):
    log_returns = np.log(SMerton[1:] / SMerton[:-1])
    log_returns = log_returns.flatten()
    sns.kdeplot(log_returns, label='Log Returns')
    plt.xlabel('Log Return')
    plt.ylabel('Density')
    plt.title(f'Distribution of Merton Log Returns for {symbol}')
    plt.legend()
    plt.show()
    return
