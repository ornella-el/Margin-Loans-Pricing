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
import scipy.stats as ss


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
def plotMertonPath(SMerton, symbol, ax=None):
    if ax is None:
        ax = plt.gca()
    plt.figure(figsize=(10, 6))
    ax.plot(SMerton)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Price')
    ax.set_title(f'Merton Jump diffusion Price Paths for {symbol}')
    return


# plot the distribution of prices average
def plotMertonDist(SMerton, symbol, ax=None):
    if ax is None:
        ax = plt.gca()
    avg_path = np.mean(SMerton, axis=0)
    ax.hist(avg_path, bins='auto')
    ax.set_xlabel('price')
    ax.set_ylabel('frequency')
    ax.set_title(f'Merton Jump diffusion: Distribution of {symbol} prices')
    return


def plotMertonLogReturns(SMerton, symbol, ax=None):
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


def plotMertonAtFixedTime(SMerton, time, symbol, ax):
    if ax is None:
        ax = plt.gca()
    fixed_values = SMerton[time, :]

    # Plotting the histogram
    hist, bins, _ = ax.hist(fixed_values, bins=30, density=True, alpha=0.9, label='Histogram')

    # Plotting the KDE approx
    sns.kdeplot(fixed_values, color='r', label='Approximation', ax=ax)

    ax.set_xlabel(f'{symbol} price after T = {time+1} days')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Merton Price at T ={time + 1}')
    ax.grid(True)
    ax.legend()
    return

