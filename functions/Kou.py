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

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss


def KouPath(T, days, N, sigma, r, lambd, p, eta1, eta2, S0):
    dt = T / days
    size = (days, N)
    SKou = np.zeros(size)
    SKou[0] = S0
    for t in range(1, days):
        # Random numbers generation
        Z = np.random.normal(size=(N,))
        Nj = np.random.poisson(lam=lambd * dt, size=(N,))

        # Generate jump sizes J
        U = np.random.uniform(size=(N,))  # Generate uniform random variables
        J = np.zeros_like(U)  # Initialize jump sizes
        J[U < p] = -np.log(1 - U[U < p]) / eta1  # Negative jumps
        J[U >= p] = np.log(U[U >= p]) / eta2  # Positive jumps

        # Find components
        jump_component = J * Nj
        drift_component = (r - 0.5 * sigma ** 2) * dt
        diffusion_component = sigma * np.sqrt(dt) * Z

        # New prices computation
        SKou[t] = SKou[t - 1] * np.exp(drift_component + diffusion_component + jump_component)
    return SKou


# plot the price paths
def plotKouPath(SKou, symbol, ax=None):
    if ax is None:
        ax = plt.gca()
    plt.figure(figsize=(10, 6))
    ax.plot(SKou)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Price')
    ax.set_title(f'Kou Jump Diffusion Price Paths for {symbol}')
    return


# plot the distribution of prices average
def plotKouDist(SKou, symbol, ax=None):
    if ax is None:
        ax = plt.gca()
    avg_path = np.mean(SKou, axis=0)
    ax.hist(avg_path, bins='auto')
    ax.set_xlabel('price')
    ax.set_ylabel('frequency')
    ax.set_title(f'Kou Jump Diffusion: Distribution of {symbol} prices')
    return


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


def plotKouAtFixedTime(SKou, time, symbol, ax):
    if ax is None:
        ax = plt.gca()
    fixed_values = SKou[time, :]

    # Plotting the histogram
    hist, bins, _ = ax.hist(fixed_values, bins=30, density=True, alpha=0.9, label='Histogram')

    # Plotting the KDE approx
    sns.kdeplot(fixed_values, color='r', label='Approximation', ax=ax)

    ax.set_xlabel(f'{symbol} price after T = {time+1} days')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Kou Price at T ={time + 1}')
    ax.grid(True)
    ax.legend()
    return
