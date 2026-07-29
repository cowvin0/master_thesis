import numpy as np
import pandas as pd

from scipy.special import expit


def pi_method(x1, x2, method):
    if method == 1:
        eta = 0.3 - 5 * x1 - 3 * x2
        return expit(eta)

    elif method == 2:
        eta = 0.3 - 5 * (x1**2) + 3 * (x2**2)
        return expit(eta)

    elif method == 3:
        eta = 0.3 - 5 * np.cos(x1) - 3 * np.sin(x2)
        return np.exp(-np.exp(eta))

    else:
        raise ValueError("method must be 1, 2, or 3")


def weibull_inverse(u, alpha, beta1, beta2, z1, z2):
    linpred = beta1 * z1 + beta2 * z2
    return ((-np.log(1 - u)) / np.exp(linpred)) ** (1 / alpha)


def simulate_pcm(
    n=300, method=1, alpha=2, beta1=1, beta2=0.5, censor_rate=0.2, seed=None
):

    if seed is not None:
        np.random.seed(seed)

    x1 = np.random.normal(size=n)
    x2 = np.random.normal(size=n)

    z1 = x1.copy()
    z2 = x2.copy()

    pi_x = np.clip(pi_method(x1, x2, method), 1e-12, 1 - 1e-12)

    t = np.zeros(n)
    delta = np.zeros(n, dtype=int)
    cured = np.zeros(n, dtype=int)

    for i in range(n):

        U = np.random.uniform()
        C = np.random.exponential(scale=1 / censor_rate)

        if U <= (1 - pi_x[i]):
            t[i] = C
            delta[i] = 0
            cured[i] = 1

        else:
            U1 = np.random.uniform(1 - pi_x[i], 1)
            v = np.log(U1) / np.log(1 - pi_x[i])
            y = weibull_inverse(
                u=v, alpha=alpha, beta1=beta1, beta2=beta2, z1=z1[i], z2=z2[i]
            )

            t[i] = min(y, C)

            if y <= C:
                delta[i] = 1
            else:
                delta[i] = 0

    df = pd.DataFrame(
        {
            "time": t,
            "event": delta,
            "x1": x1,
            "x2": x2,
            "z1": z1,
            "z2": z2,
            "pi_x": pi_x,
            "cured": cured,
        }
    )

    return df
