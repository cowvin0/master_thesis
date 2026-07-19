import ctypes
import numpy as np

from scipy.special import gamma, gammainc, digamma, gammaln
from scipy.optimize import minimize

_lib = ctypes.CDLL("./libspecfun.so")
_lib.pgamma_1st_derivative.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
]
_lib.pgamma_1st_derivative.restype = ctypes.c_double


def pgamma_shape_derivative(x, shape, scale=1.0):
    return _lib.pgamma_1st_derivative(
        float(x),
        float(shape),
        float(scale),
    )


def pgamma_shape_derivative_vec(u, shape, scale=1.0):
    u = np.asarray(u, dtype=float)

    return np.array(
        [pgamma_shape_derivative(ui, shape, scale) for ui in u],
        dtype=float,
    )


class GG_KM:

    def __init__(
        self,
        a=1.0,
        d=1.0,
        p=1.0,
        kernel="rbf",
        lambda_reg=1e-3,
        gamma=1.0,
        degree=3,
        coef0=0.0,
    ):
        self.a = a
        self.d = d
        self.p = p
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.kernel = kernel
        self.coef0 = coef0
        self.degree = degree

        self.alpha_ = None
        self.K_ = None
        self.X_train_ = None

    def _compute_kernel(self, X1, X2):
        kernels = {
            "linear": self.linear_kernel,
            "rbf": self.gaussian_kernel,
            "laplacian": self.laplacian_kernel,
            "exponential": self.exponential_kernel,
            "cauchy": self.cauchy_kernel,
            "sigmoid": self.sigmoid_kernel,
            "polynomial": self.polynomial_kernel,
        }

        if self.kernel not in kernels:
            raise ValueError(
                f"Unknown kernel '{self.kernel}'. "
                f"Available kernels: {list(kernels.keys())}"
            )

        return kernels[self.kernel](X1, X2)

    def linear_kernel(self, X1, X2):
        return X1 @ X2.T

    def laplacian_kernel(self, X1, X2):
        diff = np.abs(X1[:, None, :] - X2[None, :, :])
        dist = np.sum(diff, axis=2)
        return np.exp(-self.gamma * dist)

    def sigmoid_kernel(self, X1, X2):
        return np.tanh(self.gamma * (X1 @ X2.T) + self.coef0)

    def cauchy_kernel(self, X1, X2):
        diff = X1[:, None, :] - X2[None, :, :]
        sq_dist = np.sum(diff**2, axis=2)
        return 1 / (1 + self.gamma * sq_dist)

    def exponential_kernel(self, X1, X2):
        diff = X1[:, None, :] - X2[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        return np.exp(-self.gamma * dist)

    def polynomial_kernel(self, X1, X2):
        return (self.gamma * (X1 @ X2.T) + self.coef0) ** self.degree

    def gaussian_kernel(self, X1, X2):
        diff = X1[:, None, :] - X2[None, :, :]
        sq_dist = np.sum(diff**2, axis=2)
        return np.exp(-self.gamma * sq_dist)

    def _fGG(self, t):
        a, d, p = self.a, self.d, self.p
        s = d / p
        log_f = (
            np.log(p) + (d - 1) * np.log(t) - (t / a) ** p - d * np.log(a) - gammaln(s)
        )
        return np.exp(log_f)

    def _log_fGG(self, t):
        a, d, p = self.a, self.d, self.p
        s = d / p
        log_f = (
            np.log(p) + (d - 1) * np.log(t) - (t / a) ** p - d * np.log(a) - gammaln(s)
        )
        return log_f

    def _FGG(self, t):
        return gammainc(self.d / self.p, (t / self.a) ** self.p)

    def _objective(self, params, K, t, delta):
        alpha, a, d, p = params[:-3], params[-3], params[-2], params[-1]
        self.a, self.d, self.p = a, d, p

        f = K @ alpha
        w = np.exp(f)
        ll = np.sum(delta * (f + self._log_fGG(t)) - w * self._FGG(t))
        pen = (self.lambda_reg / 2) * (alpha @ K @ alpha)

        J = -(ll - pen)
        return J if np.isfinite(J) else 1e18

    def _gradients(self, params, K, t, delta):
        alpha, a, d, p = params[:-3], params[-3], params[-2], params[-1]
        self.a, self.d, self.p = a, d, p

        f = K @ alpha
        w = np.exp(f)
        s = d / p
        u = (t / a) ** p
        FGG = self._FGG(t)

        G = pgamma_shape_derivative_vec(u, s)

        log_ta = np.log(t / a)
        psi_s = digamma(s)
        gamma_s = gamma(s)
        us_exp_neg_u = np.exp(s * np.log(u) - u)

        grad_alpha = -K @ (delta - w * FGG) + self.lambda_reg * K @ alpha
        grad_a = -np.sum(
            delta * (p * u - d) / a + w * (p * us_exp_neg_u) / (a * gamma_s)
        )

        grad_d = -np.sum(delta * (log_ta - psi_s / p) - w * G / p)

        grad_p = -np.sum(
            delta * (1 / p - u * log_ta + s * psi_s / p)
            - w * (-d * G / p**2 + us_exp_neg_u * log_ta / gamma_s)
        )

        grads = np.concatenate([grad_alpha, [grad_a, grad_d, grad_p]])

        return grads

    def fit(self, X, t, delta):
        n = len(t)
        self.X_train_ = X
        self.K_ = self._compute_kernel(X, X)

        params_0 = np.concatenate([np.zeros(n), [self.a, self.d, self.p]])
        bounds = [(-np.inf, np.inf)] * n + [(1e-4, 1e4)] * 3

        result = minimize(
            fun=self._objective,
            x0=params_0,
            jac=self._gradients,
            args=(self.K_, t, delta),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-6},
        )

        self.alpha_ = result.x[:n]
        self.a, self.d, self.p = result.x[n], result.x[n + 1], result.x[n + 2]
        self.converged_ = result.success
        return self

    def predict_survival(self, X_new, t_grid):
        K_new = self._compute_kernel(X_new, self.X_train_)
        w = np.exp(K_new @ self.alpha_)
        F = self._FGG(t_grid)
        S = np.exp(-np.outer(w, F))
        return S

    def predict_cure_probability(self, X_new):
        K_new = self._compute_kernel(X_new, self.X_train_)
        return np.exp(-np.exp(K_new @ self.alpha_))
