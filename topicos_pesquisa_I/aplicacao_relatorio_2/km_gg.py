import warnings
import numpy as np
import pandas as pd
import optuna

from scipy.special import gamma, gammainc, digamma, gammaln
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

_EXP_MAX = 500.0
_LOG_EPS = 1e-300
_W_MAX = 1e10


class GG_KM:
    """
    GG-KM: Semiparametric Promotion Time Cure Model combining the
    Generalized Gamma distribution with Kernel Machines.

    Reference: Ferreira (2026).
    """

    def __init__(self, a=1.0, d=1.0, p=1.0, lambda_reg=1e-3, gamma_kernel=1.0):
        self.a = a
        self.d = d
        self.p = p
        self.lambda_reg = lambda_reg
        self.gamma_kernel = gamma_kernel

        self.alpha_ = None
        self.K_ = None
        self.X_train_ = None

    def gaussian_kernel(self, X1, X2):
        diff = X1[:, None, :] - X2[None, :, :]
        sq_dist = np.sum(diff**2, axis=2)
        return np.exp(-self.gamma_kernel * sq_dist)

    def _fGG(self, t):
        """GG PDF computed in log-space to avoid overflow."""
        a, d, p = self.a, self.d, self.p
        s = d / p
        log_f = (
            np.log(p)
            # + (d - 1) * np.log(np.maximum(t, _LOG_EPS))
            + (d - 1) * np.log(t)
            - (t / a) ** p
            - d * np.log(a)
            - gammaln(s)
        )
        # return np.exp(np.clip(log_f, -_EXP_MAX, _EXP_MAX))
        return np.exp(log_f)

    def _log_fGG(self, t):
        """Log of GG PDF, clipped for numerical safety."""
        a, d, p = self.a, self.d, self.p
        s = d / p
        log_f = (
            np.log(p) + (d - 1) * np.log(t) - (t / a) ** p - d * np.log(a) - gammaln(s)
        )
        # return np.clip(log_f, -_EXP_MAX, 0.0)
        return log_f

    def _FGG(self, t):
        """GG CDF via regularized lower incomplete gamma."""
        return gammainc(self.d / self.p, (t / self.a) ** self.p)

    # ------------------------------------------------------------------
    # Objective  [Eq. 5]
    # ------------------------------------------------------------------

    def _objective(self, params, K, t, delta):
        alpha, a, d, p = self._unpack(params)
        self.a, self.d, self.p = a, d, p

        f = K @ alpha
        # w = np.clip(np.exp(f), 0.0, _W_MAX)  # prevent overflow
        w = np.exp(f)
        ll = np.sum(delta * (f + self._log_fGG(t)) - w * self._FGG(t))
        pen = (self.lambda_reg / 2) * (alpha @ K @ alpha)

        J = -(ll - pen)
        return J if np.isfinite(J) else 1e18

    # ------------------------------------------------------------------
    # Analytical Gradients  [Eq. 6-9]
    # ------------------------------------------------------------------

    def _gradients(self, params, K, t, delta):
        alpha, a, d, p = self._unpack(params)
        self.a, self.d, self.p = a, d, p

        f = K @ alpha
        # w = np.clip(np.exp(f), 0.0, _W_MAX)
        w = np.exp(f)
        s = d / p
        u = (t / a) ** p
        FGG = self._FGG(t)

        eps = 1e-5
        G = (gammainc(s + eps, u) - gammainc(s - eps, u)) / (2 * eps)

        # log_ta = np.log(np.maximum(t / a, _LOG_EPS))
        log_ta = np.log(t / a)
        psi_s = digamma(s)
        gamma_s = gamma(s)
        us_exp_neg_u = np.exp(s * np.log(u) - u)  # u^s * exp(-u)
        # us_exp_neg_u = np.exp(  # u^s * exp(-u)
        #     np.clip(s * np.log(np.maximum(u, _LOG_EPS)) - u, -_EXP_MAX, 0.0)
        # )

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

        grads = np.where(np.isfinite(grads), grads, 1e10)
        return grads

    # ------------------------------------------------------------------
    # Fit  (Algorithm 1)
    # ------------------------------------------------------------------

    def fit(self, X, t, delta):
        n = len(t)
        self.X_train_ = X
        self.K_ = self.gaussian_kernel(X, X)

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

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_survival(self, X_new, t_grid):
        """
        Returns S_pop(t|x) = exp(-w(x) * F_GG(t))  of shape (m, T).
        """
        K_new = self.gaussian_kernel(X_new, self.X_train_)
        # w = np.clip(np.exp(K_new @ self.alpha_), 0.0, _W_MAX)
        w = np.exp(K_new @ self.alpha_)
        F = self._FGG(t_grid)  # (T,)
        S = np.exp(-np.outer(w, F))  # (m, T)
        # return np.clip(S, 0.0, 1.0)
        return S

    def predict_cure_probability(self, X_new):
        K_new = self.gaussian_kernel(X_new, self.X_train_)
        # return np.exp(-np.clip(np.exp(K_new @ self.alpha_), 0.0, _W_MAX))
        return np.exp(-np.exp(K_new @ self.alpha_))

    @staticmethod
    def _unpack(params):
        return params[:-3], params[-3], params[-2], params[-1]


# import numpy as np
# from scipy.special import gamma, gammainc, digamma, gammaln
# from scipy.optimize import minimize


# class GG_KM:
#     """
#     GG-KM: Semiparametric Promotion Time Cure Model
#     combining the Generalized Gamma distribution with Kernel Machines.

#     Reference: Ferreira (2026) - GG-KM: Flexible Semiparametric Promotion
#     Time Cure Modeling Based on The Generalized Gamma Distribution and
#     Kernel Machines.
#     """

#     def __init__(self, a=1.0, d=1.0, p=1.0, lambda_reg=1e-3, gamma_kernel=1.0):
#         """
#         Parameters
#         ----------
#         a            : float  - GG scale parameter (a > 0)
#         d            : float  - GG shape parameter (d > 0)
#         p            : float  - GG shape parameter (p > 0)
#         lambda_reg   : float  - Regularization strength for RKHS penalty
#         gamma_kernel : float  - Bandwidth for the Gaussian (RBF) kernel
#         """
#         self.a = a
#         self.d = d
#         self.p = p
#         self.lambda_reg = lambda_reg
#         self.gamma_kernel = gamma_kernel

#         self.alpha_ = None
#         self.K_ = None
#         self.X_train_ = None

#     def gaussian_kernel(self, X1, X2):
#         """
#         Compute the Gaussian (RBF) Gram matrix between X1 and X2.

#         K(xi, xj) = exp(-gamma * ||xi - xj||^2)

#         Parameters
#         ----------
#         X1 : (n, p) array
#         X2 : (m, p) array

#         Returns
#         -------
#         K  : (n, m) array
#         """
#         diff = X1[:, None, :] - X2[None, :, :]
#         sq_dist = np.sum(diff**2, axis=2)
#         return np.exp(-self.gamma_kernel * sq_dist)

#     def _fGG(self, t):
#         """GG PDF computed in log-space to avoid overflow."""
#         a, d, p = self.a, self.d, self.p
#         s = d / p
#         log_f = (
#             np.log(p)
#             + (d - 1) * np.log(np.maximum(t, _LOG_EPS))
#             - (t / a) ** p
#             - d * np.log(a)
#             - gammaln(s)
#         )
#         return np.exp(np.clip(log_f, -_EXP_MAX, _EXP_MAX))

#     def _log_fGG(self, t):
#         """Log of GG PDF, clipped for numerical safety."""
#         a, d, p = self.a, self.d, self.p
#         s = d / p
#         log_f = (
#             np.log(p)
#             + (d - 1) * np.log(np.maximum(t, _LOG_EPS))
#             - (t / a) ** p
#             - d * np.log(a)
#             - gammaln(s)
#         )
#         return np.clip(log_f, -_EXP_MAX, 0.0)

#     # def _fGG(self, t):
#     #     """GG probability density function."""
#     #     a, d, p = self.a, self.d, self.p
#     #     coef = p / (a**d * gammaln(d / p))
#     #     return coef * t ** (d - 1) * np.exp(-((t / a) ** p))

#     def _FGG(self, t):
#         """GG cumulative distribution function (regularized lower incomplete gamma)."""
#         return gammainc(self.d / self.p, (t / self.a) ** self.p)

#     def _objective(self, params, K, t, delta):
#         """
#         Compute the penalized objective J(Theta) to be minimized.

#         J = -sum_i { delta_i [f(xi) + log fGG(ti)] - exp(f(xi)) FGG(ti) }
#             + (lambda_reg / 2) * alpha^T K alpha
#         """
#         alpha, a, d, p = self._unpack_params(params)
#         self.a, self.d, self.p = a, d, p

#         f = K @ alpha
#         w = np.exp(f)

#         log_fGG = np.log(self._fGG(t) + 1e-300)
#         FGG = self._FGG(t)

#         log_likelihood = np.sum(delta * (f + log_fGG) - w * FGG)
#         penalty = (self.lambda_reg / 2) * alpha @ K @ alpha

#         return -(log_likelihood - penalty)

#     def _gradients(self, params, K, t, delta):
#         """
#         Compute analytical gradients of J w.r.t. all parameters.

#         Returns a flat array [grad_alpha (n,), grad_a, grad_d, grad_p].
#         """
#         alpha, a, d, p = self._unpack_params(params)
#         self.a, self.d, self.p = a, d, p

#         n = len(t)
#         f = K @ alpha
#         w = np.exp(f)
#         s = d / p
#         u = (t / a) ** p
#         FGG = self._FGG(t)

#         eps = 1e-5
#         G = (gammainc(s + eps, u) - gammainc(s - eps, u)) / (2 * eps)

#         grad_alpha = -K @ (delta - w * FGG) + self.lambda_reg * K @ alpha

#         grad_a = -np.sum(
#             delta * (p * u - d) / a + w * (p * u**s * np.exp(-u)) / (a * gamma(s))
#         )

#         psi_s = digamma(s)
#         grad_d = -np.sum(delta * (np.log(t / a) - psi_s / p) - w * G / p)

#         log_ta = np.log(t / a)
#         grad_p = -np.sum(
#             delta * (1 / p - u * log_ta + s * psi_s / p)
#             - w * (-d * G / p**2 + u**s * np.exp(-u) * log_ta / gamma(s))
#         )

#         return np.concatenate([grad_alpha, [grad_a, grad_d, grad_p]])

#     def fit(self, X, t, delta):
#         """
#         Fit the GG-KM model via L-BFGS-B.

#         Parameters
#         ----------
#         X     : (n, p) array of covariates
#         t     : (n,)   array of observed times
#         delta : (n,)   array of event indicators (1=event, 0=censored)
#         """
#         n = len(t)
#         self.X_train_ = X

#         self.K_ = self.gaussian_kernel(X, X)

#         alpha_0 = np.zeros(n)
#         gg_0 = np.array([self.a, self.d, self.p])
#         params_0 = np.concatenate([alpha_0, gg_0])

#         bounds = [(-np.inf, np.inf)] * n + [(1e-6, np.inf)] * 3

#         result = minimize(
#             fun=self._objective,
#             x0=params_0,
#             jac=self._gradients,
#             args=(self.K_, t, delta),
#             method="L-BFGS-B",
#             bounds=bounds,
#             options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
#         )

#         self.alpha_ = result.x[:n]
#         self.a, self.d, self.p = result.x[n], result.x[n + 1], result.x[n + 2]
#         self.converged_ = result.success
#         self.result_ = result

#         return self

#     def predict_cure_probability(self, X_new):
#         """
#         Predict the cure probability p0(x) = exp(-theta(x)) for new observations.

#         p0(x) = lim_{t->inf} S_pop(t|x) = exp(-exp(f(x)))
#         """
#         K_new = self.gaussian_kernel(X_new, self.X_train_)
#         f = K_new @ self.alpha_
#         return np.exp(-np.exp(f))

#     def predict_survival(self, X_new, t_grid):
#         """
#         Predict the population survival function S_pop(t|x) for a grid of times.

#         S_pop(t|x) = exp(-theta(x) * F_GG(t))

#         Parameters
#         ----------
#         X_new  : (m, p) array of covariates
#         t_grid : (T,)   array of time points

#         Returns
#         -------
#         S : (m, T) survival probabilities
#         """
#         K_new = self.gaussian_kernel(X_new, self.X_train_)
#         w = np.exp(K_new @ self.alpha_)
#         F = np.array([self._FGG(t) for t in t_grid])
#         return np.exp(-np.outer(w, F))

#     @staticmethod
#     def _unpack_params(params):
#         """Split flat parameter vector into (alpha, a, d, p)."""
#         alpha = params[:-3]
#         a, d, p = params[-3], params[-2], params[-1]
#         return alpha, a, d, p
