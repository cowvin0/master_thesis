import numpy as np

from ggkm.utils.kernels import KernelFunc
from ggkm.utils.pgamma_derivate import pgamma_shape_derivative_vec
from scipy.special import gamma, gammainc, digamma, gammaln
from scipy.optimize import minimize


class GGPoisson(KernelFunc):

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

        super().__init__(gamma=gamma, degree=degree, coef0=coef0, kernel=kernel)
        self.alpha_ = None
        self.K_ = None
        self.X_train_ = None

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

    def _penalized_loglikelihood(self, alpha, t, delta):

        f = self.K_ @ alpha
        theta = np.exp(f)
        ll = np.sum(delta * (f + self._log_fGG(t)) - theta * self._FGG(t))
        # reuse f = K_ @ alpha instead of recomputing alpha @ K_ @ alpha
        pen = (self.lambda_reg / 2) * (alpha @ f)

        return ll - pen

    def _e_step(self, alpha, t, delta):
        theta = np.exp(self.K_ @ alpha)
        SGG = 1.0 - self._FGG(t)
        Nhat = delta + theta * SGG

        return Nhat

    def _alpha_value_and_grad(self, alpha, Nhat):

        f = self.K_ @ alpha
        theta = np.exp(f)

        Q = np.sum(-theta + Nhat * f) - (self.lambda_reg / 2) * (alpha @ f)

        grad = self.K_ @ (self.lambda_reg * alpha - (Nhat - theta))

        return -Q, grad

    def _gg_value_and_grad(self, pars, t, delta, Nhat):

        a, d, p = pars
        s = d / p
        u = (t / a) ** p
        log_ta = np.log(t / a)

        FGG = gammainc(s, u)
        SGG = np.clip(1.0 - FGG, 1e-15, None)

        log_fGG = np.log(p) + (d - 1) * np.log(t) - u - d * np.log(a) - gammaln(s)

        Q = np.sum(delta * log_fGG + (Nhat - delta) * np.log(SGG))

        psi_s = digamma(s)
        gamma_s = gamma(s)
        H = np.exp(s * np.log(u) - u) / gamma_s
        G = pgamma_shape_derivative_vec(u, s)

        grad_a = np.sum(
            delta * ((p * u - d) / a) + (Nhat - delta) * ((p / a) * H / SGG)
        )
        grad_d = np.sum(delta * (log_ta - psi_s / p) - (Nhat - delta) * (G / (p * SGG)))
        grad_p = np.sum(
            delta * (1 / p - u * log_ta + d * psi_s / p**2)
            - (Nhat - delta) * ((-d * G / p**2 + H * log_ta) / SGG)
        )

        grad = np.array([grad_a, grad_d, grad_p])

        return -Q, -grad

    def fit(
        self,
        X,
        t,
        delta,
        tol=1e-6,
        max_em_iter=1000,
    ):

        n = len(t)

        self.loglik_history_ = []
        self.X_train_ = X
        self.K_ = self._compute_kernel(X, X)
        alpha = np.zeros(n)

        self.converged_ = False
        self.n_iter_ = 0
        ll_old = self._penalized_loglikelihood(alpha, t, delta)

        while True:

            self.n_iter_ += 1

            Nhat = self._e_step(alpha, t, delta)

            result_alpha = minimize(
                fun=self._alpha_value_and_grad,
                x0=alpha,
                jac=True,
                args=(Nhat,),
                method="L-BFGS-B",
            )

            alpha = result_alpha.x

            result_gg = minimize(
                fun=self._gg_value_and_grad,
                x0=[self.a, self.d, self.p],
                jac=True,
                args=(t, delta, Nhat),
                method="L-BFGS-B",
                bounds=[
                    (1e-5, None),
                    (1e-5, None),
                    (1e-5, None),
                ],
            )

            self.a, self.d, self.p = result_gg.x

            ll_new = self._penalized_loglikelihood(alpha, t, delta)
            self.loglik_history_.append(ll_new)

            diff_ll = abs(ll_new - ll_old)

            if diff_ll < tol:
                self.converged_ = True
                break

            if self.n_iter_ >= max_em_iter:
                break

            ll_old = ll_new

        self.alpha_ = alpha
        self.final_loglik_ = ll_new

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


# import numpy as np

# from utils.kernels import KernelFunc
# from utils.pgamma_derivate import pgamma_shape_derivative_vec
# from scipy.special import gamma, gammainc, digamma, gammaln
# from scipy.optimize import minimize


# class GGPoisson(KernelFunc):

#     def __init__(
#         self,
#         a=1.0,
#         d=1.0,
#         p=1.0,
#         kernel="rbf",
#         lambda_reg=1e-3,
#         gamma=1.0,
#         degree=3,
#         coef0=0.0,
#     ):
#         self.a = a
#         self.d = d
#         self.p = p
#         self.lambda_reg = lambda_reg
#         self.gamma = gamma
#         self.kernel = kernel
#         self.coef0 = coef0
#         self.degree = degree

#         super().__init__(gamma=gamma, degree=degree, coef0=coef0, kernel=kernel)
#         self.alpha_ = None
#         self.K_ = None
#         self.X_train_ = None

#     def _fGG(self, t):
#         a, d, p = self.a, self.d, self.p
#         s = d / p
#         log_f = (
#             np.log(p) + (d - 1) * np.log(t) - (t / a) ** p - d * np.log(a) - gammaln(s)
#         )
#         return np.exp(log_f)

#     def _log_fGG(self, t):
#         a, d, p = self.a, self.d, self.p
#         s = d / p
#         log_f = (
#             np.log(p) + (d - 1) * np.log(t) - (t / a) ** p - d * np.log(a) - gammaln(s)
#         )
#         return log_f

#     def _FGG(self, t):
#         return gammainc(self.d / self.p, (t / self.a) ** self.p)

#     def _penalized_loglikelihood(self, alpha, t, delta):

#         f = self.K_ @ alpha
#         theta = np.exp(f)
#         ll = np.sum(delta * (f + self._log_fGG(t)) - theta * self._FGG(t))
#         # reuse f = K_ @ alpha instead of recomputing alpha @ K_ @ alpha
#         pen = (self.lambda_reg / 2) * (alpha @ f)

#         return ll - pen

#     def _e_step(self, alpha, t, delta):
#         theta = np.exp(self.K_ @ alpha)
#         SGG = 1.0 - self._FGG(t)
#         Nhat = delta + theta * SGG

#         return Nhat

#     def _alpha_value_and_grad(self, alpha, Nhat):

#         f = self.K_ @ alpha
#         theta = np.exp(f)

#         Q = np.sum(-theta + Nhat * f) - (self.lambda_reg / 2) * (alpha @ f)

#         grad = self.K_ @ (self.lambda_reg * alpha - (Nhat - theta))

#         return -Q, grad

#     def _gg_value_and_grad(self, pars, t, delta, Nhat):

#         a, d, p = pars
#         s = d / p
#         u = (t / a) ** p
#         log_ta = np.log(t / a)

#         FGG = gammainc(s, u)
#         SGG = np.clip(1.0 - FGG, 1e-15, None)

#         log_fGG = np.log(p) + (d - 1) * np.log(t) - u - d * np.log(a) - gammaln(s)

#         Q = np.sum(delta * log_fGG + (Nhat - delta) * np.log(SGG))

#         psi_s = digamma(s)
#         gamma_s = gamma(s)
#         H = np.exp(s * np.log(u) - u) / gamma_s
#         G = pgamma_shape_derivative_vec(u, s)

#         grad_a = np.sum(
#             delta * ((p * u - d) / a) + (Nhat - delta) * ((p / a) * H / SGG)
#         )
#         grad_d = np.sum(delta * (log_ta - psi_s / p) - (Nhat - delta) * (G / (p * SGG)))
#         grad_p = np.sum(
#             delta * (1 / p - u * log_ta + d * psi_s / p**2)
#             - (Nhat - delta) * ((-d * G / p**2 + H * log_ta) / SGG)
#         )

#         grad = np.array([grad_a, grad_d, grad_p])

#         return -Q, -grad

#     def fit(
#         self,
#         X,
#         t,
#         delta,
#         tol=1e-6,
#         max_em_iter=1000,
#     ):

#         n = len(t)

#         self.loglik_history_ = []
#         self.X_train_ = X
#         self.K_ = self._compute_kernel(X, X)
#         alpha = np.zeros(n)

#         self.converged_ = False
#         self.n_iter_ = 0
#         ll_old = self._penalized_loglikelihood(alpha, t, delta)

#         while True:

#             self.n_iter_ += 1

#             Nhat = self._e_step(alpha, t, delta)

#             result_alpha = minimize(
#                 fun=self._alpha_value_and_grad,
#                 x0=alpha,
#                 jac=True,
#                 args=(Nhat,),
#                 method="L-BFGS-B",
#             )

#             alpha = result_alpha.x

#             result_gg = minimize(
#                 fun=self._gg_value_and_grad,
#                 x0=[self.a, self.d, self.p],
#                 jac=True,
#                 args=(t, delta, Nhat),
#                 method="L-BFGS-B",
#                 bounds=[
#                     (1e-5, None),
#                     (1e-5, None),
#                     (1e-5, None),
#                 ],
#             )

#             self.a, self.d, self.p = result_gg.x

#             ll_new = self._penalized_loglikelihood(alpha, t, delta)
#             self.loglik_history_.append(ll_new)

#             diff_ll = abs(ll_new - ll_old)

#             if diff_ll < tol:
#                 self.converged_ = True
#                 break

#             if self.n_iter_ >= max_em_iter:
#                 break

#             ll_old = ll_new

#         self.alpha_ = alpha
#         self.final_loglik_ = ll_new

#         return self

#     def predict_survival(self, X_new, t_grid):
#         K_new = self._compute_kernel(X_new, self.X_train_)
#         w = np.exp(K_new @ self.alpha_)
#         F = self._FGG(t_grid)
#         S = np.exp(-np.outer(w, F))
#         return S

#     def predict_cure_probability(self, X_new):
#         K_new = self._compute_kernel(X_new, self.X_train_)
#         return np.exp(-np.exp(K_new @ self.alpha_))
