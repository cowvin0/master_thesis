import numpy as np

from utils.kernels import KernelFunc
from utils.pgamma_derivate import pgamma_shape_derivative_vec
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

    def _penalized_loglikelihood(self, alpha, t, delta):

        f = self.K_ @ alpha
        theta = np.exp(f)
        ll = np.sum(delta * (f + self._log_fGG(t)) - theta * self._FGG(t))
        pen = (self.lambda_reg / 2) * (alpha @ self.K_ @ alpha)

        return ll - pen

    def _FGG(self, t):
        return gammainc(self.d / self.p, (t / self.a) ** self.p)

    def _e_step(self, alpha, t, delta):
        theta = np.exp(self.K_ @ alpha)
        SGG = 1.0 - self._FGG(t)
        Nhat = delta + theta * SGG

        return Nhat

    def _alpha_objective(self, alpha, Nhat):

        f = self.K_ @ alpha
        theta = np.exp(f)
        Q = np.sum(-theta + Nhat * f) - (self.lambda_reg / 2) * (
            alpha @ self.K_ @ alpha
        )

        return -Q

    def _alpha_gradient(self, alpha, Nhat):
        theta = np.exp(self.K_ @ alpha)

        return -self.K_ @ (Nhat - theta) + self.lambda_reg * self.K_ @ alpha

    def _gg_objective(self, pars, t, delta, Nhat):

        a, d, p = pars

        self.a = a
        self.d = d
        self.p = p

        FGG = self._FGG(t)
        SGG = np.clip(1.0 - FGG, 1e-15, None)
        Q = np.sum(delta * self._log_fGG(t) + (Nhat - delta) * np.log(SGG))

        return -Q

    def _gg_gradient(self, pars, t, delta, Nhat):

        a, d, p = pars

        self.a = a
        self.d = d
        self.p = p

        s = d / p
        u = (t / a) ** p

        FGG = self._FGG(t)
        SGG = np.clip(1.0 - FGG, 1e-15, None)

        G = pgamma_shape_derivative_vec(u, s)
        log_ta = np.log(t / a)
        psi_s = digamma(s)
        gamma_s = gamma(s)
        us_exp_neg_u = np.exp(s * np.log(u) - u)
        H = us_exp_neg_u / gamma_s
        grad_a = np.sum(
            delta * ((p * u - d) / a) + (Nhat - delta) * ((p / a) * H / SGG)
        )
        grad_d = np.sum(delta * (log_ta - psi_s / p) - (Nhat - delta) * (G / (p * SGG)))
        grad_p = np.sum(
            delta * (1 / p - u * log_ta + d * psi_s / p**2)
            - (Nhat - delta) * ((-d * G / p**2 + H * log_ta) / SGG)
        )

        return -np.array([grad_a, grad_d, grad_p])

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
                fun=self._alpha_objective,
                x0=alpha,
                jac=self._alpha_gradient,
                args=(Nhat,),
                method="L-BFGS-B",
            )

            alpha = result_alpha.x

            result_gg = minimize(
                fun=self._gg_objective,
                x0=[self.a, self.d, self.p],
                jac=self._gg_gradient,
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
