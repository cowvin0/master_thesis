from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gamma, gammainc, gammaln
from sklearn.tree import DecisionTreeRegressor
from utils.kernels import KernelFunc
from utils.pgamma_derivate import pgamma_shape_derivative_vec


class GGPoissonGB:

    def __init__(
        self,
        a=1.0,
        d=1.0,
        p=1.0,
        learning_rate=0.05,
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=20,
        max_features=None,
        random_state=None,
    ):
        self.a = a
        self.d = d
        self.p = p

        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.X_train_ = None
        self.initial_eta_ = None
        self.eta_train_ = None
        self.models_ = []
        self.model_weights_ = []
        self.loglik_history_ = []
        self.converged_ = False
        self.n_iter_ = 0
        self.final_loglik_ = None

    def _log_fGG(self, t):
        a, d, p = self.a, self.d, self.p
        s = d / p
        return (
            np.log(p) + (d - 1) * np.log(t) - (t / a) ** p - d * np.log(a) - gammaln(s)
        )

    def _fGG(self, t):
        return np.exp(self._log_fGG(t))

    def _FGG(self, t):
        return gammainc(self.d / self.p, (t / self.a) ** self.p)

    def _penalized_loglikelihood(self, t, delta):
        eta = self.eta_train_
        theta = np.exp(eta)
        FGG = self._FGG(t)
        return np.sum(delta * (eta + self._log_fGG(t)) - theta * FGG)

    def _e_step(self, t, delta):
        theta = self.eta_train_
        SGG = 1.0 - self._FGG(t)
        return delta + theta * SGG

    def _functional_gradient(self, eta, Nhat, FGG):
        theta = np.exp(eta)
        return Nhat - FGG * theta

    def _boosting_m_step(self, X, t, Nhat):
        FGG = self._FGG(t)
        eta = self.eta_train_.copy()

        for m in range(self.n_estimators):
            gradient = self._functional_gradient(eta=eta, Nhat=Nhat, FGG=FGG)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=(
                    None if self.random_state is None else self.random_state + m
                ),
            )
            tree.fit(X, gradient)
            h = tree.predict(X)

            rho = self._line_search_step(eta, h, Nhat, FGG)
            eta_new = eta + rho * h

            improvement = np.mean(np.abs(eta_new - eta))

            self.models_.append(tree)
            self.model_weights_.append(rho)

            eta = eta_new

            if improvement < 1e-8:
                break

        self.eta_train_ = eta

    def _line_search_step(self, eta, h, Nhat, FGG):

        def negative_Q(rho):
            eta_candidate = eta + rho * h
            theta_candidate = np.exp(eta_candidate)
            Q = np.sum(Nhat * eta_candidate - FGG * theta_candidate)
            return -Q

        result = minimize(
            negative_Q,
            x0=np.array([self.learning_rate]),
            method="L-BFGS-B",
            bounds=[(0.0, 10.0)],
        )

        rho = float(result.x[0])
        if not np.isfinite(rho) or rho <= 1e-12:
            rho = self.learning_rate
        return rho

    def _gg_value_and_grad(self, pars, t, delta, Nhat):
        a, d, p = pars
        s = d / p
        u = (t / a) ** p
        log_ta = np.log(t / a)

        FGG = gammainc(s, u)
        SGG = 1.0 - FGG
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

    def _gg_m_step(self, t, delta, Nhat):
        result_gg = minimize(
            fun=self._gg_value_and_grad,
            x0=[self.a, self.d, self.p],
            jac=True,
            args=(t, delta, Nhat),
            method="L-BFGS-B",
            bounds=[(1e-5, None)] * 3,
        )
        self.a, self.d, self.p = result_gg.x

    def fit(self, X, t, delta, tol=1e-6, max_em_iter=1000):
        X = np.asarray(X)
        t = np.asarray(t)
        delta = np.asarray(delta)
        n = len(t)

        self.X_train_ = X
        self.models_ = []
        self.model_weights_ = []
        self.loglik_history_ = []
        self.converged_ = False
        self.n_iter_ = 0

        self.initial_eta_ = np.zeros(n, dtype=float)
        self.eta_train_ = self.initial_eta_.copy()

        ll_old = self._penalized_loglikelihood(t, delta)
        ll_new = ll_old

        while True:
            self.n_iter_ += 1

            Nhat = self._e_step(t, delta)

            self._boosting_m_step(X=X, t=t, Nhat=Nhat)

            self._gg_m_step(t=t, delta=delta, Nhat=Nhat)

            ll_new = self._penalized_loglikelihood(t, delta)
            self.loglik_history_.append(ll_new)

            if abs(ll_new - ll_old) < tol:
                self.converged_ = True
                break
            if self.n_iter_ >= max_em_iter:
                break

            ll_old = ll_new

        self.final_loglik_ = ll_new
        return self

    def predict_eta(self, X_new):
        X_new = np.asarray(X_new)
        n = len(X_new)

        eta = np.full(n, self.initial_eta_.mean(), dtype=float)
        for tree, weight in zip(self.models_, self.model_weights_):
            eta += weight * tree.predict(X_new)

        return eta

    def predict_theta(self, X_new):
        return np.exp(self.predict_eta(X_new))

    def predict_survival(self, X_new, t_grid):
        theta = self.predict_theta(X_new)
        F = self._FGG(t_grid)
        return np.exp(-np.outer(theta, F))

    def predict_cure_probability(self, X_new):
        theta = self.predict_theta(X_new)
        return np.exp(-theta)


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
