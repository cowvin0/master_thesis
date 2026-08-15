from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, expit, gamma, gammainc, gammaln
from sklearn.tree import DecisionTreeRegressor
from utils.kernels import KernelFunc
from utils.pgamma_derivate import pgamma_shape_derivative_vec


class GGBinomialGB:

    def __init__(
        self,
        K_bin=1,
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
        self.K_bin = K_bin

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
        s = self.d / self.p
        return (
            np.log(self.p)
            + (self.d - 1.0) * np.log(t)
            - (t / self.a) ** self.p
            - self.d * np.log(self.a)
            - gammaln(s)
        )

    def _fGG(self, t):
        return np.exp(self._log_fGG(t))

    def _FGG(self, t):
        return gammainc(self.d / self.p, (t / self.a) ** self.p)

    @staticmethod
    def _theta_star(eta):
        return expit(eta)

    def _predict_eta_train(self):
        return self.eta_train_

    def _predict_theta_star_train(self):
        return self._theta_star(self.eta_train_)

    def _loglikelihood(self, t, delta):
        theta_star = self._predict_theta_star_train()
        FGG = self._FGG(t)

        log_event = np.log(self.K_bin) + np.log(theta_star) + self._log_fGG(t)
        log_censored = np.log(1.0 - theta_star * FGG)

        return np.sum(delta * log_event + (self.K_bin - delta) * log_censored)

    def _e_step(self, t, delta):
        theta_star = self._predict_theta_star_train()

        FGG = self._FGG(t)
        SGG = np.clip(1.0 - FGG, 1e-15, 1.0)

        denominator = 1.0 - theta_star + theta_star * SGG
        frac = theta_star * SGG / denominator
        return delta + (self.K_bin - delta) * frac

    def _functional_gradient(self, eta, Nhat):
        theta_star = self._theta_star(eta)
        return Nhat - self.K_bin * theta_star

    def _boosting_m_step(self, X, Nhat):
        eta = self.eta_train_.copy()

        for m in range(self.n_estimators):
            gradient = self._functional_gradient(eta=eta, Nhat=Nhat)

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

            rho = self._line_search_step(eta, h, Nhat)
            eta_new = eta + rho * h

            improvement = np.mean(np.abs(eta_new - eta))

            self.models_.append(tree)
            self.model_weights_.append(rho)

            eta = eta_new

            if improvement < 1e-8:
                break

        self.eta_train_ = eta

    def _line_search_step(self, eta, h, Nhat):
        K_bin = self.K_bin

        def negative_Q(rho_arr):
            rho = rho_arr[0]
            eta_candidate = eta + rho * h
            Q = np.sum(Nhat * eta_candidate - K_bin * np.logaddexp(0.0, eta_candidate))
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
        SGG = np.clip(1.0 - FGG, 1e-15, None)
        log_fGG = np.log(p) + (d - 1.0) * np.log(t) - u - d * np.log(a) - gammaln(s)

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
            delta * (1.0 / p - u * log_ta + d * psi_s / p**2)
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

        self.initial_eta_ = np.zeros(n, dtype=float)
        self.eta_train_ = self.initial_eta_.copy()

        ll_old = self._loglikelihood(t, delta)
        ll_new = ll_old

        self.converged_ = False
        self.n_iter_ = 0

        while True:
            self.n_iter_ += 1

            Nhat = self._e_step(t, delta)
            self._boosting_m_step(X=X, Nhat=Nhat)
            self._gg_m_step(t=t, delta=delta, Nhat=Nhat)

            ll_new = self._loglikelihood(t, delta)
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
        eta = self.predict_eta(X_new)
        return self._theta_star(eta)

    def predict_survival(self, X_new, t_grid):
        theta_star = self.predict_theta(X_new)
        FGG = self._FGG(t_grid)
        SGG = 1.0 - FGG

        return (
            1.0 - theta_star[:, None] + theta_star[:, None] * SGG[None, :]
        ) ** self.K_bin

    def predict_cure_probability(self, X_new):
        theta_star = self.predict_theta(X_new)
        return (1.0 - theta_star) ** self.K_bin


class GGBinomial(KernelFunc):

    def __init__(
        self,
        K_bin=1,
        a=1.0,
        d=1.0,
        p=1.0,
        kernel="rbf",
        lambda_reg=1e-3,
        gamma=1.0,
        degree=3,
        coef0=0.0,
    ):

        self.K_bin = K_bin

        self.a = a
        self.d = d
        self.p = p

        self.lambda_reg = lambda_reg

        self.gamma = gamma
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0

        super().__init__(
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel=kernel,
        )

        self.alpha_ = None
        self.K_ = None
        self.X_train_ = None

    def _FGG(self, t):

        return gammainc(
            self.d / self.p,
            (t / self.a) ** self.p,
        )

    def _fGG(self, t):

        s = self.d / self.p
        log_f = (
            np.log(self.p)
            + (self.d - 1) * np.log(t)
            - (t / self.a) ** self.p
            - self.d * np.log(self.a)
            - gammaln(s)
        )

        return np.exp(log_f)

    def _log_fGG(self, t):

        s = self.d / self.p

        return (
            np.log(self.p)
            + (self.d - 1) * np.log(t)
            - (t / self.a) ** self.p
            - self.d * np.log(self.a)
            - gammaln(s)
        )

    def _theta_star(self, f):

        return 1.0 / (1.0 + np.exp(-f))

    def _penalized_loglikelihood(
        self,
        alpha,
        t,
        delta,
    ):

        f = self.K_ @ alpha
        theta_star = self._theta_star(f)
        FGG = self._FGG(t)
        ll = np.sum(
            delta * (np.log(self.K_bin) + np.log(theta_star) + self._log_fGG(t))
            + (self.K_bin - delta) * np.log(1.0 - theta_star * FGG)
        )
        pen = (self.lambda_reg / 2) * (alpha @ f)

        return ll - pen

    def _e_step(
        self,
        alpha,
        t,
        delta,
    ):

        f = self.K_ @ alpha
        theta_star = self._theta_star(f)
        SGG = np.clip(
            1.0 - self._FGG(t),
            1e-15,
            None,
        )
        frac = (theta_star * SGG) / (1.0 - theta_star + theta_star * SGG)
        Nhat = delta + (self.K_bin - delta) * frac

        return Nhat

    def _alpha_value_and_grad(
        self,
        alpha,
        Nhat,
    ):

        f = self.K_ @ alpha
        theta_star = self._theta_star(f)

        Q = np.sum(Nhat * f - self.K_bin * np.log1p(np.exp(f)))
        Q -= (self.lambda_reg / 2) * (alpha @ f)

        grad = self.K_ @ (Nhat - self.K_bin * theta_star - self.lambda_reg * alpha)

        return -Q, -grad

    def _gg_value_and_grad(
        self,
        pars,
        t,
        delta,
        Nhat,
    ):

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

        self.X_train_ = X

        self.K_ = self._compute_kernel(
            X,
            X,
        )

        alpha = np.zeros(n)
        self.loglik_history_ = []

        ll_old = self._penalized_loglikelihood(
            alpha,
            t,
            delta,
        )

        self.converged_ = False
        self.n_iter_ = 0

        while True:
            self.n_iter_ += 1
            Nhat = self._e_step(
                alpha,
                t,
                delta,
            )

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
                x0=[
                    self.a,
                    self.d,
                    self.p,
                ],
                jac=True,
                args=(
                    t,
                    delta,
                    Nhat,
                ),
                method="L-BFGS-B",
                bounds=[
                    (1e-5, None),
                    (1e-5, None),
                    (1e-5, None),
                ],
            )

            self.a, self.d, self.p = result_gg.x

            ll_new = self._penalized_loglikelihood(
                alpha,
                t,
                delta,
            )

            self.loglik_history_.append(ll_new)

            if abs(ll_new - ll_old) < tol:

                self.converged_ = True
                break

            if self.n_iter_ >= max_em_iter:
                break

            ll_old = ll_new

        self.alpha_ = alpha
        self.final_loglik_ = ll_new

        return self

    def predict_survival(
        self,
        X_new,
        t_grid,
    ):

        K_new = self._compute_kernel(
            X_new,
            self.X_train_,
        )
        f = K_new @ self.alpha_
        theta_star = self._theta_star(f)
        SGG = 1.0 - self._FGG(t_grid)

        return (
            1.0 - theta_star[:, None] + theta_star[:, None] * SGG[None, :]
        ) ** self.K_bin

    def predict_cure_probability(
        self,
        X_new,
    ):

        K_new = self._compute_kernel(
            X_new,
            self.X_train_,
        )
        f = K_new @ self.alpha_
        theta_star = self._theta_star(f)

        return (1.0 - theta_star) ** self.K_bin
