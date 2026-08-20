from __future__ import annotations

import numpy as np

from scipy.optimize import minimize
from scipy.special import digamma, gamma, gammainc, gammaln
from sklearn.tree import DecisionTreeRegressor
from utils.kernels import KernelFunc
from utils.pgamma_derivate import pgamma_shape_derivative_vec


class GGNBGB:

    def __init__(
        self,
        a=1.0,
        d=1.0,
        p=1.0,
        phi=None,
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

        self.estimate_phi = phi is None
        self.phi = 0.5 if phi is None else phi

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
        self.phi_ = None

    def _log_fGG(self, t):
        a, d, p = self.a, self.d, self.p
        s = d / p
        return (
            np.log(p)
            + (d - 1.0) * np.log(t)
            - (t / a) ** p
            - d * np.log(a)
            - gammaln(s)
        )

    def _fGG(self, t):
        return np.exp(self._log_fGG(t))

    def _FGG(self, t):
        return gammainc(self.d / self.p, (t / self.a) ** self.p)

    def _predict_eta_train(self):
        return self.eta_train_

    def _predict_theta_train(self):
        return np.exp(self.eta_train_)

    def _penalized_loglikelihood(self, t, delta):
        eta = self._predict_eta_train()
        theta = np.exp(eta)
        FGG = self._FGG(t)
        phi = self.phi

        log_term = np.log1p(phi * theta * FGG)

        return np.sum(delta * (eta + self._log_fGG(t)) - (1.0 / phi + delta) * log_term)

    def _e_step(self, t, delta):
        theta = self._predict_theta_train()

        FGG = self._FGG(t)
        SGG = 1.0 - FGG

        phi = self.phi
        denominator = 1.0 + phi * theta * FGG
        return delta + ((1.0 + phi * delta) * theta * SGG) / denominator

    def _functional_gradient(self, eta, Nhat):
        phi = self.phi

        eta_safe = eta
        theta = np.exp(eta_safe)

        denominator = 1.0 + phi * theta

        gradient = Nhat - (Nhat + 1.0 / phi) * (phi * theta / denominator)

        return gradient

    def _boosting_m_step(self, X, Nhat):

        eta = self.eta_train_.copy()

        for m in range(self.n_estimators):

            gradient = self._functional_gradient(
                eta=eta,
                Nhat=Nhat,
            )

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

            rho = self._line_search_step(
                eta=eta,
                h=h,
                Nhat=Nhat,
            )

            step = self.learning_rate * rho
            eta_new = eta + step * h
            eta_new = eta_new
            improvement = np.mean(np.abs(eta_new - eta))

            self.models_.append(tree)
            self.model_weights_.append(step)

            eta = eta_new

            if improvement < 1e-8:
                break

    def _line_search_step(self, eta, h, Nhat):

        phi = self.phi

        def negative_Q(rho_arr):

            rho = rho_arr[0]
            eta_candidate = eta + rho * h
            log_term = np.logaddexp(
                0.0,
                np.log(phi) + eta_candidate,
            )
            Q = np.sum(Nhat * eta_candidate - (Nhat + 1.0 / phi) * log_term)

            return -Q

        result = minimize(
            negative_Q,
            x0=np.array([1.0]),
            method="L-BFGS-B",
            bounds=[(0.0, 10.0)],
        )
        rho = float(result.x[0])

        if not result.success or not np.isfinite(rho) or rho <= 1e-12:
            rho = 1.0

        return rho

    # def _line_search_step(self, eta, h, Nhat):
    #     phi = self.phi

    #     def negative_Q(rho_arr):
    #         rho = rho_arr[0]
    #         eta_candidate = eta + rho * h
    #         log_term = np.log1p(phi * np.exp(eta_candidate))
    #         Q = np.sum(Nhat * eta_candidate - (Nhat + 1.0 / phi) * log_term)
    #         return -Q

    #     result = minimize(
    #         negative_Q,
    #         x0=np.array([self.learning_rate]),
    #         method="L-BFGS-B",
    #         bounds=[(0.0, 10.0)],
    #     )

    #     rho = float(result.x[0])
    #     if not np.isfinite(rho) or rho <= 1e-12:
    #         rho = self.learning_rate
    #     return rho

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
            bounds=[(1e-05, None)] * 3,
        )
        self.a, self.d, self.p = result_gg.x

    def _phi_value_and_grad(self, phi_arr, const, A, delta):
        phi = phi_arr[0]

        log1p_phiA = np.log1p(phi * A)

        ll = np.sum(const - (1.0 / phi + delta) * log1p_phiA)
        grad = np.sum(log1p_phiA / phi**2 - (1.0 / phi + delta) * A / (1.0 + phi * A))

        return -ll, np.array([-grad])

    def _phi_m_step(self, t, delta):
        eta_fixed = self.eta_train_
        theta_fixed = np.exp(eta_fixed)
        FGG_fixed = self._FGG(t)
        A_fixed = theta_fixed * FGG_fixed
        const_fixed = delta * (eta_fixed + self._log_fGG(t))

        result_phi = minimize(
            fun=self._phi_value_and_grad,
            x0=np.array([self.phi]),
            jac=True,
            args=(const_fixed, A_fixed, delta),
            method="L-BFGS-B",
            bounds=[(1e-8, None)],
        )
        self.phi = float(result_phi.x[0])

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

            self._boosting_m_step(X=X, Nhat=Nhat)
            self._gg_m_step(t=t, delta=delta, Nhat=Nhat)

            if self.estimate_phi:
                self._phi_m_step(t=t, delta=delta)

            ll_new = self._penalized_loglikelihood(t, delta)
            self.loglik_history_.append(ll_new)

            if abs(ll_new - ll_old) < tol:
                self.converged_ = True
                break
            if self.n_iter_ >= max_em_iter:
                break

            ll_old = ll_new

        self.final_loglik_ = ll_new
        self.phi_ = self.phi
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
        return (1.0 + self.phi * np.outer(theta, F)) ** (-1.0 / self.phi)

    def predict_cure_probability(self, X_new):
        theta = self.predict_theta(X_new)
        return (1.0 + self.phi * theta) ** (-1.0 / self.phi)


class GGNB(KernelFunc):

    def __init__(
        self,
        a=1.0,
        d=1.0,
        p=1.0,
        phi=None,
        kernel="rbf",
        lambda_reg=1e-3,
        gamma=1.0,
        degree=3,
        coef0=0.0,
    ):

        self.a = a
        self.d = d
        self.p = p
        self.estimate_phi = phi is None
        self.phi = 0.5 if phi is None else phi

        self.lambda_reg = lambda_reg
        super().__init__(kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)

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

    def _phi_value_and_grad(
        self,
        phi_arr,
        const,
        A,
        delta,
    ):

        phi = phi_arr[0]
        log1p_phiA = np.log1p(phi * A)

        ll = np.sum(const - (1.0 / phi + delta) * log1p_phiA)

        grad = np.sum(log1p_phiA / phi**2 - (1.0 / phi + delta) * A / (1.0 + phi * A))

        return -ll, np.array([-grad])

    def _penalized_loglikelihood(
        self,
        alpha,
        t,
        delta,
    ):

        f = self.K_ @ alpha
        theta = np.exp(f)
        FGG = self._FGG(t)
        phi = self.phi
        ll = np.sum(
            delta * (f + self._log_fGG(t))
            - (1.0 / phi + delta) * np.log1p(phi * theta * FGG)
        )
        pen = (self.lambda_reg / 2) * (alpha @ f)

        return ll - pen

    def _e_step(self, alpha, t, delta):

        theta = np.exp(self.K_ @ alpha)
        FGG = self._FGG(t)
        SGG = 1.0 - FGG
        phi = self.phi
        Nhat = delta + ((1.0 + phi * delta) * theta * SGG) / (1.0 + phi * theta * FGG)

        return Nhat

    def _alpha_value_and_grad(
        self,
        alpha,
        Nhat,
    ):

        f = self.K_ @ alpha
        theta = np.exp(f)
        phi = self.phi

        Q = np.sum(Nhat * f - (Nhat + 1.0 / phi) * np.log1p(phi * theta))
        Q -= (self.lambda_reg / 2) * (alpha @ f)

        score = Nhat - (Nhat + 1.0 / phi) * (phi * theta) / (1.0 + phi * theta)
        grad = self.K_ @ (self.lambda_reg * alpha - score)

        return -Q, grad

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

        self.loglik_history_ = []
        self.X_train_ = X
        self.K_ = self._compute_kernel(X, X)
        self.converged_ = False
        self.n_iter_ = 0

        alpha = np.zeros(n)

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

            if self.estimate_phi:
                f_fixed = self.K_ @ alpha
                theta_fixed = np.exp(f_fixed)
                FGG_fixed = self._FGG(t)
                A_fixed = theta_fixed * FGG_fixed
                const_fixed = delta * (f_fixed + self._log_fGG(t))

                result_phi = minimize(
                    fun=self._phi_value_and_grad,
                    x0=np.array([self.phi]),
                    jac=True,
                    args=(const_fixed, A_fixed, delta),
                    method="L-BFGS-B",
                    bounds=[(1e-8, None)],
                )

                self.phi = result_phi.x[0]

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
        self.phi_ = self.phi

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
        theta = np.exp(K_new @ self.alpha_)
        F = self._FGG(t_grid)
        S = (1.0 + self.phi * np.outer(theta, F)) ** (-1.0 / self.phi)

        return S

    def predict_cure_probability(self, X_new):
        K_new = self._compute_kernel(
            X_new,
            self.X_train_,
        )
        theta = np.exp(K_new @ self.alpha_)

        return (1.0 + self.phi * theta) ** (-1.0 / self.phi)
