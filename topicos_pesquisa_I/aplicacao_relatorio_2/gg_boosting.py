import numpy as np
from scipy.optimize import minimize
from scipy.special import gammainc, gammaln, gamma, digamma
from sklearn.tree import DecisionTreeRegressor


class PTCMBoost:

    def __init__(
        self,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=20,
        tol=1e-6,
        a=1.0,
        d=1.0,
        p=1.0,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tol = tol

        self.a = a
        self.d = d
        self.p = p

        self.trees_ = []
        self.leaf_updates_ = []
        self.f0_ = 0.0
        self.X_train_ = None

    def _distribution_gradient(
        self,
        params,
        t,
        delta,
        f,
    ):

        a, d, p = params

        w = np.exp(f)

        s = d / p

        u = (t / a) ** p

        eps = 1e-5

        G = (gammainc(s + eps, u) - gammainc(s - eps, u)) / (2 * eps)

        log_ta = np.log(t / a)
        psi_s = digamma(s)
        gamma_s = gamma(s)
        us_exp_neg_u = np.exp(s * np.log(u) - u)

        grad_a = -np.sum(
            delta * (p * u - d) / a + w * (p * us_exp_neg_u) / (a * gamma_s)
        )
        grad_d = -np.sum(delta * (log_ta - psi_s / p) - w * G / p)
        grad_p = -np.sum(
            delta * (1.0 / p - u * log_ta + s * psi_s / p)
            - w * (-d * G / p**2 + us_exp_neg_u * log_ta / gamma_s)
        )

        return np.array(
            [
                grad_a,
                grad_d,
                grad_p,
            ]
        )

    def _FGG(self, t, a=None, d=None, p=None):
        a = self.a if a is None else a
        d = self.d if d is None else d
        p = self.p if p is None else p

        return gammainc(d / p, (t / a) ** p)

    def _log_fGG(self, t, a=None, d=None, p=None):
        a = self.a if a is None else a
        d = self.d if d is None else d
        p = self.p if p is None else p

        s = d / p

        return (
            np.log(p)
            + (d - 1.0) * np.log(t)
            - (t / a) ** p
            - d * np.log(a)
            - gammaln(s)
        )

    def _objective_distribution(self, params, t, delta, f):
        a, d, p = params

        if a <= 0 or d <= 0 or p <= 0:
            return 1e20

        F = self._FGG(t, a, d, p)
        logf = self._log_fGG(t, a, d, p)
        ll = np.sum(delta * (f + logf) - np.exp(f) * F)

        return -ll

    def _update_distribution(self, t, delta, f):

        result = minimize(
            fun=self._objective_distribution,
            jac=self._distribution_gradient,
            x0=np.array([self.a, self.d, self.p]),
            args=(t, delta, f),
            method="L-BFGS-B",
            bounds=[
                (1e-6, None),
                (1e-6, None),
                (1e-6, None),
            ],
            options={
                "maxiter": 100,
                "ftol": 1e-9,
                "gtol": 1e-6,
            },
        )

        self.a, self.d, self.p = result.x

    def _compute_leaf_gammas(
        self,
        tree,
        X,
        gradient,
        hessian,
    ):

        leaf_id = tree.apply(X)

        updates = {}

        for leaf in np.unique(leaf_id):

            idx = leaf_id == leaf
            numerator = np.sum(gradient[idx])
            denominator = np.sum(hessian[idx])
            updates[leaf] = numerator / (denominator + 1e-12)

        return updates

    # def _compute_leaf_gammas(
    #     self,
    #     tree,
    #     X,
    #     delta,
    #     F,
    #     f,
    # ):

    #     leaf_id = tree.apply(X)

    #     updates = {}

    #     for leaf in np.unique(leaf_id):

    #         idx = leaf_id == leaf
    #         numerator = np.sum(delta[idx])
    #         denominator = np.sum(np.exp(f[idx]) * F[idx])
    #         gamma_jm = np.log((numerator + 1e-12) / (denominator + 1e-12))
    #         updates[leaf] = gamma_jm

    #     return updates

    def fit(self, X, t, delta):

        X = np.asarray(X, dtype=float)
        t = np.asarray(t, dtype=float)
        delta = np.asarray(delta, dtype=float)

        self.X_train_ = X
        F0 = self._FGG(t)

        self.f0_ = np.log((np.sum(delta) + 1e-12) / (np.sum(F0) + 1e-12))
        f = np.full(len(t), self.f0_)

        for _ in range(self.n_estimators):

            F = self._FGG(t)
            gradient = delta - np.exp(f) * F
            hessian = np.exp(f) * F
            pseudo_response = gradient / (hessian + 1e-12)
            # residuals = delta - np.exp(f) * F
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            # tree.fit(X, residuals)
            tree.fit(X, pseudo_response, sample_weight=hessian)

            gamma_dict = self._compute_leaf_gammas(
                tree=tree,
                X=X,
                gradient=gradient,
                hessian=hessian,
            )
            # gamma_dict = self._compute_leaf_gammas(
            #     tree=tree,
            #     X=X,
            #     delta=delta,
            #     F=F,
            #     f=f,
            # )

            leaves = tree.apply(X)
            update = np.array([gamma_dict[l] for l in leaves])

            f += self.learning_rate * update

            self.trees_.append(tree)
            self.leaf_updates_.append(gamma_dict)

        self._update_distribution(
            t=t,
            delta=delta,
            f=f,
        )

        self.f_train_ = f
        return self

    def _predict_f(self, X):

        X = np.asarray(X, dtype=float)

        f = np.full(X.shape[0], self.f0_)

        for tree, gamma_dict in zip(self.trees_, self.leaf_updates_):

            leaves = tree.apply(X)
            update = np.array([gamma_dict.get(l, 0.0) for l in leaves])
            f += self.learning_rate * update

        return f

    def _predict_theta(self, X):

        return np.exp(self._predict_f(X))

    def predict_cure_probability(self, X):

        return np.exp(-self._predict_theta(X))

    def predict_survival(
        self,
        X,
        t_grid,
    ):

        theta = self._predict_theta(X)
        F = self._FGG(np.asarray(t_grid))
        return np.exp(-np.outer(theta, F))
