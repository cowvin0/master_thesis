import numpy as np
import os

from copy import deepcopy
from joblib import Parallel, delayed
from models.km_gg import GGPoisson

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


class GGKMKernelBagging:

    def __init__(
        self,
        estimator=None,
        kernels=None,
        n_estimators=50,
        bootstrap=True,
        random_state=None,
        n_jobs=-1,
        **estimator_params,
    ):

        self.estimator = estimator

        self.kernels = (
            kernels
            if kernels is not None
            else [
                "linear",
                "rbf",
                "laplacian",
                "exponential",
                "cauchy",
            ]
        )

        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.estimator_params = estimator_params
        self.models_ = []

    @staticmethod
    def _fit_single_estimator(
        seed,
        estimator,
        kernels,
        bootstrap,
        X,
        t,
        delta,
        estimator_params,
    ):

        rng = np.random.default_rng(seed)

        n = len(t)

        if bootstrap:

            idx = rng.choice(
                n,
                size=n,
                replace=True,
            )

            X_boot = X[idx]
            t_boot = t[idx]
            delta_boot = delta[idx]

        else:

            X_boot = X
            t_boot = t
            delta_boot = delta

        if estimator is not None:

            model = deepcopy(estimator)

            kernel_name = getattr(
                model,
                "kernel",
                "user_estimator",
            )

        else:

            kernel_name = rng.choice(kernels)

            model = GGPoisson(
                kernel=kernel_name,
                **estimator_params,
            )

        model.fit(
            X_boot,
            t_boot,
            delta_boot,
        )

        return {
            "kernel": kernel_name,
            "model": model,
        }

    @staticmethod
    def _predict_survival_single(
        obj,
        X_new,
        t_grid,
    ):
        return obj["model"].predict_survival(
            X_new,
            t_grid,
        )

    @staticmethod
    def _predict_cure_single(
        obj,
        X_new,
    ):
        return obj["model"].predict_cure_probability(
            X_new,
        )

    @staticmethod
    def _average_results(results):
        total = np.array(results[0], dtype=float, copy=True)

        for r in results[1:]:
            total += r

        total /= len(results)

        return total

    def fit(
        self,
        X,
        t,
        delta,
    ):

        rng = np.random.default_rng(self.random_state)

        seeds = rng.integers(
            low=0,
            high=2**32 - 1,
            size=self.n_estimators,
        )

        self.models_ = Parallel(
            n_jobs=self.n_jobs,
            backend="loky",
            verbose=10,
        )(
            delayed(self._fit_single_estimator)(
                seed=seed,
                estimator=self.estimator,
                kernels=self.kernels,
                bootstrap=self.bootstrap,
                X=X,
                t=t,
                delta=delta,
                estimator_params=self.estimator_params,
            )
            for seed in seeds
        )

        return self

    def predict_survival(
        self,
        X_new,
        t_grid,
    ):

        survs = Parallel(
            n_jobs=self.n_jobs,
            backend="threading",
        )(
            delayed(self._predict_survival_single)(
                obj,
                X_new,
                t_grid,
            )
            for obj in self.models_
        )

        return self._average_results(survs)

    def predict_cure_probability(
        self,
        X_new,
    ):

        cures = Parallel(
            n_jobs=self.n_jobs,
            backend="threading",
        )(
            delayed(self._predict_cure_single)(
                obj,
                X_new,
            )
            for obj in self.models_
        )

        return self._average_results(cures)

    @property
    def kernel_counts_(self):
        counts = {}

        for obj in self.models_:
            kernel = obj["kernel"]
            counts[kernel] = counts.get(kernel, 0) + 1

        return counts
