import numpy as np
import os

from joblib import Parallel, delayed
from models.km_gg import GG_KM

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _fit_single_estimator(
    seed,
    kernels,
    bootstrap,
    X,
    t,
    delta,
    ggkm_params,
):

    rng = np.random.default_rng(seed)

    kernel = rng.choice(kernels)

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

    model = GG_KM(
        kernel=kernel,
        **ggkm_params,
    )

    model.fit(
        X_boot,
        t_boot,
        delta_boot,
    )

    return {
        "kernel": kernel,
        "model": model,
    }


def _predict_survival_single(
    obj,
    X_new,
    t_grid,
):

    return obj["model"].predict_survival(
        X_new,
        t_grid,
    )


def _predict_cure_single(
    obj,
    X_new,
):

    return obj["model"].predict_cure_probability(
        X_new,
    )


class GGKMKernelBagging:

    def __init__(
        self,
        kernels=None,
        n_estimators=50,
        bootstrap=True,
        random_state=None,
        n_jobs=-1,
        **ggkm_params,
    ):

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

        self.ggkm_params = ggkm_params

        self.models_ = []

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
            delayed(_fit_single_estimator)(
                seed=seed,
                kernels=self.kernels,
                bootstrap=self.bootstrap,
                X=X,
                t=t,
                delta=delta,
                ggkm_params=self.ggkm_params,
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
            delayed(_predict_survival_single)(
                obj,
                X_new,
                t_grid,
            )
            for obj in self.models_
        )

        survs = np.asarray(
            survs,
            dtype=float,
        )

        return np.mean(
            survs,
            axis=0,
        )

    def predict_cure_probability(
        self,
        X_new,
    ):

        cures = Parallel(
            n_jobs=self.n_jobs,
            backend="threading",
        )(
            delayed(_predict_cure_single)(
                obj,
                X_new,
            )
            for obj in self.models_
        )

        cures = np.asarray(
            cures,
            dtype=float,
        )

        return np.mean(
            cures,
            axis=0,
        )

    @property
    def kernel_counts_(self):

        counts = {}

        for obj in self.models_:

            kernel = obj["kernel"]

            counts[kernel] = counts.get(kernel, 0) + 1

        return counts


# import numpy as np

# from models.km_gg import GG_KM


# class GGKMKernelBagging:

#     def __init__(
#         self,
#         # estimator,
#         kernels=None,
#         n_estimators=50,
#         bootstrap=True,
#         random_state=None,
#         **ggkm_params,
#         # **estimator_params,
#     ):

#         self.kernels = (
#             kernels
#             if kernels is not None
#             else [
#                 "linear",
#                 "rbf",
#                 "laplacian",
#                 "exponential",
#                 "cauchy",
#             ]
#         )

#         # self.estimator = estimator
#         self.n_estimators = n_estimators
#         self.bootstrap = bootstrap
#         self.random_state = random_state

#         self.ggkm_params = ggkm_params
#         # self.estimator_params = estimator_params
#         self.models_ = []

#     def fit(
#         self,
#         X,
#         t,
#         delta,
#     ):

#         rng = np.random.default_rng(self.random_state)
#         self.models_ = []
#         n = len(t)
#         for _ in range(self.n_estimators):
#             kernel = rng.choice(self.kernels)
#             if self.bootstrap:
#                 idx = rng.choice(
#                     n,
#                     size=n,
#                     replace=True,
#                 )

#                 X_boot = X[idx]
#                 t_boot = t[idx]
#                 delta_boot = delta[idx]

#             else:

#                 X_boot = X
#                 t_boot = t
#                 delta_boot = delta

#             # model = self.estimator(
#             model = GG_KM(
#                 kernel=kernel,
#                 **self.ggkm_params,
#             )

#             model.fit(
#                 X_boot,
#                 t_boot,
#                 delta_boot,
#             )

#             self.models_.append(
#                 {
#                     "kernel": kernel,
#                     "model": model,
#                 }
#             )

#         return self

#     def predict_survival(
#         self,
#         X_new,
#         t_grid,
#     ):

#         survs = []

#         for obj in self.models_:
#             S = obj["model"].predict_survival(
#                 X_new,
#                 t_grid,
#             )
#             survs.append(S)

#         return np.mean(
#             survs,
#             axis=0,
#         )

#     def predict_cure_probability(
#         self,
#         X_new,
#     ):

#         cures = []

#         for obj in self.models_:
#             p = obj["model"].predict_cure_probability(X_new)
#             cures.append(p)

#         return np.mean(
#             cures,
#             axis=0,
#         )

#     @property
#     def kernel_counts_(self):

#         counts = {}
#         for obj in self.models_:
#             k = obj["kernel"]
#             counts[k] = counts.get(k, 0) + 1

#         return counts
