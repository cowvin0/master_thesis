import numpy as np
import os

from joblib import Parallel, delayed
from models.km_gg import GG_KM

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


class GGKMKernelBagging:

    def __init__(
        self,
        kernels=None,
        n_estimators=50,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        lambda_range=(1e-5, 1.0),
        gamma_range=(1e-3, 10.0),
        degree_range=(2, 6),
        coef0_poly_range=(0.0, 5.0),
        coef0_sigmoid_range=(-5.0, 5.0),
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

        self.lambda_range = lambda_range
        self.gamma_range = gamma_range
        self.degree_range = degree_range
        self.coef0_poly_range = coef0_poly_range
        self.coef0_sigmoid_range = coef0_sigmoid_range

        self.ggkm_params = ggkm_params

        self.models_ = []

    @staticmethod
    def _sample_hyperparameters(
        rng,
        kernel,
        base_params,
        lambda_range,
        gamma_range,
        degree_range,
        coef0_poly_range,
        coef0_sigmoid_range,
    ):

        params = dict(base_params)
        params["lambda_reg"] = 10 ** rng.uniform(
            np.log10(lambda_range[0]),
            np.log10(lambda_range[1]),
        )

        if kernel in {
            "rbf",
            "gaussian",
            "laplacian",
            "exponential",
            "cauchy",
            "sigmoid",
        }:
            params["gamma"] = 10 ** rng.uniform(
                np.log10(gamma_range[0]),
                np.log10(gamma_range[1]),
            )

        if kernel == "polynomial":
            params["gamma"] = 10 ** rng.uniform(
                np.log10(gamma_range[0]),
                np.log10(gamma_range[1]),
            )
            params["degree"] = rng.integers(
                degree_range[0],
                degree_range[1] + 1,
            )
            params["coef0"] = rng.uniform(
                coef0_poly_range[0],
                coef0_poly_range[1],
            )

        if kernel == "sigmoid":
            params["coef0"] = rng.uniform(
                coef0_sigmoid_range[0],
                coef0_sigmoid_range[1],
            )

        return params

    @staticmethod
    def _fit_single_estimator(
        seed,
        kernels,
        bootstrap,
        X,
        t,
        delta,
        ggkm_params,
        lambda_range,
        gamma_range,
        degree_range,
        coef0_poly_range,
        coef0_sigmoid_range,
    ):

        rng = np.random.default_rng(seed)
        kernel = rng.choice(kernels)
        params = GGKMKernelBagging._sample_hyperparameters(
            rng=rng,
            kernel=kernel,
            base_params=ggkm_params,
            lambda_range=lambda_range,
            gamma_range=gamma_range,
            degree_range=degree_range,
            coef0_poly_range=coef0_poly_range,
            coef0_sigmoid_range=coef0_sigmoid_range,
        )

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
            **params,
        )

        model.fit(
            X_boot,
            t_boot,
            delta_boot,
        )

        return {
            "kernel": kernel,
            "params": params,
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
                kernels=self.kernels,
                bootstrap=self.bootstrap,
                X=X,
                t=t,
                delta=delta,
                ggkm_params=self.ggkm_params,
                lambda_range=self.lambda_range,
                gamma_range=self.gamma_range,
                degree_range=self.degree_range,
                coef0_poly_range=self.coef0_poly_range,
                coef0_sigmoid_range=self.coef0_sigmoid_range,
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

        return np.mean(
            np.asarray(survs, dtype=float),
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
            delayed(self._predict_cure_single)(
                obj,
                X_new,
            )
            for obj in self.models_
        )

        return np.mean(
            np.asarray(cures, dtype=float),
            axis=0,
        )

    @property
    def kernel_counts_(self):

        counts = {}
        for obj in self.models_:
            kernel = obj["kernel"]
            counts[kernel] = (
                counts.get(
                    kernel,
                    0,
                )
                + 1
            )

        return counts

    @property
    def sampled_parameters_(self):

        return [
            {
                "kernel": obj["kernel"],
                **obj["params"],
            }
            for obj in self.models_
        ]


# import numpy as np
# import os

# from joblib import Parallel, delayed
# from models.km_gg import GG_KM

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"


# class GGKMKernelBagging:
#     def __init__(
#         self,
#         kernels=None,
#         n_estimators=50,
#         bootstrap=True,
#         random_state=None,
#         n_jobs=-1,
#         **ggkm_params,
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
#         self.n_estimators = n_estimators
#         self.bootstrap = bootstrap
#         self.random_state = random_state
#         self.n_jobs = n_jobs
#         self.ggkm_params = ggkm_params
#         self.models_ = []

#     @staticmethod
#     def _fit_single_estimator(
#         seed,
#         kernels,
#         bootstrap,
#         X,
#         t,
#         delta,
#         ggkm_params,
#     ):
#         rng = np.random.default_rng(seed)
#         kernel = rng.choice(kernels)
#         n = len(t)
#         if bootstrap:
#             idx = rng.choice(
#                 n,
#                 size=n,
#                 replace=True,
#             )
#             X_boot = X[idx]
#             t_boot = t[idx]
#             delta_boot = delta[idx]
#         else:
#             X_boot = X
#             t_boot = t
#             delta_boot = delta
#         model = GG_KM(
#             kernel=kernel,
#             **ggkm_params,
#         )
#         model.fit(
#             X_boot,
#             t_boot,
#             delta_boot,
#         )
#         return {
#             "kernel": kernel,
#             "model": model,
#         }

#     @staticmethod
#     def _predict_survival_single(
#         obj,
#         X_new,
#         t_grid,
#     ):
#         return obj["model"].predict_survival(
#             X_new,
#             t_grid,
#         )

#     @staticmethod
#     def _predict_cure_single(
#         obj,
#         X_new,
#     ):
#         return obj["model"].predict_cure_probability(
#             X_new,
#         )

#     def fit(
#         self,
#         X,
#         t,
#         delta,
#     ):
#         rng = np.random.default_rng(self.random_state)
#         seeds = rng.integers(
#             low=0,
#             high=2**32 - 1,
#             size=self.n_estimators,
#         )
#         self.models_ = Parallel(
#             n_jobs=self.n_jobs,
#             backend="loky",
#             verbose=10,
#         )(
#             delayed(self._fit_single_estimator)(
#                 seed=seed,
#                 kernels=self.kernels,
#                 bootstrap=self.bootstrap,
#                 X=X,
#                 t=t,
#                 delta=delta,
#                 ggkm_params=self.ggkm_params,
#             )
#             for seed in seeds
#         )
#         return self

#     def predict_survival(
#         self,
#         X_new,
#         t_grid,
#     ):
#         survs = Parallel(
#             n_jobs=self.n_jobs,
#             backend="threading",
#         )(
#             delayed(self._predict_survival_single)(
#                 obj,
#                 X_new,
#                 t_grid,
#             )
#             for obj in self.models_
#         )
#         survs = np.asarray(
#             survs,
#             dtype=float,
#         )
#         return np.mean(
#             survs,
#             axis=0,
#         )

#     def predict_cure_probability(
#         self,
#         X_new,
#     ):
#         cures = Parallel(
#             n_jobs=self.n_jobs,
#             backend="threading",
#         )(
#             delayed(self._predict_cure_single)(
#                 obj,
#                 X_new,
#             )
#             for obj in self.models_
#         )
#         cures = np.asarray(
#             cures,
#             dtype=float,
#         )
#         return np.mean(
#             cures,
#             axis=0,
#         )

#     @property
#     def kernel_counts_(self):
#         counts = {}
#         for obj in self.models_:
#             kernel = obj["kernel"]
#             counts[kernel] = counts.get(kernel, 0) + 1
#         return counts
