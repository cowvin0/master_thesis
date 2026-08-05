import numpy as np
import os

from copy import deepcopy
from joblib import Parallel, delayed
from ggkm.models.km_gg import GGPoisson

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


# import numpy as np
# import os

# from copy import deepcopy
# from joblib import Parallel, delayed
# from models.km_gg import GGPoisson

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"


# class GGKMKernelBagging:
#     def __init__(
#         self,
#         estimator=None,
#         kernels=None,
#         n_estimators=50,
#         bootstrap=True,
#         random_state=None,
#         n_jobs=-1,
#         sample_kernel=True,
#         sample_hyperparams=True,
#         lambda_reg_range=None,
#         gamma_range=None,
#         degree_range=None,
#         coef0_poly_range=None,
#         coef0_sigmoid_range=None,
#         **estimator_params,
#     ):
#         self.estimator = estimator

#         if kernels is None:
#             kernels = [
#                 "linear",
#                 "rbf",
#                 "laplacian",
#                 "exponential",
#                 "cauchy",
#             ]
#         elif isinstance(kernels, str):
#             kernels = [kernels]

#         self.kernels = kernels
#         # self.kernels = (
#         #     kernels
#         #     if kernels is not None
#         #     else [
#         #         "linear",
#         #         "rbf",
#         #         "laplacian",
#         #         "exponential",
#         #         "cauchy",
#         #     ]
#         # )

#         self.n_estimators = n_estimators
#         self.bootstrap = bootstrap
#         self.random_state = random_state
#         self.n_jobs = n_jobs

#         self.sample_kernel = sample_kernel
#         self.sample_hyperparams = sample_hyperparams

#         self.lambda_reg_range = lambda_reg_range
#         self.gamma_range = gamma_range
#         self.degree_range = degree_range
#         self.coef0_poly_range = coef0_poly_range
#         self.coef0_sigmoid_range = coef0_sigmoid_range

#         self.estimator_params = estimator_params
#         self.models_ = []

#     @staticmethod
#     def _log_uniform(rng, low, high):
#         if low <= 0 or high <= 0:
#             raise ValueError("Log-uniform sampling requires positive bounds.")
#         if low > high:
#             raise ValueError("Invalid range: low must be <= high.")
#         if low == high:
#             return float(low)

#         return float(np.exp(rng.uniform(np.log(low), np.log(high))))

#     @staticmethod
#     def _uniform(rng, low, high):
#         if low > high:
#             raise ValueError("Invalid range: low must be <= high.")
#         if low == high:
#             return float(low)

#         return float(rng.uniform(low, high))

#     @staticmethod
#     def _int_uniform(rng, low, high):
#         if low > high:
#             raise ValueError("Invalid integer range: low must be <= high.")
#         return int(rng.integers(low, high + 1))

#     def _sample_params_for_kernel(self, rng, kernel_name):
#         params = dict(self.estimator_params)

#         if self.sample_hyperparams and self.lambda_reg_range is not None:
#             params["lambda_reg"] = self._log_uniform(
#                 rng,
#                 self.lambda_reg_range[0],
#                 self.lambda_reg_range[1],
#             )

#         if (
#             self.sample_hyperparams
#             and kernel_name
#             in {
#                 "rbf",
#                 "gaussian",
#                 "laplacian",
#                 "exponential",
#                 "cauchy",
#                 "sigmoid",
#                 "polynomial",
#             }
#             and self.gamma_range is not None
#         ):
#             params["gamma"] = self._log_uniform(
#                 rng,
#                 self.gamma_range[0],
#                 self.gamma_range[1],
#             )

#         if self.sample_hyperparams and kernel_name == "polynomial":
#             if self.degree_range is not None:
#                 params["degree"] = self._int_uniform(
#                     rng,
#                     self.degree_range[0],
#                     self.degree_range[1],
#                 )

#             if self.coef0_poly_range is not None:
#                 params["coef0"] = self._uniform(
#                     rng,
#                     self.coef0_poly_range[0],
#                     self.coef0_poly_range[1],
#                 )

#         if self.sample_hyperparams and kernel_name == "sigmoid":
#             if self.coef0_sigmoid_range is not None:
#                 params["coef0"] = self._uniform(
#                     rng,
#                     self.coef0_sigmoid_range[0],
#                     self.coef0_sigmoid_range[1],
#                 )

#         return params

#     @staticmethod
#     def _build_model_from_template(
#         estimator,
#         kernel_name,
#         params,
#     ):
#         if estimator is None:
#             return GGPoisson(kernel=kernel_name, **params)

#         if isinstance(estimator, type):
#             try:
#                 return estimator(kernel=kernel_name, **params)
#             except TypeError:
#                 return estimator(**params)

#         model = deepcopy(estimator)

#         if hasattr(model, "kernel"):
#             model.kernel = kernel_name

#         for key, value in params.items():
#             if hasattr(model, key):
#                 setattr(model, key, value)

#         return model

#     @staticmethod
#     def _fit_single_estimator(
#         seed,
#         estimator,
#         kernels,
#         bootstrap,
#         sample_kernel,
#         sample_hyperparams,
#         X,
#         t,
#         delta,
#         estimator_params,
#         lambda_reg_range,
#         gamma_range,
#         degree_range,
#         coef0_poly_range,
#         coef0_sigmoid_range,
#     ):
#         rng = np.random.default_rng(seed)

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

#         if estimator is not None and not sample_kernel:
#             kernel_name = getattr(estimator, "kernel", None)
#             if kernel_name is None:
#                 kernel_name = rng.choice(kernels)
#         else:
#             kernel_name = rng.choice(kernels)

#         params = dict(estimator_params)

#         if sample_hyperparams and lambda_reg_range is not None:
#             params["lambda_reg"] = GGKMKernelBagging._log_uniform(
#                 rng,
#                 lambda_reg_range[0],
#                 lambda_reg_range[1],
#             )

#         if (
#             sample_hyperparams
#             and kernel_name
#             in {
#                 "rbf",
#                 "gaussian",
#                 "laplacian",
#                 "exponential",
#                 "cauchy",
#                 "sigmoid",
#                 "polynomial",
#             }
#             and gamma_range is not None
#         ):
#             params["gamma"] = GGKMKernelBagging._log_uniform(
#                 rng,
#                 gamma_range[0],
#                 gamma_range[1],
#             )

#         if sample_hyperparams and kernel_name == "polynomial":
#             if degree_range is not None:
#                 params["degree"] = GGKMKernelBagging._int_uniform(
#                     rng,
#                     degree_range[0],
#                     degree_range[1],
#                 )

#             if coef0_poly_range is not None:
#                 params["coef0"] = GGKMKernelBagging._uniform(
#                     rng,
#                     coef0_poly_range[0],
#                     coef0_poly_range[1],
#                 )

#         if sample_hyperparams and kernel_name == "sigmoid":
#             if coef0_sigmoid_range is not None:
#                 params["coef0"] = GGKMKernelBagging._uniform(
#                     rng,
#                     coef0_sigmoid_range[0],
#                     coef0_sigmoid_range[1],
#                 )

#         model = GGKMKernelBagging._build_model_from_template(
#             estimator=estimator,
#             kernel_name=kernel_name,
#             params=params,
#         )

#         model.fit(
#             X_boot,
#             t_boot,
#             delta_boot,
#         )

#         return {
#             "kernel": kernel_name,
#             "params": params,
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

#     @staticmethod
#     def _average_results(results):
#         total = np.array(results[0], dtype=float, copy=True)

#         for r in results[1:]:
#             total += r

#         total /= len(results)

#         return total

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
#                 estimator=self.estimator,
#                 kernels=self.kernels,
#                 bootstrap=self.bootstrap,
#                 sample_kernel=self.sample_kernel,
#                 sample_hyperparams=self.sample_hyperparams,
#                 X=X,
#                 t=t,
#                 delta=delta,
#                 estimator_params=self.estimator_params,
#                 lambda_reg_range=self.lambda_reg_range,
#                 gamma_range=self.gamma_range,
#                 degree_range=self.degree_range,
#                 coef0_poly_range=self.coef0_poly_range,
#                 coef0_sigmoid_range=self.coef0_sigmoid_range,
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

#         return self._average_results(survs)

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

#         return self._average_results(cures)

#     @property
#     def kernel_counts_(self):
#         counts = {}

#         for obj in self.models_:
#             kernel = obj["kernel"]
#             counts[kernel] = counts.get(kernel, 0) + 1

#         return counts

#     @property
#     def sampled_params_(self):
#         return [
#             {
#                 "kernel": obj["kernel"],
#                 **obj["params"],
#             }
#             for obj in self.models_
#         ]
