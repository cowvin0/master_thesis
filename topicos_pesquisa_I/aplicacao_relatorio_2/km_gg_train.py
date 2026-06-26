import optuna
import numpy as np
import pandas as pd
import warnings

from gg_boosting import PTCMBoost
from utils import integrated_brier_score
from km_gg import GG_KM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def cross_validate_gg_km(
    X,
    t,
    delta,
    kernel="rbf",
    n_outer_splits=5,
    n_inner_splits=4,
    n_trials=50,
    t_grid_points=50,
    random_state=42,
):

    outer_cv = KFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state)

    all_test_ibs = []
    all_best_params = []

    print(f"{'='*60}")
    print(f" Nested CV: {n_outer_splits} outer × {n_inner_splits} inner folds")
    print(f" Kernel: {kernel}")
    print(f" Optuna trials per fold: {n_trials}")
    print(f"{'='*60}\n")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):

        print(f"─── Outer fold {fold_idx + 1}/{n_outer_splits} ", end="", flush=True)

        X_outer_train, X_test = X[train_idx], X[test_idx]
        t_outer_train, t_test = t[train_idx], t[test_idx]
        d_outer_train, d_test = delta[train_idx], delta[test_idx]

        t_lo = np.percentile(t_outer_train, 5)
        t_hi = np.percentile(t_outer_train, 95)
        t_grid = np.linspace(t_lo, t_hi, t_grid_points)

        inner_cv = KFold(
            n_splits=n_inner_splits, shuffle=True, random_state=random_state
        )

        def objective(trial):

            params = {
                "lambda_reg": trial.suggest_float("lambda_reg", 1e-5, 1.0, log=True)
            }

            if kernel in {
                "rbf",
                "gaussian",
                "laplacian",
                "exponential",
                "cauchy",
                "sigmoid",
            }:
                params["gamma"] = trial.suggest_float("gamma", 1e-3, 10.0, log=True)

            if kernel == "polynomial":
                params["gamma"] = trial.suggest_float("gamma", 1e-3, 10.0, log=True)

                params["degree"] = trial.suggest_int("degree", 2, 6)

                params["coef0"] = trial.suggest_float("coef0", 0.0, 5.0)

            if kernel == "sigmoid":
                params["coef0"] = trial.suggest_float("coef0", -5.0, 5.0)

            val_scores = []

            for tr_idx, val_idx in inner_cv.split(X_outer_train):

                X_tr, X_val = (X_outer_train[tr_idx], X_outer_train[val_idx])

                t_tr, t_val = (t_outer_train[tr_idx], t_outer_train[val_idx])

                d_tr, d_val = (d_outer_train[tr_idx], d_outer_train[val_idx])

                scaler = StandardScaler()

                X_tr_s = scaler.fit_transform(X_tr)
                X_val_s = scaler.transform(X_val)

                t_grid_inner = np.linspace(
                    np.percentile(t_tr, 5), np.percentile(t_tr, 95), t_grid_points
                )

                try:

                    model = GG_KM(kernel=kernel, **params)

                    model.fit(X_tr_s, t_tr, d_tr)

                    S_pred = model.predict_survival(X_val_s, t_grid_inner)

                    ibs = integrated_brier_score(
                        S_pred, t_val, d_val, t_tr, d_tr, t_grid_inner
                    )

                    val_scores.append(ibs)

                except Exception:
                    return 1.0

            return float(np.mean(val_scores))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params

        print(f"→ best val IBS = {study.best_value:.4f}")

        scaler = StandardScaler()

        X_outer_train_s = scaler.fit_transform(X_outer_train)

        X_test_s = scaler.transform(X_test)

        final_model = GG_KM(kernel=kernel, **best_params)

        final_model.fit(X_outer_train_s, t_outer_train, d_outer_train)

        S_test = final_model.predict_survival(X_test_s, t_grid)

        test_ibs = integrated_brier_score(
            S_test, t_test, d_test, t_outer_train, d_outer_train, t_grid
        )

        print(f"           test IBS = {test_ibs:.4f}\n")

        all_test_ibs.append(test_ibs)
        all_best_params.append(best_params)

    print(f"{'='*60}")
    print(" Cross-Validation Results")
    print(f"{'='*60}")
    print(f" IBS per fold : {[f'{v:.4f}' for v in all_test_ibs]}")
    print(f" Mean IBS     : {np.mean(all_test_ibs):.4f}")
    print(f" Std  IBS     : {np.std(all_test_ibs):.4f}")
    print(f"{'='*60}\n")

    return {
        "test_ibs": all_test_ibs,
        "mean_ibs": float(np.mean(all_test_ibs)),
        "std_ibs": float(np.std(all_test_ibs)),
        "best_params": all_best_params,
    }


if __name__ == "__main__":
    lung_df = (
        pd.read_csv("../data/lung.csv")
        .drop(columns=["inst", "sex", "meal.cal", "wt.loss"])
        .assign(status=lambda x: x.status.replace([1, 2], [0, 1]))
        .fillna(0)
    )

    y = lung_df[["time", "status"]]

    X_raw = lung_df.drop(columns=y.columns.tolist()).to_numpy()

    t = y["time"].to_numpy().astype(float)
    delta = y["status"].to_numpy().astype(float)

    kernels = [
        "linear",
        "rbf",
        "laplacian",
        "exponential",
        "cauchy",
        "sigmoid",
        "polynomial",
    ]

    all_results = {}

    for kernel in kernels:

        print("\n")
        print("=" * 80)
        print(f"Kernel: {kernel}")
        print("=" * 80)

        results = cross_validate_gg_km(
            X_raw,
            t,
            delta,
            n_outer_splits=5,
            n_inner_splits=4,
            n_trials=20,
            t_grid_points=50,
            random_state=42,
            kernel=kernel,
        )

        all_results[kernel] = results

    print("\n")
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for kernel, res in all_results.items():
        print(
            f"{kernel:12s} "
            f"Mean IBS = {res['mean_ibs']:.4f} "
            f"± {res['std_ibs']:.4f}"
        )
