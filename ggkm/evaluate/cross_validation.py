import optuna
import numpy as np

from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from utils.metrics import uno_c_index_rmst, integrated_brier_score, auc_cure
from evaluate.simulated_data import simulate_pcm


def cross_validate_pcm(
    n,
    method,
    estimator,
    kernel="rbf",
    n_outer_splits=5,
    n_inner_splits=4,
    n_trials=20,
    t_grid_points=50,
    random_state=42,
    bagging=False,
    bagging_estimator=None,
    kernels=None,
    estimator_name=None,
    optimize_kernel_ranges=False,
    lambda_reg_search_bounds=(1e-5, 1.0),
    gamma_search_bounds=(1e-3, 10.0),
    degree_search_bounds=(2, 6),
    coef0_poly_search_bounds=(0.0, 5.0),
    coef0_sigmoid_search_bounds=(-5.0, 5.0),
):

    df = simulate_pcm(n=n, method=method, seed=random_state)

    X = df[["x1", "x2"]].to_numpy()
    t = df["time"].to_numpy()
    delta = df["event"].to_numpy()

    p_cure_true = 1 - df["pi_x"].to_numpy()

    outer_cv = KFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state)
    total_trials = n_outer_splits * n_trials

    model_name = "Bagging" if bagging else kernel

    pbar = tqdm(total=total_trials, desc=f"{model_name} nested CV", unit="trial")

    all_test_ibs = []
    all_test_cindex = []
    all_test_auc = []
    all_test_tpr = []
    all_test_fpr = []
    all_bias = []
    all_mse = []
    all_best_params = []

    optuna_callback = lambda study, trial: pbar.update(1)

    for _, (train_idx, test_idx) in enumerate(outer_cv.split(X)):

        X_outer_train, X_test = (X[train_idx], X[test_idx])
        t_outer_train, t_test = (t[train_idx], t[test_idx])
        d_outer_train, d_test = (delta[train_idx], delta[test_idx])
        p_cure_test_true = p_cure_true[test_idx]

        t_grid = np.linspace(
            np.percentile(t_outer_train, 5),
            np.percentile(t_outer_train, 95),
            t_grid_points,
        )

        inner_cv = KFold(
            n_splits=n_inner_splits, shuffle=True, random_state=random_state
        )

        def objective(trial):

            params = {}

            if bagging:
                params["n_estimators"] = trial.suggest_int(
                    "n_estimators",
                    50,
                    1000,
                )

                if optimize_kernel_ranges:
                    params.update(
                        _suggest_kernel_ranges(
                            trial,
                            lambda_reg_bounds=lambda_reg_search_bounds,
                            gamma_bounds=gamma_search_bounds,
                            degree_bounds=degree_search_bounds,
                            coef0_poly_bounds=coef0_poly_search_bounds,
                            coef0_sigmoid_bounds=coef0_sigmoid_search_bounds,
                        )
                    )

            else:
                params = {
                    "lambda_reg": trial.suggest_float(
                        "lambda_reg", 1e-5, 1.0, log=True
                    ),
                }

                if estimator_name == "binomial":
                    params["K_bin"] = trial.suggest_int("K_bin", 2, 1000)
                elif estimator_name == "bernoulli":
                    params["K_bin"] = trial.suggest_int("K_bin", 1, 1)

                if kernel != "linear":
                    params["gamma"] = trial.suggest_float("gamma", 1e-3, 10.0, log=True)

                if kernel == "polynomial":
                    params["degree"] = trial.suggest_int("degree", 2, 6)
                    params["coef0"] = trial.suggest_float("coef0", 0.0, 5.0)

                if kernel == "sigmoid":
                    params["coef0"] = trial.suggest_float("coef0", -5.0, 5.0)

            val_scores = []

            for tr_idx, val_idx in inner_cv.split(X_outer_train):

                X_tr = X_outer_train[tr_idx]
                X_val = X_outer_train[val_idx]

                t_tr = t_outer_train[tr_idx]
                t_val = t_outer_train[val_idx]

                d_tr = d_outer_train[tr_idx]
                d_val = d_outer_train[val_idx]

                t_grid_inner = np.linspace(
                    np.percentile(t_tr, 5), np.percentile(t_tr, 95), t_grid_points
                )

                try:
                    if bagging:

                        if bagging_estimator is None:
                            model = estimator(
                                kernels=kernels,
                                random_state=random_state,
                                **params,
                            )
                        else:
                            model = estimator(
                                estimator=bagging_estimator,
                                kernels=kernels,
                                random_state=random_state,
                                **params,
                            )

                    else:

                        model = estimator(kernel=kernel, **params)

                    model.fit(X_tr, t_tr, d_tr)
                    S_pred = model.predict_survival(X_val, t_grid_inner)
                    ibs = integrated_brier_score(
                        S_pred, t_val, d_val, t_tr, d_tr, t_grid_inner
                    )
                    val_scores.append(ibs)

                except Exception:
                    return 1.0

            return float(np.mean(val_scores))

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state)
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False,
            callbacks=[optuna_callback],
            n_jobs=16,
        )

        best_params = study.best_params
        all_best_params.append(best_params)

        if bagging:

            final_params = {"n_estimators": best_params["n_estimators"]}

            if optimize_kernel_ranges:
                final_params.update(_kernel_ranges_from_best_params(best_params))

            if bagging_estimator is None:
                final_model = estimator(
                    kernels=kernels,
                    random_state=random_state,
                    **final_params,
                )
            else:
                final_model = estimator(
                    estimator=bagging_estimator,
                    kernels=kernels,
                    random_state=random_state,
                    **final_params,
                )

        else:

            final_model = estimator(kernel=kernel, **best_params)

        final_model.fit(X_outer_train, t_outer_train, d_outer_train)
        S_test = final_model.predict_survival(X_test, t_grid)

        test_ibs = integrated_brier_score(
            S_test, t_test, d_test, t_outer_train, d_outer_train, t_grid
        )
        all_test_ibs.append(test_ibs)

        test_cindex = uno_c_index_rmst(
            S_pred=S_test,
            t_eval=t_test,
            delta_eval=d_test,
            t_train=t_outer_train,
            delta_train=d_outer_train,
            t_grid=t_grid,
            tau=t_grid[-1],
        )
        all_test_cindex.append(test_cindex)

        p_cure_hat = final_model.predict_cure_probability(X_test)
        test_auc, test_tpr, test_fpr, _ = auc_cure(p_cure_hat)

        all_test_auc.append(test_auc)
        all_test_tpr.append(test_tpr)
        all_test_fpr.append(test_fpr)

        fold_bias = np.mean(p_cure_hat - p_cure_test_true)
        fold_mse = np.mean((p_cure_hat - p_cure_test_true) ** 2)
        all_bias.append(fold_bias)
        all_mse.append(fold_mse)

    pbar.close()
    return {
        "test_ibs": all_test_ibs,
        "mean_ibs": float(np.mean(all_test_ibs)),
        "std_ibs": float(np.std(all_test_ibs)),
        "test_cindex": all_test_cindex,
        "mean_cindex": float(np.nanmean(all_test_cindex)),
        "std_cindex": float(np.nanstd(all_test_cindex)),
        "test_auc": all_test_auc,
        "mean_auc": float(np.nanmean(all_test_auc)),
        "std_auc": float(np.nanstd(all_test_auc)),
        "test_tpr": all_test_tpr,
        "test_fpr": all_test_fpr,
        "mean_bias": float(np.mean(all_bias)),
        "std_bias": float(np.std(all_bias)),
        "mean_mse": float(np.mean(all_mse)),
        "std_mse": float(np.std(all_mse)),
        "best_params": all_best_params,
    }


# def cross_validate_gg_km(
#     X,
#     t,
#     delta,
#     estimator,
#     kernel="rbf",
#     n_outer_splits=5,
#     n_inner_splits=4,
#     n_trials=50,
#     t_grid_points=50,
#     random_state=42,
#     bagging=False,
#     bagging_estimator=None,
#     kernels=None,
#     estimator_name=None,
# ):

#     outer_cv = KFold(
#         n_splits=n_outer_splits,
#         shuffle=True,
#         random_state=random_state,
#     )

#     all_test_ibs = []
#     all_test_cindex = []
#     all_test_auc = []
#     all_test_tpr = []
#     all_test_fpr = []
#     all_best_params = []

#     model_name = "Bagging" if bagging else kernel

#     outer_pbar = tqdm(
#         outer_cv.split(X),
#         total=n_outer_splits,
#         desc=f"{model_name} folds",
#     )

#     for fold, (train_idx, test_idx) in enumerate(outer_pbar):

#         X_outer_train, X_test = (
#             X[train_idx],
#             X[test_idx],
#         )

#         t_outer_train, t_test = (
#             t[train_idx],
#             t[test_idx],
#         )

#         d_outer_train, d_test = (
#             delta[train_idx],
#             delta[test_idx],
#         )

#         t_lo = np.percentile(
#             t_outer_train,
#             5,
#         )

#         t_hi = np.percentile(
#             t_outer_train,
#             95,
#         )

#         t_grid = np.linspace(
#             t_lo,
#             t_hi,
#             t_grid_points,
#         )

#         inner_cv = KFold(
#             n_splits=n_inner_splits,
#             shuffle=True,
#             random_state=random_state,
#         )

#         def objective(trial):

#             params = {}

#             if bagging:
#                 params["n_estimators"] = trial.suggest_int(
#                     "n_estimators",
#                     50,
#                     1000,
#                 )

#             else:
#                 params = {
#                     "lambda_reg": trial.suggest_float(
#                         "lambda_reg",
#                         1e-5,
#                         1.0,
#                         log=True,
#                     ),
#                 }

#                 if estimator_name == "binomial":
#                     params["K_bin"] = trial.suggest_int("K_bin", 2, 1000)
#                 elif estimator_name == "bernoulli":
#                     params["K_bin"] = trial.suggest_int("K_bin", 1, 1)

#                 if kernel in {
#                     "rbf",
#                     "gaussian",
#                     "laplacian",
#                     "exponential",
#                     "cauchy",
#                     "sigmoid",
#                 }:
#                     params["gamma"] = trial.suggest_float(
#                         "gamma",
#                         1e-3,
#                         10.0,
#                         log=True,
#                     )

#                 if kernel == "polynomial":
#                     params["gamma"] = trial.suggest_float(
#                         "gamma",
#                         1e-3,
#                         10.0,
#                         log=True,
#                     )

#                     params["degree"] = trial.suggest_int(
#                         "degree",
#                         2,
#                         6,
#                     )

#                     params["coef0"] = trial.suggest_float(
#                         "coef0",
#                         0.0,
#                         5.0,
#                     )

#                 if kernel == "sigmoid":
#                     params["coef0"] = trial.suggest_float(
#                         "coef0",
#                         -5.0,
#                         5.0,
#                     )

#             val_scores = []

#             for tr_idx, val_idx in inner_cv.split(X_outer_train):

#                 X_tr, X_val = (
#                     X_outer_train[tr_idx],
#                     X_outer_train[val_idx],
#                 )

#                 t_tr, t_val = (
#                     t_outer_train[tr_idx],
#                     t_outer_train[val_idx],
#                 )

#                 d_tr, d_val = (
#                     d_outer_train[tr_idx],
#                     d_outer_train[val_idx],
#                 )

#                 scaler = StandardScaler()

#                 X_tr_s = scaler.fit_transform(X_tr)
#                 X_val_s = scaler.transform(X_val)

#                 t_grid_inner = np.linspace(
#                     np.percentile(t_tr, 5),
#                     np.percentile(t_tr, 95),
#                     t_grid_points,
#                 )

#                 try:

#                     if bagging:

#                         if bagging_estimator is None:
#                             model = estimator(
#                                 kernels=kernels,
#                                 random_state=random_state,
#                                 **params,
#                             )
#                         else:
#                             model = estimator(
#                                 estimator=bagging_estimator,
#                                 kernels=kernels,
#                                 random_state=random_state,
#                                 **params,
#                             )

#                     else:

#                         model = estimator(
#                             kernel=kernel,
#                             **params,
#                         )

#                     model.fit(
#                         X_tr_s,
#                         t_tr,
#                         d_tr,
#                     )

#                     S_pred = model.predict_survival(
#                         X_val_s,
#                         t_grid_inner,
#                     )

#                     ibs = integrated_brier_score(
#                         S_pred,
#                         t_val,
#                         d_val,
#                         t_tr,
#                         d_tr,
#                         t_grid_inner,
#                     )

#                     val_scores.append(ibs)

#                 except Exception:

#                     return 1.0

#             return float(np.mean(val_scores))

#         trial_pbar = tqdm(
#             total=n_trials,
#             desc=f"Fold {fold + 1}",
#             leave=False,
#         )

#         callback = lambda study, trial: trial_pbar.update(1)

#         study = optuna.create_study(
#             direction="minimize",
#             sampler=optuna.samplers.TPESampler(seed=random_state),
#         )

#         try:

#             study.optimize(
#                 objective,
#                 n_trials=n_trials,
#                 show_progress_bar=False,
#                 callbacks=[callback],
#             )

#         finally:

#             trial_pbar.close()

#         best_params = study.best_params

#         scaler = StandardScaler()

#         X_outer_train_s = scaler.fit_transform(X_outer_train)
#         X_test_s = scaler.transform(X_test)

#         if bagging:

#             if bagging_estimator is None:
#                 final_model = estimator(
#                     kernels=kernels,
#                     random_state=random_state,
#                     **best_params,
#                 )
#             else:
#                 final_model = estimator(
#                     estimator=bagging_estimator,
#                     kernels=kernels,
#                     random_state=random_state,
#                     **best_params,
#                 )

#         else:

#             final_model = estimator(
#                 kernel=kernel,
#                 **best_params,
#             )

#         final_model.fit(
#             X_outer_train_s,
#             t_outer_train,
#             d_outer_train,
#         )

#         S_test = final_model.predict_survival(
#             X_test_s,
#             t_grid,
#         )

#         test_ibs = integrated_brier_score(
#             S_test,
#             t_test,
#             d_test,
#             t_outer_train,
#             d_outer_train,
#             t_grid,
#         )

#         test_cindex = uno_c_index_rmst(
#             S_pred=S_test,
#             t_eval=t_test,
#             delta_eval=d_test,
#             t_train=t_outer_train,
#             delta_train=d_outer_train,
#             t_grid=t_grid,
#             tau=t_grid[-1],
#         )

#         p_cure_hat = final_model.predict_cure_probability(X_test_s)
#         test_auc, test_tpr, test_fpr, _ = auc_cure(p_cure_hat)

#         all_test_ibs.append(test_ibs)
#         all_test_cindex.append(test_cindex)
#         all_test_auc.append(test_auc)
#         all_test_tpr.append(test_tpr)
#         all_test_fpr.append(test_fpr)
#         all_best_params.append(best_params)

#     return {
#         "test_ibs": all_test_ibs,
#         "mean_ibs": float(np.mean(all_test_ibs)),
#         "std_ibs": float(np.std(all_test_ibs)),
#         "test_cindex": all_test_cindex,
#         "mean_cindex": float(np.nanmean(all_test_cindex)),
#         "std_cindex": float(np.nanstd(all_test_cindex)),
#         "test_auc": all_test_auc,
#         "mean_auc": float(np.nanmean(all_test_auc)),
#         "std_auc": float(np.nanstd(all_test_auc)),
#         "test_tpr": all_test_tpr,
#         "test_fpr": all_test_fpr,
#         "best_params": all_best_params,
#     }


# def cross_validate_gg_km_breast_cancer(
#     df,
#     estimator,
#     preprocessor_factory,
#     kernel="rbf",
#     n_outer_splits=5,
#     n_inner_splits=4,
#     n_trials=50,
#     t_grid_points=50,
#     random_state=42,
#     bagging=False,
#     bagging_estimator=None,
#     kernels=None,
#     estimator_name=None,
# ):

#     df = df.reset_index(drop=True)

#     outer_cv = KFold(
#         n_splits=n_outer_splits,
#         shuffle=True,
#         random_state=random_state,
#     )

#     all_test_ibs = []
#     all_test_cindex = []
#     all_test_auc = []
#     all_test_tpr = []
#     all_test_fpr = []
#     all_best_params = []

#     model_name = "Bagging" if bagging else kernel

#     outer_pbar = tqdm(
#         outer_cv.split(df),
#         total=n_outer_splits,
#         desc=f"{model_name} folds",
#     )

#     for fold, (train_idx, test_idx) in enumerate(outer_pbar):

#         df_outer_train = df.iloc[train_idx].reset_index(drop=True)
#         df_test = df.iloc[test_idx].reset_index(drop=True)

#         outer_preprocessor = preprocessor_factory()
#         X_outer_train, t_outer_train, d_outer_train = outer_preprocessor.fit(
#             df_outer_train
#         )
#         X_test, t_test, d_test = outer_preprocessor.transform(df_test)

#         t_lo = np.percentile(t_outer_train, 5)
#         t_hi = np.percentile(t_outer_train, 95)
#         t_grid = np.linspace(t_lo, t_hi, t_grid_points)

#         inner_cv = KFold(
#             n_splits=n_inner_splits,
#             shuffle=True,
#             random_state=random_state,
#         )

#         inner_splits = []
#         for tr_idx, val_idx in inner_cv.split(df_outer_train):
#             df_tr = df_outer_train.iloc[tr_idx].reset_index(drop=True)
#             df_val = df_outer_train.iloc[val_idx].reset_index(drop=True)

#             inner_preprocessor = preprocessor_factory()
#             X_tr, t_tr, d_tr = inner_preprocessor.fit(df_tr)
#             X_val, t_val, d_val = inner_preprocessor.transform(df_val)

#             t_grid_inner = np.linspace(
#                 np.percentile(t_tr, 5),
#                 np.percentile(t_tr, 95),
#                 t_grid_points,
#             )
#             inner_splits.append((X_tr, t_tr, d_tr, X_val, t_val, d_val, t_grid_inner))

#         def objective(trial):

#             params = {}

#             if bagging:
#                 params["n_estimators"] = trial.suggest_int(
#                     "n_estimators",
#                     50,
#                     1000,
#                 )

#             else:
#                 params = {
#                     "lambda_reg": trial.suggest_float(
#                         "lambda_reg",
#                         1e-5,
#                         1.0,
#                         log=True,
#                     ),
#                 }

#                 if estimator_name == "binomial":
#                     params["K_bin"] = trial.suggest_int("K_bin", 2, 1000)
#                 elif estimator_name == "bernoulli":
#                     params["K_bin"] = trial.suggest_int("K_bin", 1, 1)

#                 if kernel in {
#                     "rbf",
#                     "gaussian",
#                     "laplacian",
#                     "exponential",
#                     "cauchy",
#                     "sigmoid",
#                 }:
#                     params["gamma"] = trial.suggest_float(
#                         "gamma",
#                         1e-3,
#                         10.0,
#                         log=True,
#                     )

#                 if kernel == "polynomial":
#                     params["gamma"] = trial.suggest_float(
#                         "gamma",
#                         1e-3,
#                         10.0,
#                         log=True,
#                     )

#                     params["degree"] = trial.suggest_int(
#                         "degree",
#                         2,
#                         6,
#                     )

#                     params["coef0"] = trial.suggest_float(
#                         "coef0",
#                         0.0,
#                         5.0,
#                     )

#                 if kernel == "sigmoid":
#                     params["coef0"] = trial.suggest_float(
#                         "coef0",
#                         -5.0,
#                         5.0,
#                     )

#             val_scores = []

#             for (
#                 X_tr,
#                 t_tr,
#                 d_tr,
#                 X_val,
#                 t_val,
#                 d_val,
#                 t_grid_inner,
#             ) in inner_splits:

#                 try:

#                     if bagging:

#                         if bagging_estimator is None:
#                             model = estimator(
#                                 kernels=kernels,
#                                 random_state=random_state,
#                                 **params,
#                             )
#                         else:
#                             model = estimator(
#                                 estimator=bagging_estimator,
#                                 kernels=kernels,
#                                 random_state=random_state,
#                                 **params,
#                             )

#                     else:

#                         model = estimator(
#                             kernel=kernel,
#                             **params,
#                         )

#                     model.fit(
#                         X_tr,
#                         t_tr,
#                         d_tr,
#                     )

#                     S_pred = model.predict_survival(
#                         X_val,
#                         t_grid_inner,
#                     )

#                     ibs = integrated_brier_score(
#                         S_pred,
#                         t_val,
#                         d_val,
#                         t_tr,
#                         d_tr,
#                         t_grid_inner,
#                     )

#                     val_scores.append(ibs)

#                 except Exception:

#                     return 1.0

#             return float(np.mean(val_scores))

#         trial_pbar = tqdm(
#             total=n_trials,
#             desc=f"Fold {fold + 1}",
#             leave=False,
#         )

#         callback = lambda study, trial: trial_pbar.update(1)

#         study = optuna.create_study(
#             direction="minimize",
#             sampler=optuna.samplers.TPESampler(seed=random_state),
#         )

#         try:

#             study.optimize(
#                 objective,
#                 n_trials=n_trials,
#                 show_progress_bar=False,
#                 callbacks=[callback],
#             )

#         finally:

#             trial_pbar.close()

#         best_params = study.best_params

#         if bagging:

#             if bagging_estimator is None:
#                 final_model = estimator(
#                     kernels=kernels,
#                     random_state=random_state,
#                     **best_params,
#                 )
#             else:
#                 final_model = estimator(
#                     estimator=bagging_estimator,
#                     kernels=kernels,
#                     random_state=random_state,
#                     **best_params,
#                 )

#         else:

#             final_model = estimator(
#                 kernel=kernel,
#                 **best_params,
#             )

#         final_model.fit(
#             X_outer_train,
#             t_outer_train,
#             d_outer_train,
#         )

#         S_test = final_model.predict_survival(
#             X_test,
#             t_grid,
#         )

#         test_ibs = integrated_brier_score(
#             S_test,
#             t_test,
#             d_test,
#             t_outer_train,
#             d_outer_train,
#             t_grid,
#         )

#         test_cindex = uno_c_index_rmst(
#             S_pred=S_test,
#             t_eval=t_test,
#             delta_eval=d_test,
#             t_train=t_outer_train,
#             delta_train=d_outer_train,
#             t_grid=t_grid,
#             tau=t_grid[-1],
#         )

#         p_cure_hat = final_model.predict_cure_probability(X_test)
#         test_auc, test_tpr, test_fpr, _ = auc_cure(p_cure_hat)

#         all_test_ibs.append(test_ibs)
#         all_test_cindex.append(test_cindex)
#         all_test_auc.append(test_auc)
#         all_test_tpr.append(test_tpr)
#         all_test_fpr.append(test_fpr)
#         all_best_params.append(best_params)

#     return {
#         "test_ibs": all_test_ibs,
#         "mean_ibs": float(np.mean(all_test_ibs)),
#         "std_ibs": float(np.std(all_test_ibs)),
#         "test_cindex": all_test_cindex,
#         "mean_cindex": float(np.nanmean(all_test_cindex)),
#         "std_cindex": float(np.nanstd(all_test_cindex)),
#         "test_auc": all_test_auc,
#         "mean_auc": float(np.nanmean(all_test_auc)),
#         "std_auc": float(np.nanstd(all_test_auc)),
#         "test_tpr": all_test_tpr,
#         "test_fpr": all_test_fpr,
#         "best_params": all_best_params,
#     }


def _suggest_range(trial, base_name, low_bound, high_bound, log=False, is_int=False):
    if is_int:
        lo = trial.suggest_int(f"{base_name}_low", low_bound, high_bound)
        hi = trial.suggest_int(f"{base_name}_high", lo, high_bound)
    else:
        lo = trial.suggest_float(f"{base_name}_low", low_bound, high_bound, log=log)
        hi = trial.suggest_float(f"{base_name}_high", lo, high_bound, log=log)

    return (lo, hi)


def _range_from_best_params(best_params, base_name):
    return (
        best_params[f"{base_name}_low"],
        best_params[f"{base_name}_high"],
    )


def _suggest_kernel_ranges(
    trial,
    lambda_reg_bounds,
    gamma_bounds,
    degree_bounds,
    coef0_poly_bounds,
    coef0_sigmoid_bounds,
):
    return {
        "lambda_reg_range": _suggest_range(
            trial, "lambda_reg_range", *lambda_reg_bounds, log=True
        ),
        "gamma_range": _suggest_range(trial, "gamma_range", *gamma_bounds, log=True),
        "degree_range": _suggest_range(
            trial, "degree_range", *degree_bounds, is_int=True
        ),
        "coef0_poly_range": _suggest_range(
            trial, "coef0_poly_range", *coef0_poly_bounds
        ),
        "coef0_sigmoid_range": _suggest_range(
            trial, "coef0_sigmoid_range", *coef0_sigmoid_bounds
        ),
    }


def _kernel_ranges_from_best_params(best_params):
    return {
        "lambda_reg_range": _range_from_best_params(best_params, "lambda_reg_range"),
        "gamma_range": _range_from_best_params(best_params, "gamma_range"),
        "degree_range": _range_from_best_params(best_params, "degree_range"),
        "coef0_poly_range": _range_from_best_params(best_params, "coef0_poly_range"),
        "coef0_sigmoid_range": _range_from_best_params(
            best_params, "coef0_sigmoid_range"
        ),
    }


def cross_validate_gg_km_breast_cancer(
    df,
    estimator,
    preprocessor_factory,
    kernel="rbf",
    n_outer_splits=5,
    n_inner_splits=4,
    n_trials=50,
    t_grid_points=50,
    random_state=42,
    bagging=False,
    bagging_estimator=None,
    kernels=None,
    estimator_name=None,
    optimize_kernel_ranges=False,
    lambda_reg_search_bounds=(1e-5, 1.0),
    gamma_search_bounds=(1e-3, 10.0),
    degree_search_bounds=(2, 6),
    coef0_poly_search_bounds=(0.0, 5.0),
    coef0_sigmoid_search_bounds=(-5.0, 5.0),
):

    df = df.reset_index(drop=True)

    outer_cv = KFold(
        n_splits=n_outer_splits,
        shuffle=True,
        random_state=random_state,
    )

    all_test_ibs = []
    all_test_cindex = []
    all_test_auc = []
    all_test_tpr = []
    all_test_fpr = []
    all_best_params = []

    model_name = "Bagging" if bagging else kernel

    outer_pbar = tqdm(
        outer_cv.split(df),
        total=n_outer_splits,
        desc=f"{model_name} folds",
    )

    for fold, (train_idx, test_idx) in enumerate(outer_pbar):

        df_outer_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        outer_preprocessor = preprocessor_factory()
        X_outer_train, t_outer_train, d_outer_train = outer_preprocessor.fit(
            df_outer_train
        )
        X_test, t_test, d_test = outer_preprocessor.transform(df_test)

        t_lo = np.percentile(t_outer_train, 5)
        t_hi = np.percentile(t_outer_train, 95)
        t_grid = np.linspace(t_lo, t_hi, t_grid_points)

        inner_cv = KFold(
            n_splits=n_inner_splits,
            shuffle=True,
            random_state=random_state,
        )

        inner_splits = []
        for tr_idx, val_idx in inner_cv.split(df_outer_train):
            df_tr = df_outer_train.iloc[tr_idx].reset_index(drop=True)
            df_val = df_outer_train.iloc[val_idx].reset_index(drop=True)

            inner_preprocessor = preprocessor_factory()
            X_tr, t_tr, d_tr = inner_preprocessor.fit(df_tr)
            X_val, t_val, d_val = inner_preprocessor.transform(df_val)

            t_grid_inner = np.linspace(
                np.percentile(t_tr, 5),
                np.percentile(t_tr, 95),
                t_grid_points,
            )
            inner_splits.append((X_tr, t_tr, d_tr, X_val, t_val, d_val, t_grid_inner))

        def objective(trial):

            params = {}

            if bagging:
                params["n_estimators"] = trial.suggest_int(
                    "n_estimators",
                    50,
                    1000,
                )

                if optimize_kernel_ranges:
                    params.update(
                        _suggest_kernel_ranges(
                            trial,
                            lambda_reg_bounds=lambda_reg_search_bounds,
                            gamma_bounds=gamma_search_bounds,
                            degree_bounds=degree_search_bounds,
                            coef0_poly_bounds=coef0_poly_search_bounds,
                            coef0_sigmoid_bounds=coef0_sigmoid_search_bounds,
                        )
                    )

            else:
                params = {
                    "lambda_reg": trial.suggest_float(
                        "lambda_reg",
                        1e-5,
                        1.0,
                        log=True,
                    ),
                }

                if estimator_name == "binomial":
                    params["K_bin"] = trial.suggest_int("K_bin", 2, 1000)
                elif estimator_name == "bernoulli":
                    params["K_bin"] = trial.suggest_int("K_bin", 1, 1)

                if kernel in {
                    "rbf",
                    "gaussian",
                    "laplacian",
                    "exponential",
                    "cauchy",
                    "sigmoid",
                }:
                    params["gamma"] = trial.suggest_float(
                        "gamma",
                        1e-3,
                        10.0,
                        log=True,
                    )

                if kernel == "polynomial":
                    params["gamma"] = trial.suggest_float(
                        "gamma",
                        1e-3,
                        10.0,
                        log=True,
                    )

                    params["degree"] = trial.suggest_int(
                        "degree",
                        2,
                        6,
                    )

                    params["coef0"] = trial.suggest_float(
                        "coef0",
                        0.0,
                        5.0,
                    )

                if kernel == "sigmoid":
                    params["coef0"] = trial.suggest_float(
                        "coef0",
                        -5.0,
                        5.0,
                    )

            val_scores = []

            for (
                X_tr,
                t_tr,
                d_tr,
                X_val,
                t_val,
                d_val,
                t_grid_inner,
            ) in inner_splits:

                try:

                    if bagging:

                        if bagging_estimator is None:
                            model = estimator(
                                kernels=kernels,
                                random_state=random_state,
                                **params,
                            )
                        else:
                            model = estimator(
                                estimator=bagging_estimator,
                                kernels=kernels,
                                random_state=random_state,
                                **params,
                            )

                    else:

                        model = estimator(
                            kernel=kernel,
                            **params,
                        )

                    model.fit(
                        X_tr,
                        t_tr,
                        d_tr,
                    )

                    S_pred = model.predict_survival(
                        X_val,
                        t_grid_inner,
                    )

                    ibs = integrated_brier_score(
                        S_pred,
                        t_val,
                        d_val,
                        t_tr,
                        d_tr,
                        t_grid_inner,
                    )

                    val_scores.append(ibs)

                except Exception:

                    return 1.0

            return float(np.mean(val_scores))

        trial_pbar = tqdm(
            total=n_trials,
            desc=f"Fold {fold + 1}",
            leave=False,
        )

        callback = lambda study, trial: trial_pbar.update(1)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )

        try:

            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=False,
                callbacks=[callback],
            )

        finally:

            trial_pbar.close()

        best_params = study.best_params

        if bagging:

            final_params = {"n_estimators": best_params["n_estimators"]}

            if optimize_kernel_ranges:
                final_params.update(_kernel_ranges_from_best_params(best_params))

            if bagging_estimator is None:
                final_model = estimator(
                    kernels=kernels,
                    random_state=random_state,
                    **final_params,
                )
            else:
                final_model = estimator(
                    estimator=bagging_estimator,
                    kernels=kernels,
                    random_state=random_state,
                    **final_params,
                )

        else:

            final_model = estimator(
                kernel=kernel,
                **best_params,
            )

        final_model.fit(
            X_outer_train,
            t_outer_train,
            d_outer_train,
        )

        S_test = final_model.predict_survival(
            X_test,
            t_grid,
        )

        test_ibs = integrated_brier_score(
            S_test,
            t_test,
            d_test,
            t_outer_train,
            d_outer_train,
            t_grid,
        )

        test_cindex = uno_c_index_rmst(
            S_pred=S_test,
            t_eval=t_test,
            delta_eval=d_test,
            t_train=t_outer_train,
            delta_train=d_outer_train,
            t_grid=t_grid,
            tau=t_grid[-1],
        )

        p_cure_hat = final_model.predict_cure_probability(X_test)
        test_auc, test_tpr, test_fpr, _ = auc_cure(p_cure_hat)

        all_test_ibs.append(test_ibs)
        all_test_cindex.append(test_cindex)
        all_test_auc.append(test_auc)
        all_test_tpr.append(test_tpr)
        all_test_fpr.append(test_fpr)
        all_best_params.append(best_params)

    return {
        "test_ibs": all_test_ibs,
        "mean_ibs": float(np.mean(all_test_ibs)),
        "std_ibs": float(np.std(all_test_ibs)),
        "test_cindex": all_test_cindex,
        "mean_cindex": float(np.nanmean(all_test_cindex)),
        "std_cindex": float(np.nanstd(all_test_cindex)),
        "test_auc": all_test_auc,
        "mean_auc": float(np.nanmean(all_test_auc)),
        "std_auc": float(np.nanstd(all_test_auc)),
        "test_tpr": all_test_tpr,
        "test_fpr": all_test_fpr,
        "best_params": all_best_params,
    }
