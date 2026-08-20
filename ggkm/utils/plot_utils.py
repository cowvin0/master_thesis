import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from utils.metrics import kaplan_meier
from sklearn.model_selection import StratifiedKFold


def plot_survival_curves_gb(
    df,
    results,
    model_class,
    preprocessor,
    distribution_name,
    random_state=42,
    n_grid=300,
):
    df = df.reset_index(drop=True)

    best_fold = results.sort_values("cindex", ascending=False).iloc[0]
    best_params = best_fold["best_params"]

    print(f"Best fold C-index: {best_fold['cindex']:.5f}")
    print(f"Selected hyperparameters: {best_params}")

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    outer_splits = list(outer_cv.split(df, df["delta"]))
    best_fold_number = best_fold.name
    train_idx, test_idx = outer_splits[best_fold_number]

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    X_train, t_train, delta_train = preprocessor.fit(df_train)
    X_test, t_test, delta_test = preprocessor.transform(df_test)

    model = model_class(
        random_state=random_state,
        **best_params,
    )

    model.fit(
        X_train,
        t_train,
        delta_train,
    )

    t_grid_test = np.linspace(
        np.min(t_test),
        np.max(t_test),
        n_grid,
    )

    S_mean_test = model.predict_survival(
        X_test,
        t_grid_test,
    ).mean(axis=0)

    km_t_test, km_s_test = kaplan_meier(
        t_test,
        delta_test,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.step(
        km_t_test,
        km_s_test,
        where="post",
        linewidth=2,
        label="KM",
    )

    ax.plot(
        t_grid_test,
        S_mean_test,
        linewidth=2,
        linestyle="--",
        label=f"GG-GB (test, {model_class.__name__.replace('GG', '')})",
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel("t")
    ax.set_ylabel("S(t)")
    ax.set_title("Kaplan-Meier vs GG-GB survival curves")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"../theses_writing/"
        f"survival_curve_ggkm_testset_boost_{distribution_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return model, best_params, train_idx, test_idx


def plot_survival_curves_ggkm(
    df,
    results,
    model_class,
    preprocessor,
    random_state=42,
    n_grid=300,
):
    df = df.reset_index(drop=True)

    kernel_selection = (
        results.groupby("kernel", as_index=False)
        .agg(
            cindex_mean=("cindex", "mean"),
            std_cindex=("cindex", "std"),
        )
        .sort_values(
            ["cindex_mean", "std_cindex"],
            ascending=[False, True],
        )
    )

    best_kernel = kernel_selection.iloc[0]["kernel"]
    best_kernel_results = results[results["kernel"] == best_kernel]
    best_fold = best_kernel_results.sort_values("cindex", ascending=False).iloc[0]
    best_params = best_fold["best_params"]
    best_fold_number = best_fold.name

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    outer_splits = list(
        outer_cv.split(
            df,
            df["delta"],
        )
    )

    train_idx, test_idx = outer_splits[best_fold_number]

    print(f"Training observations: {len(train_idx)}")
    print(f"Test observations: {len(test_idx)}")

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    X_train, t_train, delta_train = preprocessor.fit(df_train)
    X_test, t_test, delta_test = preprocessor.transform(df_test)

    model = model_class(
        kernel=best_kernel,
        **best_params,
    )

    model.fit(
        X_train,
        t_train,
        delta_train,
    )

    t_grid_test = np.linspace(
        np.min(t_test),
        np.max(t_test),
        n_grid,
    )

    S_mean_test = model.predict_survival(
        X_test,
        t_grid_test,
    ).mean(axis=0)

    km_t_test, km_s_test = kaplan_meier(
        t_test,
        delta_test,
    )

    _, ax = plt.subplots(figsize=(8, 6))

    ax.step(
        km_t_test,
        km_s_test,
        where="post",
        linewidth=2,
        label="KM",
    )

    ax.plot(
        t_grid_test,
        S_mean_test,
        linewidth=2,
        linestyle="--",
        label=f"GG-KM (test, {best_kernel})",
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel("t")
    ax.set_ylabel("S(t)")
    ax.set_title("Kaplan-Meier vs GG-KM survival curves")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    return model, best_kernel, best_params
