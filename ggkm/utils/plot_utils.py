import numpy as np

import matplotlib.pyplot as plt
from utils.metrics import kaplan_meier
from sklearn.model_selection import train_test_split


def plot_survival_curves_gb(
    df,
    results,
    model_class,
    preprocessor,
    distribution_name,
    test_size=0.2,
    random_state=42,
    n_grid=300,
):
    df = df.reset_index(drop=True)

    best_fold = results.sort_values("cindex", ascending=False).iloc[0]
    best_params = best_fold["best_params"]

    print(f"Best fold C-index: {best_fold['cindex']:.5f}")
    print(f"Selected hyperparameters: {best_params}")

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["delta"],
        random_state=random_state,
    )

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

    # Predict survival curves
    t_grid_test = np.linspace(
        np.min(t_test),
        np.max(t_test),
        n_grid,
    )

    S_mean_test = model.predict_survival(
        X_test,
        t_grid_test,
    ).mean(axis=0)

    # Kaplan-Meier curve
    km_t_test, km_s_test = kaplan_meier(
        t_test,
        delta_test,
    )

    # Plot
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
        f"../theses_writing/images/"
        f"survival_curve_ggkm_testset_boost_{distribution_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return model, best_params


import numpy as np

import matplotlib.pyplot as plt
from utils.metrics import kaplan_meier
from sklearn.model_selection import train_test_split


def plot_survival_curves_ggkm(
    df,
    results,
    model_class,
    preprocessor,
    distribution_name,
    test_size=0.2,
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

    print(f"Selected kernel: {best_kernel}")
    print(f"Mean C-index: " f"{kernel_selection.iloc[0]['cindex_mean']:.5f}")

    best_kernel_results = results[results["kernel"] == best_kernel]
    best_fold = best_kernel_results.sort_values("cindex", ascending=False).iloc[0]
    best_params = best_fold["best_params"]

    print(f"Best fold C-index: {best_fold['cindex']:.5f}")
    print(f"Selected hyperparameters: {best_params}")

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["delta"],
        random_state=random_state,
    )

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

    # Kaplan-Meier curve
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

    plt.savefig(
        f"../theses_writing/images/"
        f"survival_curve_ggkm_testset_{distribution_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return model, best_kernel, best_params
