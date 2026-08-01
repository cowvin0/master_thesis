import os
import time
import ast
import re
import pandas as pd

from pathlib import Path
from ggkm.models.km_gg_bin import GGBinomial
from ggkm.models.km_gg import GGPoisson
from ggkm.models.km_gg_bn import GGNB
from ggkm.models.bagging import GGKMKernelBagging
from ggkm.evaluate.cross_validation import cross_validate_gg_km

MODELS = {
    "bernoulli": GGBinomial,
    "binomial": GGBinomial,
    "poisson": GGPoisson,
    "bn": GGNB,
}


OUTPUT_DIR = Path("data/melanoma_results")


def parse_cell(x):
    s = str(x).strip()
    s = re.sub(
        r"np\.float64\(([^()]*)\)",
        r"\1",
        s,
    )

    return ast.literal_eval(s)


def build_experiments():

    experiments = []

    for model_name in MODELS:
        experiments.append(
            {
                "model_name": model_name,
            }
        )

    return experiments


def get_experiment(
    experiments,
    task_id,
):

    if task_id >= len(experiments):
        raise ValueError(f"Invalid PBS_ARRAY_INDEX={task_id}")

    return experiments[task_id]


def select_best_model(model_name):

    file = f"ggkm/data/melanoma_results/melanoma_results_em_{model_name}.csv"

    df = pd.read_csv(file)

    best_params = df.assign(
        test_cindex=lambda x: x.test_cindex.apply(parse_cell),
        best_params=lambda x: x.best_params.apply(parse_cell),
    ).rename(
        columns={
            "test_cindex": "cindex",
            "mean_cindex": "cindex_mean",
        }
    )

    flat_df = best_params.explode(
        [
            "cindex",
            "best_params",
        ],
        ignore_index=True,
    )

    selected_kernel = best_params.sort_values(
        [
            "cindex_mean",
            "std_cindex",
        ],
        ascending=[
            False,
            True,
        ],
    ).iloc[0]["kernel"]

    selected_row = (
        flat_df.query("kernel == @selected_kernel")
        .sort_values(
            "cindex",
            ascending=False,
        )
        .iloc[0]
    )

    return (
        selected_kernel,
        selected_row["best_params"],
        flat_df,
    )


def load_melanoma():

    melanoma = (
        pd.read_csv("ggkm/data/melanoma.csv")
        .assign(status=lambda x: (x.status == 1).astype(int))
        .fillna(0)
    )

    y = melanoma[
        [
            "time",
            "status",
        ]
    ]

    X = melanoma.drop(columns=y.columns.tolist()).to_numpy()
    t = y["time"].to_numpy().astype(float)
    delta = y["status"].to_numpy().astype(float)

    return X, t, delta


def build_estimator(
    model_name,
    kernel,
    best_params,
):
    ModelClass = MODELS[model_name]

    return ModelClass(
        kernel=kernel,
        **best_params,
    )


def save_results(
    results,
    task_id,
    model_name,
    kernel,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = OUTPUT_DIR / (
        f"melanoma_bagging_task{task_id}_{model_name}_kernel_{kernel}.csv"
    )

    pd.DataFrame([results]).to_csv(
        output,
        index=False,
    )

    return output


def main():

    start = time.time()
    task_id = int(os.environ["PBS_ARRAY_INDEX"])
    experiments = build_experiments()
    exp = get_experiment(
        experiments,
        task_id,
    )

    model_name = exp["model_name"]

    print(
        f"Running model={model_name}",
        flush=True,
    )

    kernel, best_params, flat_df = select_best_model(model_name)

    print(
        f"Selected kernel={kernel}",
        flush=True,
    )

    base_estimator = build_estimator(
        model_name,
        kernel,
        best_params,
    )

    X, t, delta = load_melanoma()

    results = cross_validate_gg_km(
        X,
        t,
        delta,
        estimator=GGKMKernelBagging,
        bagging=True,
        bagging_estimator=base_estimator,
        kernels=flat_df["kernel"],
        n_outer_splits=5,
        n_inner_splits=4,
        n_trials=20,
        t_grid_points=50,
        random_state=42,
        estimator_name=model_name,
    )

    output = save_results(
        results,
        task_id,
        model_name,
        kernel,
    )

    elapsed = time.time() - start

    print(
        "Finished",
        flush=True,
    )

    print(
        f"Elapsed seconds: {elapsed:.2f}",
        flush=True,
    )

    print(
        f"Elapsed minutes: {elapsed/60:.2f}",
        flush=True,
    )

    print(
        f"Saved: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
