import os
import time
import pandas as pd

from pathlib import Path
from ggkm.models.km_gg_bin import GGBinomial
from ggkm.models.km_gg import GGPoisson
from ggkm.models.km_gg_bn import GGNB
from ggkm.evaluate.cross_validation import cross_validate_gg_km
from ggkm.utils.preprocessing import MelanomaSurvivalPreprocessor

KERNELS = [
    "linear",
    "rbf",
    "laplacian",
    "exponential",
    "cauchy",
    "sigmoid",
    "polynomial",
]


MODELS = {
    "bernoulli": GGBinomial,
    "binomial": GGBinomial,
    "poisson": GGPoisson,
    "bn": GGNB,
}


OUTPUT_DIR = Path("data/melanoma_results")


def build_experiments():

    experiments = []

    for model_name in MODELS:
        for kernel in KERNELS:
            experiments.append(
                {
                    "model_name": model_name,
                    "kernel": kernel,
                }
            )

    return experiments


def get_experiment(
    experiments,
    task_id,
):

    if task_id >= len(experiments):
        raise ValueError(
            f"Invalid PBS_ARRAY_INDEX={task_id}. Expected 0-{len(experiments)-1}"
        )

    return experiments[task_id]


def load_melanoma():

    melanoma = (
        pd.read_csv("ggkm/data/melanoma.csv")
        .assign(status=lambda x: (x.status == 1).astype(int))
        .fillna(0)
        .rename(columns={"status": "delta"})
    )

    return melanoma


def run_experiment(
    df,
    model_name,
    kernel,
):

    estimator = MODELS[model_name]

    results = cross_validate_gg_km(
        df=df,
        preprocessor_factory=lambda: MelanomaSurvivalPreprocessor(),
        estimator=estimator,
        kernel=kernel,
        n_outer_splits=5,
        n_inner_splits=4,
        n_trials=20,
        t_grid_points=50,
        random_state=42,
        estimator_name=model_name,
    )

    return results


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

    output_file = OUTPUT_DIR / (
        f"melanoma_task{task_id}_{model_name}_kernel_{kernel}.csv"
    )

    pd.DataFrame([results]).to_csv(
        output_file,
        index=False,
    )

    return output_file


def main():

    start = time.time()
    task_id = int(os.environ["PBS_ARRAY_INDEX"])
    experiments = build_experiments()
    exp = get_experiment(
        experiments,
        task_id,
    )

    model_name = exp["model_name"]
    kernel = exp["kernel"]

    print(
        f"Task {task_id}",
        flush=True,
    )

    print(
        f"Model={model_name}, kernel={kernel}",
        flush=True,
    )

    df = load_melanoma()

    results = run_experiment(
        df=df,
        model_name=model_name,
        kernel=kernel,
    )

    output_file = save_results(
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
        f"Saved: {output_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
