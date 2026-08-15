import os
import time
import pandas as pd

from pathlib import Path
from ggkm.models.km_gg_bin import GGBinomialGB
from ggkm.models.km_gg import GGPoissonGB
from ggkm.models.km_gg_bn import GGNBGB
from ggkm.evaluate.cross_validation import cross_validate_pcm_boost

SAMPLE_SIZES = [300, 600]

METHODS = [1, 2, 3]

MODELS = {
    "bernoulli": GGBinomialGB,
    "binomial": GGBinomialGB,
    "poisson": GGPoissonGB,
    "bn": GGNBGB,
}

OUTPUT_DIR = Path("data/results/boost")


def build_experiments():

    experiments = []

    for model_name in MODELS:
        for n in SAMPLE_SIZES:
            for method in METHODS:
                experiments.append(
                    {
                        "model_name": model_name,
                        "n": n,
                        "method": method,
                    }
                )

    return experiments


def get_experiment(experiments, task_id):

    if task_id >= len(experiments):

        raise ValueError(
            f"Invalid PBS_ARRAY_INDEX={task_id}. " f"Expected 0-{len(experiments)-1}."
        )

    return experiments[task_id]


def run_experiment(model_name, n, method):

    estimator_class = MODELS[model_name]

    return cross_validate_pcm_boost(
        n=n,
        method=method,
        estimator=estimator_class,
        n_outer_splits=5,
        n_inner_splits=4,
        n_trials=20,
        t_grid_points=50,
        random_state=42,
        estimator_name=model_name,
    )


def save_results(
    results,
    task_id,
    model_name,
    n,
    method,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = OUTPUT_DIR / (
        f"result_task{task_id}_"
        f"{model_name}_"
        f"n{n}_"
        f"method{method}_"
        f"boost.csv"
    )

    pd.DataFrame([results]).to_csv(
        output_file,
        index=False,
    )

    return output_file


def main():

    start_time = time.time()

    task_id = int(os.environ["PBS_ARRAY_INDEX"])

    experiments = build_experiments()

    exp = get_experiment(
        experiments,
        task_id,
    )

    model_name = exp["model_name"]
    n = exp["n"]
    method = exp["method"]

    print(
        f"Task {task_id} started",
        flush=True,
    )

    print(
        f"Model={model_name} " f"n={n} " f"method={method} ",
        flush=True,
    )

    results = run_experiment(
        model_name,
        n,
        method,
    )

    output_file = save_results(
        results,
        task_id,
        model_name,
        n,
        method,
    )

    elapsed = time.time() - start_time

    print(
        f"Task {task_id} finished",
        flush=True,
    )

    print(
        f"Elapsed seconds: {elapsed:.2f}",
        flush=True,
    )

    print(
        f"Saved: {output_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
