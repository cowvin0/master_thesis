import os
import time
import warnings
from pathlib import Path

import optuna
import pandas as pd

from ggkm.models.km_gg_bin import GGBinomial
from ggkm.models.km_gg import GGPoisson
from ggkm.models.km_gg_bn import GGNB
from ggkm.models.bagging import GGKMKernelBagging
from ggkm.evaluate.cross_validation import cross_validate_pcm

optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")


MODELS = {
    "bernoulli": (
        GGBinomial,
        {},
        # {
        #     "K_bin": 1,
        # },
    ),
    "binomial": (
        GGBinomial,
        {},
    ),
    "poisson": (
        GGPoisson,
        {},
    ),
    "bn": (
        GGNB,
        {},
    ),
}


SAMPLE_SIZES = [
    300,
    600,
]


METHODS = [
    1,
    2,
    3,
]


KERNELS = [
    "linear",
    "rbf",
    "laplacian",
    "exponential",
    "cauchy",
    "polynomial",
    "sigmoid",
]


OUTPUT_DIR = Path("data/results_bagging_random")


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


def get_experiment(
    experiments,
    task_id,
):

    if task_id >= len(experiments):

        raise ValueError(
            f"Invalid PBS_ARRAY_INDEX={task_id}. " f"Expected 0-{len(experiments) - 1}."
        )

    return experiments[task_id]


def main():

    task_start = time.time()

    task_id = int(os.environ["PBS_ARRAY_INDEX"])

    experiments = build_experiments()

    experiment = get_experiment(
        experiments,
        task_id,
    )

    model_name = experiment["model_name"]
    n = experiment["n"]
    method = experiment["method"]

    ModelClass, model_kwargs = MODELS[model_name]

    print(
        f"Task {task_id}: " f"model={model_name}, " f"n={n}, " f"method={method}",
        flush=True,
    )

    print(
        "Starting randomized kernel/hyperparameter " "bagging experiment...",
        flush=True,
    )

    results = cross_validate_pcm(
        n=n,
        method=method,
        estimator=GGKMKernelBagging,
        bagging=True,
        kernels=KERNELS,
        optimize_kernel_ranges=True,
        bagging_estimator=ModelClass,
        estimator_name=model_name,
        **model_kwargs,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = OUTPUT_DIR / (
        f"task{task_id}_" f"{model_name}_" f"n{n}_" f"method{method}.csv"
    )

    pd.DataFrame([results]).to_csv(
        output_file,
        index=False,
    )

    elapsed = time.time() - task_start

    print(
        "",
        flush=True,
    )

    print(
        "==================================",
        flush=True,
    )

    print(
        f"Task {task_id} finished",
        flush=True,
    )

    print(
        f"Model: {model_name}",
        flush=True,
    )

    print(
        f"Sample size: {n}",
        flush=True,
    )

    print(
        f"Method: {method}",
        flush=True,
    )

    print(
        f"Elapsed seconds: {elapsed:.2f}",
        flush=True,
    )

    print(
        f"Elapsed minutes: {elapsed / 60:.2f}",
        flush=True,
    )

    print(
        f"Saved: {output_file}",
        flush=True,
    )

    print(
        "==================================",
        flush=True,
    )


if __name__ == "__main__":
    main()
