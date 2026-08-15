import os
import time
from pathlib import Path

import pandas as pd

from ggkm.models.km_gg_bin import GGBinomial
from ggkm.models.km_gg import GGPoisson
from ggkm.models.km_gg_bn import GGNB
from ggkm.models.bagging import GGKMKernelBagging

from ggkm.evaluate.cross_validation import (
    cross_validate_gg_km,
)

from ggkm.utils.preprocessing import (
    MelanomaSurvivalPreprocessor,
)

MODELS = {
    "bernoulli": (
        GGBinomial,
        {},
        # {"K_bin": 1},
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


OUTPUT_DIR = Path("data/melanoma_results")


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

    ModelClass, extra_args = MODELS[model_name]

    print(
        f"Task {task_id}: " f"model={model_name}",
        flush=True,
    )

    print(
        "Loading melanoma data...",
        flush=True,
    )

    melanoma = (
        pd.read_csv("ggkm/data/melanoma.csv")
        .assign(status=lambda x: (x.status == 1).astype(int))
        .fillna(0)
        .rename(columns={"status": "delta"})
    )

    print(
        f"Number of observations: {len(melanoma)}",
        flush=True,
    )

    print(
        f"Starting randomized bagging experiment " f"for model={model_name}",
        flush=True,
    )

    results = cross_validate_gg_km(
        df=melanoma,
        estimator=GGKMKernelBagging,
        preprocessor_factory=lambda: MelanomaSurvivalPreprocessor(),
        bagging=True,
        kernels=[
            "linear",
            "rbf",
            "laplacian",
            "exponential",
            "cauchy",
            "polynomial",
            "sigmoid",
        ],
        optimize_kernel_ranges=True,
        bagging_estimator=ModelClass,
        estimator_name=model_name,
        **extra_args,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = OUTPUT_DIR / (
        f"task{task_id}_" f"bagging_random_" f"{model_name}.csv"
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
