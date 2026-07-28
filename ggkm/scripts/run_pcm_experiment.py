import os
import pandas as pd

from pathlib import Path
from ggkm.models.km_gg_bin import GGBinomial
from ggkm.evaluate.cross_validation import cross_validate_pcm

KERNELS = [
    "linear",
    "rbf",
    "laplacian",
    "exponential",
    "cauchy",
    "sigmoid",
    "polynomial",
]
SAMPLE_SIZES = [300, 600]
METHODS = [1, 2, 3]

OUTPUT_DIR = Path("data/results")


def build_experiments():
    experiments = []
    for n in SAMPLE_SIZES:
        for method in METHODS:
            for kernel in KERNELS:
                experiments.append(
                    {
                        "n": n,
                        "method": method,
                        "kernel": kernel,
                    }
                )
    return experiments


def get_experiment(experiments, task_id):
    if task_id >= len(experiments):
        raise ValueError(
            f"Invalid PBS_ARRAY_INDEX={task_id}. " f"Expected 0-{len(experiments)-1}."
        )
    return experiments[task_id]


def run_experiment(n, method, kernel):
    return cross_validate_pcm(
        n=n,
        method=method,
        kernel=kernel,
        estimator=GGBinomial,
        n_outer_splits=5,
        n_inner_splits=4,
        n_trials=20,
        t_grid_points=50,
        random_state=42,
    )


def save_results(results, task_id, n, method, kernel):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_file = OUTPUT_DIR / (
        f"result_task{task_id}" f"_n{n}" f"_method{method}" f"_kernel{kernel}.csv"
    )
    df = pd.DataFrame([results])
    df.to_csv(
        output_file,
        index=False,
    )
    return output_file


def main():
    task_id = int(os.environ["PBS_ARRAY_INDEX"])

    experiments = build_experiments()
    experiment = get_experiment(experiments, task_id)

    n = experiment["n"]
    method = experiment["method"]
    kernel = experiment["kernel"]

    print(
        f"Task {task_id}: " f"Running n={n}, method={method}, kernel={kernel}",
        flush=True,
    )

    results = run_experiment(n, method, kernel)

    output_file = save_results(results, task_id, n, method, kernel)

    print(
        f"Task {task_id} finished. " f"Saved {output_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
