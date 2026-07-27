import os
from pathlib import Path

import pandas as pd

from ggkm.models.km_gg_bin import GGBinomial
from ggkm.evaluate.cross_validation import cross_validate_pcm

kernels = [
    "linear",
    "rbf",
    "laplacian",
    "exponential",
    "cauchy",
    "sigmoid",
    "polynomial",
]

sample_sizes = [300, 600]
methods = [1, 2, 3]

task_id = int(os.environ["PBS_ARRAY_INDEX"])

experiments = []

for n in sample_sizes:
    for method in methods:
        for kernel in kernels:
            experiments.append(
                {
                    "n": n,
                    "method": method,
                    "kernel": kernel,
                }
            )


if task_id >= len(experiments):
    raise ValueError(
        f"Invalid PBS_ARRAY_INDEX={task_id}. " f"Expected 0-{len(experiments)-1}."
    )


experiment = experiments[task_id]

n = experiment["n"]
method = experiment["method"]
kernel = experiment["kernel"]


print(
    f"Task {task_id}: " f"Running n={n}, method={method}, kernel={kernel}",
    flush=True,
)

results = cross_validate_pcm(
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

output_dir = Path("data/results")

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


output_file = output_dir / (
    f"result_task{task_id}" f"_n{n}" f"_method{method}" f"_kernel{kernel}.csv"
)

df = pd.DataFrame([results])


df.to_csv(
    output_file,
    index=False,
)


print(
    f"Task {task_id} finished. " f"Saved {output_file}",
    flush=True,
)
