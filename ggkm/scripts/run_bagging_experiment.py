import os
import time
import traceback
import pandas as pd

from ggkm.models.km_gg_bin import GGBinomial
from ggkm.models.km_gg import GGPoisson
from ggkm.models.km_gg_bn import GGNB
from ggkm.models.bagging import GGKMKernelBagging
from ggkm.evaluate.cross_validation import cross_validate_pcm
from ggkm.utils.experiment_utils import explode_experimental_data

DATA_MODEL = {
    "bernoulli": GGBinomial,
    "binomial": GGBinomial,
    "poisson": GGPoisson,
    "bn": GGNB,
}

RESULTS_DIR = "results_bagging"

START_BANNER = """
=================================
Starting bagging experiment
=================================
Task ID: {task}
Model: {model}
Method: {method}
Sample size: {n}
Kernel: {kernel}
Start time: {time}
=================================
"""

END_BANNER = """
=================================
Finished bagging experiment
=================================
Task ID: {task}
Model: {model}
Method: {method}
Sample size: {n}
Kernel: {kernel}
Status: {status}
End time: {time}
Elapsed seconds: {seconds:.2f}
Elapsed minutes: {minutes:.2f}
=================================
"""


def build_experiments():
    experiments = []
    for model_name in DATA_MODEL:
        df = pd.read_csv(
            f"ggkm/data/experimental_results/experiment_results_em_{model_name}.csv"
        )
        best_hp = explode_experimental_data(
            df,
            model_name=model_name,
        )
        for _, row in best_hp.iterrows():
            experiments.append(
                {
                    "model_name": model_name,
                    "method": int(row["method"]),
                    "sample_size": int(row["sample_size"]),
                    "kernel": row["kernel"],
                    "best_params": row["best_params"],
                }
            )
    return experiments


def build_base_estimator(model_name, kernel, best_params):
    ModelClass = DATA_MODEL[model_name]
    if model_name == "bernoulli":
        return ModelClass(
            kernel=kernel,
            K_bin=1,
            **best_params,
        )
    return ModelClass(
        kernel=kernel,
        **best_params,
    )


def run_experiment(model_name, method, sample_size, kernel, best_params):
    base_estimator = build_base_estimator(model_name, kernel, best_params)
    return cross_validate_pcm(
        method=method,
        n=sample_size,
        estimator=GGKMKernelBagging,
        bagging=True,
        bagging_estimator=base_estimator,
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
    method,
    sample_size,
    kernel,
    elapsed_seconds,
    elapsed_minutes,
    status,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_df = pd.DataFrame([results])
    results_df["task_id"] = task_id
    results_df["model_name"] = model_name
    results_df["method"] = method
    results_df["sample_size"] = sample_size
    results_df["kernel"] = kernel
    results_df["elapsed_seconds"] = elapsed_seconds
    results_df["elapsed_minutes"] = elapsed_minutes
    results_df["status"] = status

    output_file = (
        f"{RESULTS_DIR}/"
        f"task{task_id}_"
        f"{model_name}_"
        f"method{method}_"
        f"n{sample_size}_"
        f"kernel{kernel}.csv"
    )
    results_df.to_csv(
        output_file,
        index=False,
    )
    print(
        f"Saved results: {output_file}",
        flush=True,
    )


def main():
    experiments = build_experiments()

    task_id = int(os.environ["PBS_ARRAY_INDEX"])
    exp = experiments[task_id]
    model_name = exp["model_name"]
    method = exp["method"]
    sample_size = exp["sample_size"]
    kernel = exp["kernel"]
    best_params = exp["best_params"]

    start_time = time.time()
    print(
        START_BANNER.format(
            task=task_id,
            model=model_name,
            method=method,
            n=sample_size,
            kernel=kernel,
            time=time.ctime(),
        ),
        flush=True,
    )

    try:
        results = run_experiment(model_name, method, sample_size, kernel, best_params)
        status = "success"
    except Exception:
        status = "failed"
        print(
            "Experiment failed with exception:",
            flush=True,
        )
        traceback.print_exc()
        results = {"error": traceback.format_exc()}

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60

    print(
        END_BANNER.format(
            task=task_id,
            model=model_name,
            method=method,
            n=sample_size,
            kernel=kernel,
            status=status,
            time=time.ctime(),
            seconds=elapsed_seconds,
            minutes=elapsed_minutes,
        ),
        flush=True,
    )

    save_results(
        results,
        task_id,
        model_name,
        method,
        sample_size,
        kernel,
        elapsed_seconds,
        elapsed_minutes,
        status,
    )


if __name__ == "__main__":
    main()
