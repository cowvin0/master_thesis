import os
import pandas as pd

from ggkm.models.km_gg_bin import GGBinomial
from ggkm.models.km_gg_pois import GGPoisson
from ggkm.models.km_gg_nb import GGNB
from ggkm.models.kernel_bagging import GGKMKernelBagging

from ggkm.evaluate.cross_validation import cross_validate_pcm
from ggkm.utils.experiment_utils import explode_experimental_data

data_model = {
    "bernoulli": GGBinomial,
    "binomial": GGBinomial,
    "poisson": GGPoisson,
    "bn": GGNB,
}


experiments = []

for model_name in data_model:

    df = pd.read_csv(
        f"data/experimental_results/experiment_results_em_{model_name}.csv"
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


task_id = int(os.environ["PBS_ARRAY_INDEX"])

exp = experiments[task_id]

model_name = exp["model_name"]
method = exp["method"]
sample_size = exp["sample_size"]
kernel = exp["kernel"]
best_params = exp["best_params"]

ModelClass = data_model[model_name]

print(
    f"Running model={model_name}, "
    f"method={method}, "
    f"n={sample_size}, "
    f"kernel={kernel}",
    flush=True,
)

if model_name == "bernoulli":

    base_estimator = ModelClass(
        kernel=kernel,
        K_bin=1,
        **best_params,
    )

else:

    base_estimator = ModelClass(
        kernel=kernel,
        **best_params,
    )


results = cross_validate_pcm(
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
)

os.makedirs("results_bagging", exist_ok=True)

pd.DataFrame([results]).to_csv(
    (
        f"results_bagging/"
        f"task{task_id}_"
        f"{model_name}_"
        f"method{method}_"
        f"n{sample_size}.csv"
    ),
    index=False,
)

print("Finished", flush=True)
