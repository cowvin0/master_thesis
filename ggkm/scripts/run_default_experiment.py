import os
import time
import pandas as pd

from pathlib import Path
from ggkm.models.km_gg_bin import GGBinomial
from ggkm.models.km_gg import GGPoisson
from ggkm.models.km_gg_bn import GGNB
from ggkm.evaluate.cross_validation import cross_validate_gg_km
from ggkm.utils.preprocessing import DefaultSurvivalPreprocessor

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


OUTPUT_DIR = Path("data/default_results")


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


def load_default():

    origination = pd.read_csv("ggkm/data/sample_orig_2024.txt", sep="|").filter(
        regex="channel|number_of_borrowers"
        "|dti|first_time_homebuyer_indicator"
        "|postal_code|property_type|harp_indicator"
        "|seller_name|cltv|ltv|fico|original_interest_rate"
        "|mortgage_insurance_percentage|original_loan_term|original_upb"
        "|loan_identifier"
    )

    performance = pd.read_csv("ggkm/data/sample_perf_2024.txt", sep="|").filter(
        regex="loan_identifier|current_loan_deliquency_status|^period|loan_age"
    )

    df_long = pd.merge(
        origination, performance, on="loan_identifier", how="inner"
    ).assign(
        delinquency_status=lambda x: pd.to_numeric(
            x.current_loan_deliquency_status, errors="coerce"
        ),
        event=lambda x: x.delinquency_status >= 2,
    )

    event_times = (
        df_long.loc[df_long["delinquency_status"] >= 2]
        .groupby("loan_identifier")["loan_age"]
        .min()
        .rename("t")
    )

    last_observed = (
        df_long.groupby("loan_identifier")["loan_age"].max().rename("last_loan_age")
    )

    df = (
        origination.merge(event_times, on="loan_identifier", how="left")
        .merge(last_observed, on="loan_identifier", how="left")
        .assign(
            delta=lambda x: x["t"].notna().astype(int),
            t=lambda x: x["t"].fillna(x["last_loan_age"]),
            postal_code=lambda x: x["postal_code"].astype(str),
            first_time_homebuyer_indicator=lambda x: (
                x["first_time_homebuyer_indicator"] == "Y"
            ).astype(int),
            harp_indicator=lambda x: (
                x["harp_indicator(relief_refinance)"] == "Y"
            ).astype(int),
        )
        .drop(
            columns=[
                "harp_indicator(relief_refinance)",
                "last_loan_age",
                "loan_identifier",
            ]
        )
    )

    return df


def run_experiment(
    df,
    model_name,
    kernel,
):

    estimator = MODELS[model_name]

    results = cross_validate_gg_km(
        df=df,
        preprocessor_factory=lambda: DefaultSurvivalPreprocessor(),
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
        f"default_task{task_id}_{model_name}_kernel_{kernel}.csv"
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

    df = load_default()

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

    # output_file = save_results(
    #     results,
    #     task_id,
    #     model_name,
    #     kernel,
    # )

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
