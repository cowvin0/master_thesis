import pandas as pd
import re
import ast


def parse_cell(x):
    s = str(x).strip()
    s = re.sub(r"np\.float64\(([^()]*)\)", r"\1", s)
    return ast.literal_eval(s)


def explode_experimental_data(df, model_name):

    if model_name != "binomial":
        records = []

        for idx, row in df.iterrows():
            for n in df.columns:
                cell = row[n]
                if isinstance(cell, str):
                    results = parse_cell(cell)
                else:
                    results = cell

                if not isinstance(results, dict):
                    continue

                for kernel, values in results.items():
                    records.append(
                        {
                            "method": idx + 1,
                            "sample_size": int(n),
                            "kernel": kernel,
                            "cindex": values.get("test_cindex"),
                            "cindex_mean": values.get("mean_cindex"),
                            "std_cindex": values.get("std_cindex"),
                            "best_params": values.get("best_params"),
                            "test_ibs": values.get("test_ibs"),
                            "ibs_mean": values.get("mean_ibs"),
                            "std_ibs": values.get("std_ibs"),
                        }
                    )

        results_df = pd.DataFrame(records).explode(
            ["cindex", "best_params", "test_ibs"]
        )
    else:
        results_df = (
            df.assign(
                test_ibs=lambda x: x.test_ibs.apply(parse_cell),
                test_cindex=lambda x: x.test_cindex.apply(parse_cell),
                best_params=lambda x: x.best_params.apply(parse_cell),
            )
            .explode(["test_ibs", "test_cindex", "best_params"])
            .rename(columns={"test_cindex": "cindex"})
        )

    kernel_selection = results_df.groupby(
        ["method", "sample_size", "kernel"], as_index=False
    ).agg(cindex_mean=("cindex", "mean"), std_cindex=("cindex", "std"))

    best_kernel = (
        kernel_selection.sort_values(
            ["method", "sample_size", "cindex_mean", "std_cindex"],
            ascending=[True, True, False, True],
        )
        .groupby(["method", "sample_size"], as_index=False)
        .first()
    )

    best_hyperparameters = results_df.merge(
        best_kernel[["method", "sample_size", "kernel"]],
        on=["method", "sample_size", "kernel"],
    )

    best_hyperparameters = (
        best_hyperparameters.sort_values(
            ["method", "sample_size", "cindex"], ascending=[True, True, False]
        )
        .groupby(["method", "sample_size"], as_index=False)
        .first()
    )

    return best_hyperparameters
