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

experiment_results = {}

for n in sample_sizes:
    experiment_results[n] = {}
    for method in methods:
        experiment_results[n][method] = {}

        for kernel in kernels:
            print(f"Running n={n}, method={method}, kernel={kernel}", flush=True)

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

            experiment_results[n][method][kernel] = results


df_results = pd.DataFrame(experiment_results)
output_path = "data/experimental_results_em_binomial.csv"
df_results.to_csv(output_path, index=False)

print(f"Results saved to {output_path}", flush=True)
