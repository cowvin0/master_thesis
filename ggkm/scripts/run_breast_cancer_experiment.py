import os
import time
import pandas as pd

from pathlib import Path
from ggkm.models.km_gg_bin import GGBinomial
from ggkm.models.km_gg import GGPoisson
from ggkm.models.km_gg_bn import GGNB
from ggkm.evaluate.cross_validation import cross_validate_gg_km
from ggkm.utils.preprocessing import BreastCancerSurvivalPreprocessor

MODELS = {
    "bernoulli": (
        GGBinomial,
        {"K_bin": 1},
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


KERNELS = [
    "linear",
    "rbf",
    "laplacian",
    "exponential",
    "cauchy",
    "sigmoid",
    "polynomial",
]


MUNICIPALITIES = [
    "Campina Grande",
    "João Pessoa",
    "Outros",
]


OUTPUT_DIR = Path("data/breast_cancer_results")


def build_experiments():

    experiments = []

    for model_name in MODELS:
        for municipality in MUNICIPALITIES:
            for kernel in KERNELS:
                experiments.append(
                    {
                        "model_name": model_name,
                        "municipality": municipality,
                        "kernel": kernel,
                    }
                )

    return experiments


def load_data(municipality):

    df = (
        pd.read_csv("ggkm/data/cancer_de_mama_rhc_e_fap.csv")
        .drop(
            columns=[
                "RhcDataDeNascimento",
                "Limitacao",
                "DiffEmDiasEntreRhcDtNascimentoDtObito",
                "Instrução",
                "EstCong",
                "HistFamCan",
                "SimDataDeNascimento",
                "CosemsCodIbge",
                "Renda",
                "Macroregião",
                "DIFFTRATCONS1",
                "DIFFDIAGTRAT",
                "DIFFDIAGCONS1",
                "Diagnostico",
                "Raça",
                "AnoCateg",
                "Idade",
                "Ano",
                "Estadiamento",
            ]
        )
        .fillna(
            {
                "SimRecordID": "vivo",
                "RhcRecordID": "desconhecido",
                "SimCausaBasicaCategoria": "vivo",
                "SimCausaBasica": "vivo",
            }
        )
        .rename(columns={"Status": "delta"})
        .assign(delta=lambda x: (x.delta == "Morte por cancer de Mama").astype(int))
    )

    if municipality != "Outros":

        df = df.query("Município == @municipality")

    else:

        df = df.query("Município != 'Campina Grande' and Município != 'João Pessoa'")

    return df.drop_duplicates()


def main():

    start = time.time()
    task_id = int(os.environ["PBS_ARRAY_INDEX"])
    experiments = build_experiments()
    exp = experiments[task_id]
    model_name = exp["model_name"]
    municipality = exp["municipality"]
    kernel = exp["kernel"]

    ModelClass, _ = MODELS[model_name]

    print(
        f"Model={model_name} " f"Municipality={municipality} " f"Kernel={kernel}",
        flush=True,
    )

    cancer_mama = load_data(municipality)
    results = cross_validate_gg_km(
        df=cancer_mama,
        estimator=ModelClass,
        estimator_name=model_name,
        kernel=kernel,
        preprocessor_factory=lambda: BreastCancerSurvivalPreprocessor(),
        n_outer_splits=5,
        n_inner_splits=4,
        n_trials=20,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    municipality_file = (
        municipality.replace(" ", "_").replace("ã", "a").replace("ó", "o")
    )

    output = OUTPUT_DIR / (
        f"task{task_id}_" f"{model_name}_" f"{municipality_file}_" f"{kernel}.csv"
    )

    pd.DataFrame([results]).to_csv(
        output,
        index=False,
    )

    elapsed = time.time() - start

    print(
        f"Task {task_id} finished",
        flush=True,
    )

    print(
        f"Elapsed seconds: {elapsed:.2f}",
        flush=True,
    )

    print(
        f"Saved: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
