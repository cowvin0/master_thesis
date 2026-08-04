import time
from pathlib import Path

import pandas as pd

from ggkm.models.km_gg_bin import GGBinomial
from ggkm.utils.metrics import integrated_brier_score
from ggkm.utils.preprocessing_breast_cancer import (
    BreastCancerSurvivalPreprocessor,
    SurvivalModelForwardSelector,
)


def load_data():

    return (
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
        .drop_duplicates()
    )


def main():

    start = time.time()

    print("Loading data...", flush=True)

    cancer_mama = load_data()

    Path("outputs").mkdir(
        parents=True,
        exist_ok=True,
    )

    selector = SurvivalModelForwardSelector(
        candidate_columns=BreastCancerSurvivalPreprocessor.RAW_FEATURES,
        model_factory=lambda: GGBinomial(
            kernel="rbf",
            gamma=0.002,
            lambda_reg=2.4362917997548086e-05,
            K_bin=113,
        ),
        validation_size=0.5,
        random_state=42,
        max_features=2,
        min_improvement=1e-5,
        log_path="outputs/ggbinomial_feature_selection_history.csv",
        secondary_metric=integrated_brier_score,
    )

    print("Starting forward feature selection...", flush=True)

    selector.fit(cancer_mama)

    elapsed = time.time() - start

    print("", flush=True)
    print("======================================", flush=True)
    print("Feature selection finished.", flush=True)
    print(f"Elapsed seconds: {elapsed:.2f}", flush=True)
    print(f"Elapsed minutes: {elapsed/60:.2f}", flush=True)
    print("History saved to:", flush=True)
    print("outputs/ggbinomial_feature_selection_history.csv", flush=True)
    print("======================================", flush=True)


if __name__ == "__main__":
    main()
