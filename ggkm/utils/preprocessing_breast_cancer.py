from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Dict, List, Optional
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer


class OrdinalMapEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, mapping: Dict[str, float]):
        self.column = column
        self.mapping = mapping

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.column] = X[self.column].map(self.mapping)
        return X


class MissingFlagOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        column: str,
        mapping: Dict[str, float],
        missing_category: str,
        missing_flag_name: str,
    ):
        self.column = column
        self.mapping = mapping
        self.missing_category = missing_category
        self.missing_flag_name = missing_flag_name

    def fit(self, X: pd.DataFrame, y=None):
        mapped = X[self.column].map(self.mapping)
        self.median_ = mapped.median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.missing_flag_name] = (X[self.column] == self.missing_category).astype(
            int
        )
        X[self.column] = X[self.column].map(self.mapping)
        X[self.column] = X[self.column].fillna(self.median_)
        return X


class CategoryFlagEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, flags: Dict[str, str], drop_original: bool = True):
        self.column = column
        self.flags = flags
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for new_col, category in self.flags.items():
            X[new_col] = (X[self.column] == category).astype(int)
        if self.drop_original:
            X = X.drop(columns=[self.column])
        return X


class BreastCancerSurvivalPreprocessor:
    TIME_COLUMN = "Tempo"
    STATUS_COLUMN = "Status"
    EVENT_LABEL = "Morte por cancer de Mama"

    RAW_FEATURES: List[str] = [
        "DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumorIntevalo",
        "DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumorIntevalo",
        "DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1ConsultaIntevalo",
        "PndrRenda",
        "FaixaETCAT",
        "RhcEstadiamentoClinico",
        "RhcIdadeNo1Consulta",
        "DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumor",
        "RhcHistoricoTabaco",
        "RhcHistoricoFamiliarCancer",
        "RhcHistoricoAlcool",
        "RhcPrimeiroTratamentoRecebidoNoHospitalNenhum",
        "RhcPrimeiroTratamentoRecebidoNoHospitalCirurgia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalRadioterapia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalQuimioterapia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalHormonioterapia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalImunoterapia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalOutras",
        "RhcPrimeiroTratamentoRecebidoNoHospitalSemInformacao",
    ]

    SCALED_FEATURES: List[str] = [
        "RhcIdadeNo1Consulta",
        "DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumor",
    ]

    DUMMY_FEATURES: List[str] = [
        "1tratamento_diagnostico_tumor_ate_90dias",
        "1tratamento_diagnostico_tumor_acima_90dias",
        "1tratamento_consulta_ate_90dias",
        "1tratamento_consulta_acima_90dias",
        "1consulta_diagnostico_tumor_ate_90dias",
        "1consulta_diagnostico_tumor_acima_90dias",
        "baixa_renda",
        "media_renda",
        "alta_renda",
        "renda_sem_informacao",
        "20_39",
        "40_49",
        "50_59",
        "60_69",
        "70_99",
        "I_II",
        "III_IV",
        "estadiamento_seminformacao",
        "Tabaco_Sim",
        "Tabaco_ExConsumidor",
        "Tabaco_SemInformacao",
        "HistFamCancer_Sim",
        "HistFamCancer_SemInformacao",
        "Alcool_Sim",
        "Alcool_ExConsumidor",
        "Alcool_SemInformacao",
        "RhcPrimeiroTratamentoRecebidoNoHospitalNenhum",
        "RhcPrimeiroTratamentoRecebidoNoHospitalCirurgia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalRadioterapia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalQuimioterapia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalHormonioterapia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalImunoterapia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalOutras",
        "RhcPrimeiroTratamentoRecebidoNoHospitalSemInformacao",
    ]

    def __init__(self):
        self.feature_pipeline_ = Pipeline(
            steps=[
                (
                    "1consulta_diagnostico_tumor",
                    CategoryFlagEncoder(
                        column="DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumorIntevalo",
                        flags={
                            "1consulta_diagnostico_tumor_ate_90dias": "1. até 90 dias",
                            "1consulta_diagnostico_tumor_acima_90dias": "2. acima de 90 dias",
                        },
                    ),
                ),
                (
                    "1tratamento_consulta",
                    CategoryFlagEncoder(
                        column="DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1ConsultaIntevalo",
                        flags={
                            "1tratamento_consulta_ate_90dias": "1. até 90 dias",
                            "1tratamento_consulta_acima_90dias": "2. acima de 90 dias",
                        },
                    ),
                ),
                (
                    "1tratamento_diagnostico_tumor",
                    CategoryFlagEncoder(
                        column="DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumorIntevalo",
                        flags={
                            "1tratamento_diagnostico_tumor_ate_90dias": "1. até 90 dias",
                            "1tratamento_diagnostico_tumor_acima_90dias": "2. acima de 90 dias",
                        },
                    ),
                ),
                (
                    "renda",
                    CategoryFlagEncoder(
                        column="PndrRenda",
                        flags={
                            "baixa_renda": "Baixa Renda",
                            "media_renda": "Média Renda",
                            "alta_renda": "Alta Renda",
                            "renda_sem_informacao": "Sem Informação",
                        },
                    ),
                ),
                (
                    "faixa_etaria",
                    CategoryFlagEncoder(
                        column="FaixaETCAT",
                        flags={
                            "20_39": "20 a 39",
                            "40_49": "40 a 49",
                            "50_59": "50 a 59",
                            "60_69": "60 a 69",
                            "70_99": "70 a 99",
                        },
                    ),
                ),
                (
                    "estadiamento",
                    CategoryFlagEncoder(
                        column="RhcEstadiamentoClinico",
                        flags={
                            "I_II": "I e II",
                            "III_IV": "III e IV",
                            "estadiamento_seminformacao": "Sem Informação",
                        },
                    ),
                ),
                (
                    "tabaco",
                    CategoryFlagEncoder(
                        column="RhcHistoricoTabaco",
                        flags={
                            "Tabaco_Sim": "Sim",
                            "Tabaco_ExConsumidor": "Ex-consumidor",
                            "Tabaco_SemInformacao": "Sem Informação",
                        },
                    ),
                ),
                (
                    "hist_familiar",
                    CategoryFlagEncoder(
                        column="RhcHistoricoFamiliarCancer",
                        flags={
                            "HistFamCancer_Sim": "Sim",
                            "HistFamCancer_SemInformacao": "Sem Informação",
                        },
                    ),
                ),
                (
                    "alcool",
                    CategoryFlagEncoder(
                        column="RhcHistoricoAlcool",
                        flags={
                            "Alcool_Sim": "Sim",
                            "Alcool_ExConsumidor": "Ex-consumidor",
                            "Alcool_SemInformacao": "Sem Informação",
                        },
                    ),
                ),
            ]
        )

        self.column_transformer_ = ColumnTransformer(
            transformers=[
                (
                    "power",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                    self.SCALED_FEATURES,
                ),
                ("dummy", "passthrough", self.DUMMY_FEATURES),
            ]
        )

        self.df_model_: Optional[pd.DataFrame] = None

    def _build_delta(self, df: pd.DataFrame) -> pd.Series:
        return (df[self.STATUS_COLUMN] == self.EVENT_LABEL).astype(int)

    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()
        df["delta"] = self._build_delta(df)

        cols = self.RAW_FEATURES + [self.TIME_COLUMN, "delta"]
        df_model = df[cols].copy()

        df_model = self.feature_pipeline_.fit_transform(df_model)
        self.df_model_ = df_model

        t = df_model[self.TIME_COLUMN].values
        delta = df_model["delta"].values

        X = self.column_transformer_.fit_transform(df_model)
        X = np.asarray(X, dtype=float)

        return X, t, delta
