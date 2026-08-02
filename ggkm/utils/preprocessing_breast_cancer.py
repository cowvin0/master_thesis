from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Union
from sklearn.preprocessing import PowerTransformer, TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold


class MissingIndicatorNumericEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        columns: List[str],
    ):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):

        self.medians_ = {}

        for col in self.columns:
            self.medians_[col] = X[col].median()

        return self

    def transform(self, X: pd.DataFrame):

        X = X.copy()

        for col in self.columns:
            missing_name = f"{col}_missing"
            X[missing_name] = (X[col].isna()).astype(int)
            X[col] = X[col].fillna(self.medians_[col])

        return X


class StageTreatmentInteractionEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        stage_columns: List[str],
        treatment_columns: List[str],
    ):
        self.stage_columns = stage_columns
        self.treatment_columns = treatment_columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X = X.copy()

        for stage in self.stage_columns:
            for treatment in self.treatment_columns:
                interaction_name = f"{stage}_{treatment}"
                X[interaction_name] = X[stage].astype(int) * X[treatment].astype(int)

        return X


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


class CategoryTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        column: str,
        output_column: Optional[str] = None,
        drop_original: bool = True,
        target_type: str = "auto",
        smooth: Union[str, float] = "auto",
        cv: int = 5,
        random_state: Optional[int] = None,
    ):
        self.column = column
        self.output_column = output_column
        self.drop_original = drop_original
        self.target_type = target_type
        self.smooth = smooth
        self.cv = cv
        self.random_state = random_state

    def _out_col(self) -> str:
        return self.output_column or f"{self.column}_te"

    def fit(self, X: pd.DataFrame, y=None):
        if y is None:
            raise ValueError(
                "CategoryTargetEncoder requires y (the target) to fit. "
                "Pass it through pipeline.fit_transform(X, y)."
            )
        self.encoder_ = TargetEncoder(
            target_type=self.target_type,
            smooth=self.smooth,
            cv=self.cv,
            random_state=self.random_state,
        )
        self.encoder_.fit(X[[self.column]], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        encoded = self.encoder_.transform(X[[self.column]])
        X[self._out_col()] = encoded.ravel()
        if self.drop_original and self.column != self._out_col():
            X = X.drop(columns=[self.column])
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if y is None:
            raise ValueError(
                "CategoryTargetEncoder requires y (the target) to fit_transform. "
                "Pass it through pipeline.fit_transform(X, y)."
            )
        self.encoder_ = TargetEncoder(
            target_type=self.target_type,
            smooth=self.smooth,
            cv=self.cv,
            random_state=self.random_state,
        )
        encoded = self.encoder_.fit_transform(X[[self.column]], y)
        X = X.copy()
        X[self._out_col()] = encoded.ravel()
        if self.drop_original and self.column != self._out_col():
            X = X.drop(columns=[self.column])
        return X


class SurvivalTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        column: str,
        time_column: str,
        event_column: str,
        output_column: Optional[str] = None,
        drop_original: bool = True,
        tau: Optional[float] = None,
        smooth: float = 10.0,
        cv: int = 5,
        random_state: Optional[int] = None,
    ):
        self.column = column
        self.time_column = time_column
        self.event_column = event_column
        self.output_column = output_column
        self.drop_original = drop_original
        self.tau = tau
        self.smooth = smooth
        self.cv = cv
        self.random_state = random_state

    def _out_col(self) -> str:
        return self.output_column or f"{self.column}_rmst"

    @staticmethod
    def _kaplan_meier(times: np.ndarray, events: np.ndarray):
        times = np.asarray(times, dtype=float)
        events = np.asarray(events, dtype=float)

        unique_event_times = np.unique(times[events == 1])
        if unique_event_times.size == 0:
            return np.array([]), np.array([])

        survival = []
        s = 1.0
        for t in unique_event_times:
            n_at_risk = np.sum(times >= t)
            n_events = np.sum((times == t) & (events == 1))
            if n_at_risk > 0:
                s *= 1.0 - (n_events / n_at_risk)
            survival.append(s)
        return unique_event_times, np.array(survival)

    @staticmethod
    def _rmst(
        unique_event_times: np.ndarray, survival: np.ndarray, tau: float
    ) -> float:
        if tau <= 0:
            return 0.0
        if unique_event_times.size == 0:
            return float(tau)

        t_grid = np.concatenate(([0.0], unique_event_times))
        s_grid = np.concatenate(([1.0], survival))

        mask = t_grid <= tau
        t_grid = t_grid[mask]
        s_grid = s_grid[mask]

        if t_grid[-1] < tau:
            t_grid = np.concatenate([t_grid, [tau]])
            s_grid = np.concatenate([s_grid, [s_grid[-1]]])

        area = 0.0
        for i in range(len(t_grid) - 1):
            area += s_grid[i] * (t_grid[i + 1] - t_grid[i])
        return float(area)

    def _category_stats(self, df: pd.DataFrame, tau: float) -> Dict:
        stats = {}
        for category, group in df.groupby(self.column, dropna=False):
            u_times, surv = self._kaplan_meier(
                group[self.time_column].to_numpy(),
                group[self.event_column].to_numpy(),
            )
            stats[category] = {
                "rmst": self._rmst(u_times, surv, tau),
                "count": len(group),
            }
        return stats

    def _global_rmst(self, df: pd.DataFrame, tau: float) -> float:
        u_times, surv = self._kaplan_meier(
            df[self.time_column].to_numpy(), df[self.event_column].to_numpy()
        )
        return self._rmst(u_times, surv, tau)

    def _encode_from_stats(self, category, stats: Dict, global_rmst: float) -> float:
        cat_stats = stats.get(category)
        if cat_stats is None:
            return global_rmst
        n = cat_stats["count"]
        weight = n / (n + self.smooth)
        return weight * cat_stats["rmst"] + (1 - weight) * global_rmst

    def fit(self, X: pd.DataFrame, y=None):
        tau = self.tau if self.tau is not None else X[self.time_column].max()
        self.tau_ = tau
        self.global_rmst_ = self._global_rmst(X, tau)
        self.category_stats_ = self._category_stats(X, tau)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self._out_col()] = X[self.column].map(
            lambda c: self._encode_from_stats(
                c, self.category_stats_, self.global_rmst_
            )
        )
        if self.drop_original and self.column != self._out_col():
            X = X.drop(columns=[self.column])
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        tau = self.tau if self.tau is not None else X[self.time_column].max()
        self.tau_ = tau

        self.global_rmst_ = self._global_rmst(X, tau)
        self.category_stats_ = self._category_stats(X, tau)

        encoded = np.empty(len(X), dtype=float)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        for train_idx, holdout_idx in kf.split(np.arange(len(X))):
            fold_df = X.iloc[train_idx]
            fold_global = self._global_rmst(fold_df, tau)
            fold_stats = self._category_stats(fold_df, tau)

            holdout_categories = X.iloc[holdout_idx][self.column].to_numpy()
            encoded[holdout_idx] = [
                self._encode_from_stats(c, fold_stats, fold_global)
                for c in holdout_categories
            ]

        X = X.copy()
        X[self._out_col()] = encoded
        if self.drop_original and self.column != self._out_col():
            X = X.drop(columns=[self.column])
        return X


class BreastCancerSurvivalPreprocessor:
    TIME_COLUMN = "Tempo"
    STATUS_COLUMN = "Status"
    EVENT_LABEL = "Morte por cancer de Mama"

    RAW_FEATURES: List[str] = [
        "SimCausaBasicaCategoria",
        "RhcLOCTUPRI",
        "CosemsMacrorregiaoSaude",
        "Origem",
        "RhcOrigemEncamiamento",
        "RhcFonte",
        "RhcTipoHistológico",
        "DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumorIntevalo",
        "DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumorIntevalo",
        "DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1ConsultaIntevalo",
        "PndrRenda",
        "FaixaETCAT",
        "RhcEstadiamentoClinico",
        "RhcIdadeNo1Consulta",
        "DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumor",
        "DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumor",
        "DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1Consulta",
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
        "DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumor",
        "DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1Consulta",
    ]

    TARGET_ENCODED_FEATURES: List[str] = [
        # "RhcTipoHistológico_te",
        "RhcTipoHistológico_rmst",
        "RhcLOCTUPRI_rmst",
        "SimCausaBasicaCategoria_rmst",
    ]

    DUMMY_FEATURES: List[str] = [
        "1_macro",
        "2_macro",
        "3_macro",
        "macro_regiao_sem_informacao",
        "origem_sus",
        "origem_nao_sus",
        "origem_nenhum",
        "sus",
        "origem_sem_informacao",
        "nao_sus",
        "veio_por_conta",
        "nao_se_aplica",
        "laureano",
        "hu_cg",
        "fap",
        "DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumor_missing",
        "DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1Consulta_missing",
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
        "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalNenhum",
        "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalCirurgia",
        "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalRadioterapia",
        "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalQuimioterapia",
        "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalHormonioterapia",
        "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalImunoterapia",
        "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalOutras",
        "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalSemInformacao",
        "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalNenhum",
        "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalCirurgia",
        "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalRadioterapia",
        "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalQuimioterapia",
        "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalHormonioterapia",
        "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalImunoterapia",
        "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalOutras",
        "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalSemInformacao",
    ]

    def __init__(self):
        self.feature_pipeline_ = Pipeline(
            steps=[
                (
                    "continuous_delay_missing",
                    MissingIndicatorNumericEncoder(
                        columns=[
                            "DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumor",
                            "DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1Consulta",
                        ]
                    ),
                ),
                (
                    "macro_regiao",
                    CategoryFlagEncoder(
                        column="CosemsMacrorregiaoSaude",
                        flags={
                            "1_macro": "1ª Macro",
                            "2_macro": "2ª Macro",
                            "3_macro": "3ª Macro",
                            "macro_regiao_sem_informacao": "Sem Informação",
                        },
                    ),
                ),
                (
                    "origem",
                    CategoryFlagEncoder(
                        column="Origem",
                        flags={
                            "origem_sus": "SUS",
                            "origem_nao_sus": "Não SUS",
                            "origem_nenhum": np.nan,
                        },
                    ),
                ),
                (
                    "origem_encaminhamento",
                    CategoryFlagEncoder(
                        column="RhcOrigemEncamiamento",
                        flags={
                            "sus": "SUS",
                            "origem_sem_informacao": "Sem Informação",
                            "nao_sus": "Não SUS",
                            "veio_por_conta": "Veio Por Conta Própria",
                            "nao_se_aplica": "Não se Aplica",
                        },
                    ),
                ),
                (
                    "fonte",
                    CategoryFlagEncoder(
                        column="RhcFonte",
                        flags={"laureano": "Laureano", "fap": "FAP", "hu_cg": "HU CG"},
                    ),
                ),
                # --- Target encoding: histological type ---
                # Current choice: CategoryTargetEncoder against `delta`
                # (binary event rate per category). Kept as-is so it's easy
                # to keep using.
                # (
                #     "tipo_histologico",
                #     CategoryTargetEncoder(
                #         column="RhcTipoHistológico",
                #         output_column="RhcTipoHistológico_te",
                #         target_type="binary",
                #         smooth="auto",
                #         cv=5,
                #         random_state=42,
                #     ),
                # ),
                # --- Alternative: censoring-aware version ---
                # Swap the step above for this one to encode against the
                # per-category Kaplan-Meier RMST instead of the raw event
                # rate. Remember to also update TARGET_ENCODED_FEATURES to
                # "RhcTipoHistológico_rmst" if you switch.
                #
                (
                    "causa_morte",
                    SurvivalTargetEncoder(
                        column="SimCausaBasicaCategoria",
                        time_column="Tempo",
                        event_column="delta",
                        output_column="SimCausaBasicaCategoria_rmst",
                        smooth=10.0,
                        cv=5,
                        random_state=42,
                    ),
                ),
                (
                    "classificacao_cancer",
                    SurvivalTargetEncoder(
                        column="RhcLOCTUPRI",
                        time_column="Tempo",
                        event_column="delta",
                        output_column="RhcLOCTUPRI_rmst",
                        smooth=10.0,
                        cv=5,
                        random_state=42,
                    ),
                ),
                (
                    "tipo_histologico",
                    SurvivalTargetEncoder(
                        column="RhcTipoHistológico",
                        time_column="Tempo",
                        event_column="delta",
                        output_column="RhcTipoHistológico_rmst",
                        smooth=10.0,
                        cv=5,
                        random_state=42,
                    ),
                ),
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
                    "stage_treatment_interactions",
                    StageTreatmentInteractionEncoder(
                        stage_columns=[
                            "I_II",
                            "III_IV",
                        ],
                        treatment_columns=[
                            "RhcPrimeiroTratamentoRecebidoNoHospitalNenhum",
                            "RhcPrimeiroTratamentoRecebidoNoHospitalCirurgia",
                            "RhcPrimeiroTratamentoRecebidoNoHospitalRadioterapia",
                            "RhcPrimeiroTratamentoRecebidoNoHospitalQuimioterapia",
                            "RhcPrimeiroTratamentoRecebidoNoHospitalHormonioterapia",
                            "RhcPrimeiroTratamentoRecebidoNoHospitalImunoterapia",
                            "RhcPrimeiroTratamentoRecebidoNoHospitalOutras",
                            "RhcPrimeiroTratamentoRecebidoNoHospitalSemInformacao",
                        ],
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
                ("target_encoded", "passthrough", self.TARGET_ENCODED_FEATURES),
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

        y = df_model["delta"]
        df_model = self.feature_pipeline_.fit_transform(df_model, y)
        self.df_model_ = df_model

        t = df_model[self.TIME_COLUMN].values
        delta = df_model["delta"].values

        X = self.column_transformer_.fit_transform(df_model)
        X = np.asarray(X, dtype=float)

        return X, t, delta


# from __future__ import annotations

# import numpy as np
# import pandas as pd

# from typing import Dict, List, Optional, Union
# from sklearn.preprocessing import PowerTransformer, TargetEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.compose import ColumnTransformer


# class MissingIndicatorNumericEncoder(BaseEstimator, TransformerMixin):
#     def __init__(
#         self,
#         columns: List[str],
#     ):
#         self.columns = columns

#     def fit(self, X: pd.DataFrame, y=None):

#         self.medians_ = {}

#         for col in self.columns:
#             self.medians_[col] = X[col].median()

#         return self

#     def transform(self, X: pd.DataFrame):

#         X = X.copy()

#         for col in self.columns:
#             missing_name = f"{col}_missing"
#             X[missing_name] = (X[col].isna()).astype(int)
#             X[col] = X[col].fillna(self.medians_[col])

#         return X


# class StageTreatmentInteractionEncoder(BaseEstimator, TransformerMixin):
#     def __init__(
#         self,
#         stage_columns: List[str],
#         treatment_columns: List[str],
#     ):
#         self.stage_columns = stage_columns
#         self.treatment_columns = treatment_columns

#     def fit(self, X: pd.DataFrame, y=None):
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:

#         X = X.copy()

#         for stage in self.stage_columns:
#             for treatment in self.treatment_columns:
#                 interaction_name = f"{stage}_{treatment}"
#                 X[interaction_name] = X[stage].astype(int) * X[treatment].astype(int)

#         return X


# class OrdinalMapEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self, column: str, mapping: Dict[str, float]):
#         self.column = column
#         self.mapping = mapping

#     def fit(self, X: pd.DataFrame, y=None):
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         X = X.copy()
#         X[self.column] = X[self.column].map(self.mapping)
#         return X


# class MissingFlagOrdinalEncoder(BaseEstimator, TransformerMixin):
#     def __init__(
#         self,
#         column: str,
#         mapping: Dict[str, float],
#         missing_category: str,
#         missing_flag_name: str,
#     ):
#         self.column = column
#         self.mapping = mapping
#         self.missing_category = missing_category
#         self.missing_flag_name = missing_flag_name

#     def fit(self, X: pd.DataFrame, y=None):
#         mapped = X[self.column].map(self.mapping)
#         self.median_ = mapped.median()
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         X = X.copy()
#         X[self.missing_flag_name] = (X[self.column] == self.missing_category).astype(
#             int
#         )
#         X[self.column] = X[self.column].map(self.mapping)
#         X[self.column] = X[self.column].fillna(self.median_)
#         return X


# class CategoryFlagEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self, column: str, flags: Dict[str, str], drop_original: bool = True):
#         self.column = column
#         self.flags = flags
#         self.drop_original = drop_original

#     def fit(self, X: pd.DataFrame, y=None):
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         X = X.copy()
#         for new_col, category in self.flags.items():
#             X[new_col] = (X[self.column] == category).astype(int)
#         if self.drop_original:
#             X = X.drop(columns=[self.column])
#         return X


# class CategoryTargetEncoder(BaseEstimator, TransformerMixin):
#     def __init__(
#         self,
#         column: str,
#         output_column: Optional[str] = None,
#         drop_original: bool = True,
#         target_type: str = "auto",
#         smooth: Union[str, float] = "auto",
#         cv: int = 5,
#         random_state: Optional[int] = None,
#     ):
#         self.column = column
#         self.output_column = output_column
#         self.drop_original = drop_original
#         self.target_type = target_type
#         self.smooth = smooth
#         self.cv = cv
#         self.random_state = random_state

#     def _out_col(self) -> str:
#         return self.output_column or f"{self.column}_te"

#     def fit(self, X: pd.DataFrame, y=None):
#         if y is None:
#             raise ValueError(
#                 "CategoryTargetEncoder requires y (the target) to fit. "
#                 "Pass it through pipeline.fit_transform(X, y)."
#             )
#         self.encoder_ = TargetEncoder(
#             target_type=self.target_type,
#             smooth=self.smooth,
#             cv=self.cv,
#             random_state=self.random_state,
#         )
#         self.encoder_.fit(X[[self.column]], y)
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         X = X.copy()
#         encoded = self.encoder_.transform(X[[self.column]])
#         X[self._out_col()] = encoded.ravel()
#         if self.drop_original and self.column != self._out_col():
#             X = X.drop(columns=[self.column])
#         return X

#     def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
#         if y is None:
#             raise ValueError(
#                 "CategoryTargetEncoder requires y (the target) to fit_transform. "
#                 "Pass it through pipeline.fit_transform(X, y)."
#             )
#         self.encoder_ = TargetEncoder(
#             target_type=self.target_type,
#             smooth=self.smooth,
#             cv=self.cv,
#             random_state=self.random_state,
#         )
#         encoded = self.encoder_.fit_transform(X[[self.column]], y)
#         X = X.copy()
#         X[self._out_col()] = encoded.ravel()
#         if self.drop_original and self.column != self._out_col():
#             X = X.drop(columns=[self.column])
#         return X


# class BreastCancerSurvivalPreprocessor:
#     TIME_COLUMN = "Tempo"
#     STATUS_COLUMN = "Status"
#     EVENT_LABEL = "Morte por cancer de Mama"

#     RAW_FEATURES: List[str] = [
#         "Origem",
#         "RhcOrigemEncamiamento",
#         "RhcFonte",
#         "DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumorIntevalo",
#         "DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumorIntevalo",
#         "DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1ConsultaIntevalo",
#         "PndrRenda",
#         "FaixaETCAT",
#         "RhcEstadiamentoClinico",
#         "RhcIdadeNo1Consulta",
#         "DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumor",
#         "DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumor",
#         "DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1Consulta",
#         "RhcHistoricoTabaco",
#         "RhcHistoricoFamiliarCancer",
#         "RhcHistoricoAlcool",
#         "RhcPrimeiroTratamentoRecebidoNoHospitalNenhum",
#         "RhcPrimeiroTratamentoRecebidoNoHospitalCirurgia",
#         "RhcPrimeiroTratamentoRecebidoNoHospitalRadioterapia",
#         "RhcPrimeiroTratamentoRecebidoNoHospitalQuimioterapia",
#         "RhcPrimeiroTratamentoRecebidoNoHospitalHormonioterapia",
#         "RhcPrimeiroTratamentoRecebidoNoHospitalImunoterapia",
#         "RhcPrimeiroTratamentoRecebidoNoHospitalOutras",
#         "RhcPrimeiroTratamentoRecebidoNoHospitalSemInformacao",
#     ]

#     SCALED_FEATURES: List[str] = [
#         "RhcIdadeNo1Consulta",
#         "DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumor",
#         "DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumor",
#         "DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1Consulta",
#     ]

#     DUMMY_FEATURES: List[str] = [
#         "origem_sus",
#         "origem_nao_sus",
#         "origem_nenhum",
#         "sus",
#         "origem_sem_informacao",
#         "nao_sus",
#         "veio_por_conta",
#         "nao_se_aplica",
#         "laureano",
#         "hu_cg",
#         "fap",
#         "DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumor_missing",
#         "DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1Consulta_missing",
#         "1tratamento_diagnostico_tumor_ate_90dias",
#         "1tratamento_diagnostico_tumor_acima_90dias",
#         "1tratamento_consulta_ate_90dias",
#         "1tratamento_consulta_acima_90dias",
#         "1consulta_diagnostico_tumor_ate_90dias",
#         "1consulta_diagnostico_tumor_acima_90dias",
#         "baixa_renda",
#         "media_renda",
#         "alta_renda",
#         "renda_sem_informacao",
#         "20_39",
#         "40_49",
#         "50_59",
#         "60_69",
#         "70_99",
#         "I_II",
#         "III_IV",
#         "estadiamento_seminformacao",
#         "Tabaco_Sim",
#         "Tabaco_ExConsumidor",
#         "Tabaco_SemInformacao",
#         "HistFamCancer_Sim",
#         "HistFamCancer_SemInformacao",
#         "Alcool_Sim",
#         "Alcool_ExConsumidor",
#         "Alcool_SemInformacao",
#         "RhcPrimeiroTratamentoRecebidoNoHospitalNenhum",
#         "RhcPrimeiroTratamentoRecebidoNoHospitalCirurgia",
#         "RhcPrimeiroTratamentoRecebidoNoHospitalRadioterapia",
#         "RhcPrimeiroTratamentoRecebidoNoHospitalQuimioterapia",
#         "RhcPrimeiroTratamentoRecebidoNoHospitalHormonioterapia",
#         "RhcPrimeiroTratamentoRecebidoNoHospitalImunoterapia",
#         "RhcPrimeiroTratamentoRecebidoNoHospitalOutras",
#         "RhcPrimeiroTratamentoRecebidoNoHospitalSemInformacao",
#         # Stage × treatment interactions
#         "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalNenhum",
#         "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalCirurgia",
#         "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalRadioterapia",
#         "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalQuimioterapia",
#         "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalHormonioterapia",
#         "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalImunoterapia",
#         "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalOutras",
#         "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalSemInformacao",
#         "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalNenhum",
#         "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalCirurgia",
#         "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalRadioterapia",
#         "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalQuimioterapia",
#         "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalHormonioterapia",
#         "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalImunoterapia",
#         "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalOutras",
#         "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalSemInformacao",
#     ]

#     def __init__(self):
#         self.feature_pipeline_ = Pipeline(
#             steps=[
#                 (
#                     "continuous_delay_missing",
#                     MissingIndicatorNumericEncoder(
#                         columns=[
#                             "DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumor",
#                             "DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1Consulta",
#                         ]
#                     ),
#                 ),
#                 (
#                     "origem",
#                     CategoryFlagEncoder(
#                         column="Origem",
#                         flags={
#                             "origem_sus": "SUS",
#                             "origem_nao_sus": "Não SUS",
#                             "origem_nenhum": np.nan,
#                         },
#                     ),
#                 ),
#                 (
#                     "origem_encaminhamento",
#                     CategoryFlagEncoder(
#                         column="RhcOrigemEncamiamento",
#                         flags={
#                             "sus": "SUS",
#                             "origem_sem_informacao": "Sem Informação",
#                             "nao_sus": "Não SUS",
#                             "veio_por_conta": "Veio Por Conta Própria",
#                             "nao_se_aplica": "Não se Aplica",
#                         },
#                     ),
#                 ),
#                 (
#                     "fonte",
#                     CategoryFlagEncoder(
#                         column="RhcFonte",
#                         flags={"laureano": "Laureano", "fap": "FAP", "hu_cg": "HU CG"},
#                     ),
#                 ),
#                 (
#                     "1consulta_diagnostico_tumor",
#                     CategoryFlagEncoder(
#                         column="DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumorIntevalo",
#                         flags={
#                             "1consulta_diagnostico_tumor_ate_90dias": "1. até 90 dias",
#                             "1consulta_diagnostico_tumor_acima_90dias": "2. acima de 90 dias",
#                         },
#                     ),
#                 ),
#                 (
#                     "1tratamento_consulta",
#                     CategoryFlagEncoder(
#                         column="DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1ConsultaIntevalo",
#                         flags={
#                             "1tratamento_consulta_ate_90dias": "1. até 90 dias",
#                             "1tratamento_consulta_acima_90dias": "2. acima de 90 dias",
#                         },
#                     ),
#                 ),
#                 (
#                     "1tratamento_diagnostico_tumor",
#                     CategoryFlagEncoder(
#                         column="DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumorIntevalo",
#                         flags={
#                             "1tratamento_diagnostico_tumor_ate_90dias": "1. até 90 dias",
#                             "1tratamento_diagnostico_tumor_acima_90dias": "2. acima de 90 dias",
#                         },
#                     ),
#                 ),
#                 (
#                     "renda",
#                     CategoryFlagEncoder(
#                         column="PndrRenda",
#                         flags={
#                             "baixa_renda": "Baixa Renda",
#                             "media_renda": "Média Renda",
#                             "alta_renda": "Alta Renda",
#                             "renda_sem_informacao": "Sem Informação",
#                         },
#                     ),
#                 ),
#                 (
#                     "faixa_etaria",
#                     CategoryFlagEncoder(
#                         column="FaixaETCAT",
#                         flags={
#                             "20_39": "20 a 39",
#                             "40_49": "40 a 49",
#                             "50_59": "50 a 59",
#                             "60_69": "60 a 69",
#                             "70_99": "70 a 99",
#                         },
#                     ),
#                 ),
#                 (
#                     "estadiamento",
#                     CategoryFlagEncoder(
#                         column="RhcEstadiamentoClinico",
#                         flags={
#                             "I_II": "I e II",
#                             "III_IV": "III e IV",
#                             "estadiamento_seminformacao": "Sem Informação",
#                         },
#                     ),
#                 ),
#                 (
#                     "stage_treatment_interactions",
#                     StageTreatmentInteractionEncoder(
#                         stage_columns=[
#                             "I_II",
#                             "III_IV",
#                         ],
#                         treatment_columns=[
#                             "RhcPrimeiroTratamentoRecebidoNoHospitalNenhum",
#                             "RhcPrimeiroTratamentoRecebidoNoHospitalCirurgia",
#                             "RhcPrimeiroTratamentoRecebidoNoHospitalRadioterapia",
#                             "RhcPrimeiroTratamentoRecebidoNoHospitalQuimioterapia",
#                             "RhcPrimeiroTratamentoRecebidoNoHospitalHormonioterapia",
#                             "RhcPrimeiroTratamentoRecebidoNoHospitalImunoterapia",
#                             "RhcPrimeiroTratamentoRecebidoNoHospitalOutras",
#                             "RhcPrimeiroTratamentoRecebidoNoHospitalSemInformacao",
#                         ],
#                     ),
#                 ),
#                 (
#                     "tabaco",
#                     CategoryFlagEncoder(
#                         column="RhcHistoricoTabaco",
#                         flags={
#                             "Tabaco_Sim": "Sim",
#                             "Tabaco_ExConsumidor": "Ex-consumidor",
#                             "Tabaco_SemInformacao": "Sem Informação",
#                         },
#                     ),
#                 ),
#                 (
#                     "hist_familiar",
#                     CategoryFlagEncoder(
#                         column="RhcHistoricoFamiliarCancer",
#                         flags={
#                             "HistFamCancer_Sim": "Sim",
#                             "HistFamCancer_SemInformacao": "Sem Informação",
#                         },
#                     ),
#                 ),
#                 (
#                     "alcool",
#                     CategoryFlagEncoder(
#                         column="RhcHistoricoAlcool",
#                         flags={
#                             "Alcool_Sim": "Sim",
#                             "Alcool_ExConsumidor": "Ex-consumidor",
#                             "Alcool_SemInformacao": "Sem Informação",
#                         },
#                     ),
#                 ),
#             ]
#         )

#         self.column_transformer_ = ColumnTransformer(
#             transformers=[
#                 (
#                     "power",
#                     PowerTransformer(method="yeo-johnson", standardize=True),
#                     self.SCALED_FEATURES,
#                 ),
#                 ("dummy", "passthrough", self.DUMMY_FEATURES),
#             ]
#         )

#         self.df_model_: Optional[pd.DataFrame] = None

#     def _build_delta(self, df: pd.DataFrame) -> pd.Series:
#         return (df[self.STATUS_COLUMN] == self.EVENT_LABEL).astype(int)

#     def fit_transform(self, df: pd.DataFrame):
#         df = df.copy()
#         df["delta"] = self._build_delta(df)

#         cols = self.RAW_FEATURES + [self.TIME_COLUMN, "delta"]
#         df_model = df[cols].copy()

#         df_model = self.feature_pipeline_.fit_transform(df_model)
#         self.df_model_ = df_model

#         t = df_model[self.TIME_COLUMN].values
#         delta = df_model["delta"].values

#         X = self.column_transformer_.fit_transform(df_model)
#         X = np.asarray(X, dtype=float)

#         return X, t, delta
