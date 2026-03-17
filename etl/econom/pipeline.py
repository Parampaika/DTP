import logging
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd


class EconomPipeline:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger("EconomPipeline")

        self.required_columns: List[str] = self.config.get("required_columns", [])
        self.key_columns: List[str] = self.config.get(
            "key_columns",
            ["region_name", "mun_district", "municipality", "year"]
        )
        self.string_columns: List[str] = self.config.get(
            "string_columns",
            ["region_name", "mun_district", "municipality"]
        )
        self.numeric_columns: List[str] = self.config.get("numeric_columns", [])
        self.year_column: str = self.config.get("year_column", "year")

        self.na_values = {
            str(x).strip().lower()
            for x in self.config.get("na_values", ["Н/Д", "н/д", "Н/д", "н/Д"])
        }

        self.drop_rows_with_missing_keys: bool = bool(
            self.config.get("drop_rows_with_missing_keys", True)
        )
        self.fail_on_duplicate_keys: bool = bool(
            self.config.get("fail_on_duplicate_keys", True)
        )
        self.sort_by_keys: bool = bool(
            self.config.get("sort_by_keys", True)
        )

    def _load_source(self, source_path: Path) -> pd.DataFrame:
        if not source_path.exists():
            raise FileNotFoundError(f"ECON source file не найден: {source_path}")

        suffix = source_path.suffix.lower()

        if suffix in [".xlsx", ".xls"]:
            sheet_name = self.config.get("sheet_name", 0)
            df = pd.read_excel(source_path, sheet_name=sheet_name)
        elif suffix == ".csv":
            df = pd.read_csv(source_path)
        else:
            raise ValueError(f"Неподдерживаемый формат ECON source: {source_path}")

        self.logger.info(f"Загружен ECON source: {source_path}")
        self.logger.info(f"RAW shape: {df.shape}")
        return df

    def _validate_required_columns(self, df: pd.DataFrame):
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(
                "В ECON dataset отсутствуют обязательные колонки: "
                f"{missing}"
            )

    def _normalize_na_value(self, value):
        if pd.isna(value):
            return np.nan

        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lower() in self.na_values:
                return np.nan
            return stripped

        return value

    def _clean_string_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.string_columns:
            if col not in df.columns:
                continue

            series = df[col].map(self._normalize_na_value)
            series = series.map(
                lambda x: " ".join(x.split()) if isinstance(x, str) else x
            )
            series = series.mask(series == "", np.nan)
            df[col] = series

        return df

    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.numeric_columns:
            if col not in df.columns:
                continue

            series = df[col].map(self._normalize_na_value)
            series = series.map(
                lambda x: x.replace(" ", "").replace(",", ".")
                if isinstance(x, str)
                else x
            )
            df[col] = pd.to_numeric(series, errors="coerce")

        return df

    def _clean_year_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.year_column not in df.columns:
            raise KeyError(f"Колонка года '{self.year_column}' не найдена")

        series = df[self.year_column].map(self._normalize_na_value)
        series = series.map(
            lambda x: x.replace(" ", "").replace(",", ".")
            if isinstance(x, str)
            else x
        )

        df[self.year_column] = pd.to_numeric(series, errors="coerce").astype("Int64")
        return df

    def _drop_fully_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(how="all").copy()
        dropped = before - len(df)

        if dropped > 0:
            self.logger.warning(f"Удалено полностью пустых строк: {dropped}")

        return df

    def _drop_rows_with_missing_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        existing_keys = [col for col in self.key_columns if col in df.columns]

        if not existing_keys or not self.drop_rows_with_missing_keys:
            return df

        before = len(df)
        df = df.dropna(subset=existing_keys).copy()
        dropped = before - len(df)

        if dropped > 0:
            self.logger.warning(
                f"Удалено строк с пропусками в ключах {existing_keys}: {dropped}"
            )

        return df

    def _validate_duplicates(self, df: pd.DataFrame):
        existing_keys = [col for col in self.key_columns if col in df.columns]
        if not existing_keys:
            return

        dup_mask = df.duplicated(subset=existing_keys, keep=False)
        dup_rows = int(dup_mask.sum())

        if dup_rows == 0:
            self.logger.info(
                f"Дубликатов по ключу {existing_keys} не найдено."
            )
            return

        dup_info = (
            df.groupby(existing_keys)
            .size()
            .reset_index(name="n")
            .query("n > 1")
            .sort_values("n", ascending=False)
        )

        msg = (
            f"Найдены дубликаты по ключу {existing_keys}. "
            f"Количество строк в дубликатах: {dup_rows}\n"
            f"Топ дубликатов:\n{dup_info.head(20).to_string(index=False)}"
        )

        if self.fail_on_duplicate_keys:
            raise ValueError(msg)

        self.logger.warning(msg)

    def _log_summary(self, df: pd.DataFrame, stage: str):
        self.logger.info(f"[{stage}] shape: {df.shape}")

        for col in ["region_name", "mun_district", "municipality"]:
            if col in df.columns:
                self.logger.info(
                    f"[{stage}] unique {col}: {df[col].nunique(dropna=True)}"
                )

        if self.year_column in df.columns:
            years = df[self.year_column].dropna()
            if len(years) > 0:
                self.logger.info(
                    f"[{stage}] years: min={int(years.min())}, "
                    f"max={int(years.max())}, "
                    f"n_unique={years.nunique()}"
                )

        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            self.logger.info(f"[{stage}] missing values:\n{missing.to_string()}")
        else:
            self.logger.info(f"[{stage}] missing values: нет")

    def run(self, source_path: str | Path) -> pd.DataFrame:
        source_path = Path(source_path)

        df = self._load_source(source_path)
        self._validate_required_columns(df)
        self._log_summary(df, "RAW")

        df = self._drop_fully_empty_rows(df)
        df = self._clean_string_columns(df)
        df = self._clean_numeric_columns(df)
        df = self._clean_year_column(df)
        df = self._drop_rows_with_missing_keys(df)

        self._validate_required_columns(df)

        if self.sort_by_keys:
            existing_keys = [col for col in self.key_columns if col in df.columns]
            if existing_keys:
                df = df.sort_values(existing_keys).reset_index(drop=True)

        self._validate_duplicates(df)
        self._log_summary(df, "CLEAN")

        return df