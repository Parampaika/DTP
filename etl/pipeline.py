import pandas as pd
import os
import glob
import logging
from typing import Optional

from etl.loaders.json_loader import JsonDataLoader
from etl.processors.cleaner import CleaningProcessor
from etl.processors.temporal import TemporalProcessor
from etl.processors.tags import TagsProcessor
from etl.processors.spatial import SpatialProcessor
from etl.processors.category import CategoryProcessor
from etl.processors.conditions import ConditionsProcessor
from etl.processors.vehicles import VehicleProcessor
from etl.processors.participants import ParticipantProcessor
from etl.processors.target import TargetProcessor


class DtpPipeline:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger("DtpPipeline")
        self.loader = JsonDataLoader()

        self.processors = [
            CleaningProcessor(self.config),
            TemporalProcessor(self.config),
            TagsProcessor(self.config),
            SpatialProcessor(self.config),
            CategoryProcessor(self.config),
            ConditionsProcessor(self.config),
            VehicleProcessor(self.config),
            ParticipantProcessor(self.config),
            TargetProcessor(self.config),
        ]

    def process_file(self, file_path: str) -> pd.DataFrame:
        filename = os.path.basename(file_path)
        separator_line = "=" * 80
        header_msg = f"===== ЗАПУСК ОБРАБОТКИ ФАЙЛА: {filename} ====="

        self.logger.info(separator_line)
        self.logger.info(header_msg)
        self.logger.info(separator_line)
        df = self.loader.load(file_path)
        if df.empty:
            return pd.DataFrame()

        for processor in self.processors:
            try:
                df = processor.process(df)
                if df.empty:
                    break
            except Exception as e:
                self.logger.critical(
                    f"Ошибка в {processor.__class__.__name__}: {e}", exc_info=True
                )
                return pd.DataFrame()
        return df

    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("--- ЗАПУСК ФИНАЛЬНОЙ ЗАЧИСТКИ ---")

        if not df.columns.is_unique:
            self.logger.warning(
                "Обнаружены дублирующиеся названия колонок! Удаляем дубли..."
            )
            df = df.loc[:, ~df.columns.duplicated()]

        initial_cols = len(df.columns)
        cols_to_drop = []

        for col in df.columns:
            if col in ["target", "region_id"]:
                continue

            try:
                unique_count = df[col].nunique(dropna=False)
            except TypeError:
                unique_count = df[col].astype(str).nunique(dropna=False)

            if unique_count <= 1:
                cols_to_drop.append(col)

        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            self.logger.info(
                f"Удалено {len(cols_to_drop)} константных колонок: {cols_to_drop}"
            )
        else:
            self.logger.info("Константных колонок не найдено.")

        self.logger.info(f"Итого колонок: {initial_cols} -> {len(df.columns)}")
        return df

    def run_directory(self, input_dir: str, output_file: str):
        files = (
            glob.glob(os.path.join(input_dir, "*.zip"))
            + glob.glob(os.path.join(input_dir, "*.geojson"))
            + glob.glob(os.path.join(input_dir, "*.json"))
        )

        if not files:
            self.logger.error("Файлов не найдено.")
            return

        all_dfs = []
        for f in files:
            df = self.process_file(f)
            if not df.empty:
                all_dfs.append(df)

        if all_dfs:
            self.logger.info("Объединение датафреймов...")
            full_df = pd.concat(all_dfs, ignore_index=True)

            num_cols = full_df.select_dtypes(include=["number"]).columns
            full_df[num_cols] = full_df[num_cols].fillna(0)

            full_df = self._final_cleanup(full_df)

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            full_df.to_parquet(output_file, index=False)
            self.logger.info(f"Готово! Сохранено в {output_file}")
        else:
            self.logger.warning("Результат пуст.")
