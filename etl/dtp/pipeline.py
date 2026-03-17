import pandas as pd
import os
import glob
import logging
from typing import Optional, List

from etl.dtp.loaders.json_loader import JsonDataLoader
from etl.dtp.processors.cleaner import CleaningProcessor
from etl.dtp.processors.temporal import TemporalProcessor
from etl.dtp.processors.tags import TagsProcessor
from etl.dtp.processors.spatial import SpatialProcessor
from etl.dtp.processors.category import CategoryProcessor
from etl.dtp.processors.conditions import ConditionsProcessor
from etl.dtp.processors.vehicles import VehicleProcessor
from etl.dtp.processors.participants import ParticipantProcessor
from etl.dtp.processors.target import TargetProcessor


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
            self.logger.warning(f"Файл {filename} дал пустой DataFrame после загрузки.")
            return pd.DataFrame()

        for processor in self.processors:
            try:
                self.logger.info(f"Обработка: {processor.__class__.__name__}")
                df = processor.process(df)
                if df.empty:
                    self.logger.warning(
                        f"После {processor.__class__.__name__} DataFrame стал пустым."
                    )
                    break
            except Exception as e:
                self.logger.critical(
                    f"Ошибка в {processor.__class__.__name__}: {e}",
                    exc_info=True
                )
                return pd.DataFrame()

        if not df.columns.is_unique:
            self.logger.warning(f"Файл {filename}: Обнаружены дублирующиеся названия колонок. Удаляем...")
            df = df.loc[:, ~df.columns.duplicated()]

        self.logger.info(f"Файл {filename} обработан. Итоговая форма: {df.shape}")
        return df

    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("--- ЗАПУСК ФИНАЛЬНОЙ ЗАЧИСТКИ ---")

        if not df.columns.is_unique:
            self.logger.warning(
                "Обнаружены дублирующиеся названия колонок. Удаляем дубли..."
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
            df = df.drop(columns=cols_to_drop)
            self.logger.info(
                f"Удалено {len(cols_to_drop)} константных колонок: {cols_to_drop}"
            )
        else:
            self.logger.info("Константных колонок не найдено.")

        self.logger.info(f"Итого колонок: {initial_cols} -> {len(df.columns)}")
        return df

    def run_files(self, file_paths: List[str], output_file: str):
        existing_files = []
        for fp in file_paths:
            if os.path.exists(fp):
                existing_files.append(fp)
            else:
                self.logger.warning(f"Файл не найден и будет пропущен: {fp}")

        if not existing_files:
            self.logger.error("Нет ни одного существующего файла для обработки.")
            return

        self.logger.info(f"Найдено файлов для обработки: {len(existing_files)}")

        all_dfs = []
        for f in existing_files:
            df = self.process_file(f)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            self.logger.warning("После обработки все датафреймы пустые.")
            return

        self.logger.info("Объединение датафреймов...")
        full_df = pd.concat(all_dfs, ignore_index=True)
        self.logger.info(f"После concat shape: {full_df.shape}")

        if self.config.get("fill_numeric_na_with_zero", True):
            num_cols = full_df.select_dtypes(include=["number"]).columns
            full_df[num_cols] = full_df[num_cols].fillna(0)
            self.logger.info(f"Числовые NaN заполнены нулями. Колонок: {len(num_cols)}")

        if self.config.get("drop_constant_columns", True):
            full_df = self._final_cleanup(full_df)

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        full_df.to_parquet(output_file, index=False)
        self.logger.info(f"Готово! Сохранено в {output_file}")
        self.logger.info(f"Итоговый dataset shape: {full_df.shape}")

    def run_directory(self, input_dir: str, output_file: str):
        extensions = self.config.get("input_extensions", [".zip", ".geojson", ".json"])
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))

        files = sorted(set(files))

        if not files:
            self.logger.error(f"В папке {input_dir} не найдено файлов с расширениями {extensions}.")
            return

        self.logger.info(f"Найдено {len(files)} файлов в директории {input_dir}")
        self.run_files(files, output_file)