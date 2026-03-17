import sys
import os
import logging
import pandas as pd

# Добавляем корень проекта для путей
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from etl.utils import load_full_config, setup_logging, find_project_root
from etl.merge.enricher import EconomicEnricher

def main():
    # Загружаем ВСЕ нужные конфиги (включая modeling, чтобы прочитать target_mode)
    config = load_full_config([
        "configs/logging.yaml",
        "configs/data/datasets.yaml",
        "configs/data/paths.yaml",
        "configs/modeling/base.yaml"
    ])

    setup_logging(config)
    logger = logging.getLogger("BuildMergedDataset")
    ROOT = find_project_root()

    logger.info("=== 1. ПРОВЕРКА ПУТЕЙ И ЧТЕНИЕ ДАННЫХ ===")
    
    # Берем пути из datasets.yaml
    dtp_path = ROOT / config["datasets"]["dtp_current"]
    econ_path = ROOT / config["datasets"]["econom_current"]
    
    out_micro_path = ROOT / config["datasets"]["merged_micro"]
    out_macro_path = ROOT / config["datasets"]["merged_macro"]

    if not dtp_path.exists():
        logger.error(f"DTP dataset не найден: {dtp_path}")
        return
    if not econ_path.exists():
        logger.error(f"ECON dataset не найден: {econ_path}")
        return

    logger.info("Загрузка DataFrame в память...")
    df_dtp = pd.read_parquet(dtp_path)
    df_econ = pd.read_parquet(econ_path)
    logger.info(f"DTP: {df_dtp.shape} | ECON: {df_econ.shape}")

    # Инициализация нашего ETL-процессора
    enricher = EconomicEnricher(config=config, logger=logger)
    
    # Запуск магии
    df_micro, df_macro = enricher.build_datasets(df_dtp, df_econ)

    logger.info("=== 5. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===")
    out_micro_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_micro.to_parquet(out_micro_path, index=False)
    logger.info(f"MICRO датасет сохранен: {out_micro_path}")
    
    df_macro.to_parquet(out_macro_path, index=False)
    logger.info(f"MACRO датасет сохранен: {out_macro_path}")
    
    logger.info("Пайплайн слияния завершен успешно!")

if __name__ == "__main__":
    main()