import os
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from etl.utils import load_full_config, setup_logging, find_project_root
from etl.econom.pipeline import EconomPipeline


def main():
    config = load_full_config([
        "configs/logging.yaml",
        "configs/data/datasets.yaml",
        "configs/data/econom_etl.yaml",
    ])

    setup_logging(config)
    logger = logging.getLogger("BuildEconomDataset")
    root = find_project_root()

    if "econom_etl" not in config:
        raise KeyError("В конфиге отсутствует секция 'econom_etl'")

    econ_cfg = config["econom_etl"]

    source_file = root / econ_cfg["source_file"]

    output_dataset_key = econ_cfg.get("output_dataset_key", "econom_current")
    if output_dataset_key not in config.get("datasets", {}):
        raise KeyError(
            f"Ключ датасета '{output_dataset_key}' не найден в configs/data/datasets.yaml"
        )

    output_file = root / config["datasets"][output_dataset_key]

    logger.info("Запуск сборки ECON dataset...")
    logger.info(f"Читаем из: {source_file}")
    logger.info(f"Сохраняем в: {output_file}")

    pipeline = EconomPipeline(config=econ_cfg)
    df = pipeline.run(source_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)

    logger.info(f"ECON dataset сохранён: {output_file}")
    logger.info(f"Итоговый shape: {df.shape}")


if __name__ == "__main__":
    main()