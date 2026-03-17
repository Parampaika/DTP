import sys
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from etl.utils import load_full_config, setup_logging
from etl.dtp.pipeline import DtpPipeline


def main():
    config = load_full_config(
        config_files=[
            "configs/logging.yaml",
            "configs/data/paths.yaml",
            "configs/data/datasets.yaml",
            "configs/data/dtp_etl.yaml",
        ],
        start_path=PROJECT_ROOT
    )

    setup_logging(config, start_path=PROJECT_ROOT)
    logger = logging.getLogger("BuildDtpDataset")

    input_dir = PROJECT_ROOT / config["paths"]["need_process_dir"]
    output_file = PROJECT_ROOT / config["datasets"]["dtp_current"]

    logger.info("Запуск сборки DTP dataset...")
    logger.info(f"Читаем из: {input_dir}")
    logger.info(f"Сохраняем в: {output_file}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Папка входных данных не найдена: {input_dir}")

    pipeline = DtpPipeline(config=config["dtp_etl"])
    pipeline.run_directory(
        input_dir=str(input_dir),
        output_file=str(output_file)
    )

    logger.info("Сборка DTP dataset завершена.")


if __name__ == "__main__":
    main()