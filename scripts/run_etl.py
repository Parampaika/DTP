import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from etl.utils import load_config, setup_logging
from etl.pipeline import DtpPipeline

def main():
    config_path = "config.yaml"
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        return

    setup_logging(config)
    logger = logging.getLogger("RunETL")
    
    logger.info("Запуск ETL процесса...")
    logger.info(f"Конфигурация: {config['etl']}")

    pipeline = DtpPipeline(config=config['etl'])
    
    input_dir = config['paths']['input_dir']
    output_file = config['paths']['output_file']
    
    pipeline.run_directory(
        input_dir=input_dir,
        output_file=output_file
    )

if __name__ == "__main__":
    main()