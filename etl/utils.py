import yaml
import logging
import sys
import os
from typing import Dict, Any
from pathlib import Path

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Загружает конфигурацию из YAML файла.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Ошибка при чтении YAML: {e}")

def setup_logging(config):
    log_cfg = config.get('logging', {})
    
    level_str = log_cfg.get('level', 'INFO').upper()
    level = getattr(logging, level_str, logging.INFO)
    
    fmt = log_cfg.get('format', "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s")
    datefmt = log_cfg.get('datefmt', "%H:%M:%S")
    
    formatter = logging.Formatter(fmt, datefmt)
    
    logger = logging.getLogger()
    logger.setLevel(level)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    log_filename = log_cfg.get('filename')
    
    if log_filename:
        log_dir_name = log_cfg.get('log_dir', 'logs')
        
        log_dir = Path(os.getcwd()) / log_dir_name
        
        os.makedirs(log_dir, exist_ok=True)
        log_path = log_dir / log_filename
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)