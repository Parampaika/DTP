import yaml
import logging
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd


def find_project_root(marker: str = "configs", start_path: Optional[Path] = None) -> Path:
    current = (start_path or Path.cwd()).resolve()

    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent

    raise FileNotFoundError(
        f"Не удалось найти корень проекта (папка '{marker}' не найдена). "
        f"Стартовый путь: {current}"
    )


def load_yaml(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Ошибка при чтении YAML {config_path}: {e}")


def deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in extra.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_full_config(
    config_files: Optional[List[str]] = None,
    start_path: Optional[Path] = None
) -> Dict[str, Any]:
    root = find_project_root(start_path=start_path)

    default_files = [
        "configs/logging.yaml",
        "configs/data/paths.yaml",
        "configs/data/datasets.yaml",
        "configs/data/dtp_etl.yaml",
    ]

    config = {}
    for rel_path in (config_files or default_files):
        cfg_part = load_yaml(root / rel_path)
        config = deep_update(config, cfg_part)

    return config


def setup_logging(config: Dict[str, Any], start_path: Optional[Path] = None):
    log_cfg = config.get("logging", {})

    level_str = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    fmt = log_cfg.get(
        "format",
        "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
    )
    datefmt = log_cfg.get("datefmt", "%H:%M:%S")

    formatter = logging.Formatter(fmt, datefmt)

    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_filename = log_cfg.get("filename")
    if log_filename:
        root = find_project_root(start_path=start_path)
        log_dir_name = log_cfg.get("log_dir", "logs")
        log_dir = root / log_dir_name
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / log_filename
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_dataset_path(dataset_key: str, config: Dict[str, Any], start_path: Optional[Path] = None) -> Path:
    datasets_cfg = config.get("datasets", {})
    if dataset_key not in datasets_cfg:
        raise KeyError(f"Dataset key '{dataset_key}' не найден в config['datasets']")
    return find_project_root(start_path=start_path) / datasets_cfg[dataset_key]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError(f"Неподдерживаемый формат файла: {path}")


def write_table(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in [".xlsx", ".xls"]:
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Неподдерживаемый формат файла для сохранения: {path}")


def load_dataset(dataset_key: str, config: Dict[str, Any], start_path: Optional[Path] = None) -> pd.DataFrame:
    path = get_dataset_path(dataset_key, config, start_path=start_path)
    if not path.exists():
        raise FileNotFoundError(f"Датасет не найден: {path}")
    return read_table(path)


def save_dataset(df: pd.DataFrame, dataset_key: str, config: Dict[str, Any], start_path: Optional[Path] = None) -> Path:
    path = get_dataset_path(dataset_key, config, start_path=start_path)
    write_table(df, path)
    return path