import sys
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

from etl.utils import find_project_root, load_full_config


def setup_env() -> Tuple[Path, Dict]:
    root = find_project_root()

    if str(root) not in sys.path:
        sys.path.append(str(root))

    config = load_full_config()
    return root, config


def transform_target(df: pd.DataFrame, target_col: str, mode: str) -> pd.DataFrame:
    df_mod = df.copy()

    if mode == "multiclass":
        pass
    elif mode == "binary_severe":
        df_mod[target_col] = df_mod[target_col].replace({2: 1})
    elif mode == "binary_fatal":
        df_mod[target_col] = df_mod[target_col].apply(lambda x: 1 if x == 2 else 0)
    else:
        raise ValueError(f"Неизвестный режим target_mode: {mode}")

    return df_mod


def load_data_for_modeling(
    config: Dict,
    project_root: Path,
    data_source: str = "dtp_full",
    fill_cats: bool = True,
    need_drop: bool = True
) -> pd.DataFrame:
    datasets_cfg = config.get("datasets", {})

    if data_source in datasets_cfg:
        data_path = project_root / datasets_cfg[data_source]
    else:
        data_path = project_root / data_source

    if not data_path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {data_path}\n"
            f"Проверь ключ '{data_source}' в config['datasets'] или путь."
        )

    print(f"Загрузка данных из: {data_path}")

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        try:
            df = pd.read_parquet(data_path)
        except Exception:
            df = pd.read_excel(data_path)

    feats = config.get("features", {})
    drop_cols = feats.get("drop_cols", []) if need_drop else []
    cat_cols = feats.get("cat_cols", [])
    target_col = feats.get("target_col", "target")
    target_mode = config.get("modeling", {}).get("target_mode", "multiclass")

    existing_drop = [c for c in drop_cols if c in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)

    if target_col in df.columns:
        df = transform_target(df, target_col, target_mode)
        print(
            f"Режим таргета: {target_mode}. Распределение:\n"
            f"{df[target_col].value_counts(normalize=True).round(4)}"
        )
    else:
        print(f"Внимание: Таргет '{target_col}' не найден.")

    if fill_cats:
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Missing").astype(str)

    return df