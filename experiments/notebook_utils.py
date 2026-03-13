import os
import sys
import yaml
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import joblib
import json
import torch
from catboost import CatBoostClassifier


def find_project_root(marker: str = "config.yaml") -> Path:
    """
    Ищет корень проекта, поднимаясь вверх от текущей директории.
    """
    current = Path.cwd().resolve()

    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent

    raise FileNotFoundError(
        f"Не удалось найти корень проекта (файл {marker} не найден)."
    )


def setup_env() -> Tuple[Path, Dict]:
    root = find_project_root()
    
    if str(root) not in sys.path:
        sys.path.append(str(root))
        
    config_path = root / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return root, config

def transform_target(df: pd.DataFrame, target_col: str, mode: str) -> pd.DataFrame:
    """
    Трансформирует целевую переменную в зависимости от режима.
    """
    df_mod = df.copy()

    if mode == "multiclass":
        pass  # 0, 1, 2
    elif mode == "binary_severe":
        # 0 (Легко/Нет) vs 1 (Тяжко/Смерть)
        df_mod[target_col] = df_mod[target_col].replace({2: 1})
    elif mode == "binary_fatal":
        # 0 (Выжил) vs 1 (Погиб)
        df_mod[target_col] = df_mod[target_col].apply(lambda x: 1 if x == 2 else 0)
    else:
        raise ValueError(f"Неизвестный режим target_mode: {mode}")

    return df_mod


def load_data_for_modeling(
    config: Dict,
    project_root: Path,
    data_source: str = "full_df",
    fill_cats: bool = True,
    need_drop: bool = True
) -> pd.DataFrame:
    """
    Универсальная загрузка данных для моделирования.
    """
    # 1. Определяем путь к файлу
    paths_cfg = config.get("paths", {})

    if data_source in paths_cfg:
        # Если передан ключ из конфига (напр. "enriched_df")
        file_path = paths_cfg[data_source]
        data_path = project_root / file_path
    else:
        # Если передан путь напрямую (напр. "data/processed/temp.parquet")
        data_path = project_root / data_source

    if not data_path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {data_path}\nПроверь ключ '{data_source}' в конфиге или путь."
        )

    print(f"Загрузка данных из: {data_path}")

    # 2. Чтение
    # Поддержка разных форматов на всякий случай
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        # Fallback для excel и прочего
        try:
            df = pd.read_parquet(data_path)
        except:
            df = pd.read_excel(data_path)

    # 3. Чтение параметров признаков
    feats = config.get("features", {})
    drop_cols = feats.get("drop_cols", []) if need_drop else []
    cat_cols = feats.get("cat_cols", [])
    target_col = feats.get("target_col", "target")
    target_mode = config.get("experiment", {}).get("target_mode", "multiclass")

    # 4. Удаление лишнего
    existing_drop = [c for c in drop_cols if c in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)

    # 5. Трансформация таргета
    if target_col in df.columns:
        df = transform_target(df, target_col, target_mode)
        print(
            f"Режим таргета: {target_mode}. Распределение:\n{df[target_col].value_counts(normalize=True).round(4)}"
        )
    else:
        print(
            f"Внимание: Таргет '{target_col}' не найден (возможно, это тестовая выборка)."
        )

    # 6. Обработка категорий (CatBoost-friendly)
    if fill_cats:
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("Missing")
                if df[col].nunique() == 0:
                    df[col] = "Empty"

    return df


class ExperimentManager:
    def __init__(self, project_root: Path, config: dict):
        self.root = project_root
        self.config = config
        self.exp_name = config.get("current_experiment", "default_exp")
        self.base_dir = config.get("experiments_config", {}).get("base_dir", "res")
        self.exp_path = self.root / self.base_dir / self.exp_name

    def get_paths(self, model_name: str = None, stage: str = None):
        """
        stage: Название этапа (например, '02_simple_models')
        model_name: Название модели (например, 'LogisticRegression')
        """
        # Базовые пути эксперимента: res/01_dtp_stat/models
        models_path = self.exp_path / "models"
        reports_path = self.exp_path / "reports"

        # Добавляем этап (если есть): res/01_dtp_stat/models/02_simple_models
        if stage:
            models_path = models_path / stage
            reports_path = reports_path / stage

        # Добавляем модель (если есть): .../02_simple_models/LogisticRegression
        if model_name:
            models_path = models_path / model_name
            reports_path = reports_path / model_name

        # Создаем папки
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(reports_path, exist_ok=True)

        return {"root": self.exp_path, "model": models_path, "report": reports_path}


# Хелпер для быстрой инициализации
def get_exp_manager():
    root, cfg = setup_env()
    return ExperimentManager(root, cfg)


# ==========================================
# 1. SCIKIT-LEARN (Joblib)
# ==========================================


def save_sklearn_model(model, exp_manager, model_name: str, stage: str = None):
    paths = exp_manager.get_paths(model_name=model_name, stage=stage)
    save_path = paths["model"] / "model.joblib"
    joblib.dump(model, save_path)
    print(f"Модель сохранена: {save_path}")


def load_sklearn_model(exp_manager, model_name: str, stage: str = None):
    paths = exp_manager.get_paths(model_name=model_name, stage=stage)
    load_path = paths["model"] / "model.joblib"

    if not load_path.exists():
        fallback_paths = exp_manager.get_paths(model_name=model_name, stage=None)
        fallback_load_path = fallback_paths["model"] / "model.joblib"
        if fallback_load_path.exists():
            print(
                f"Warning: Модель найдена по старому пути (без stage): {fallback_load_path}"
            )
            return joblib.load(fallback_load_path)

        raise FileNotFoundError(f"Модель не найдена: {load_path}")

    return joblib.load(load_path)


# ==========================================
# 2. CATBOOST (.cbm)
# ==========================================


def save_catboost_model(model, exp_manager, model_name: str, stage: str = None):
    paths = exp_manager.get_paths(model_name=model_name, stage=stage)
    save_path = paths["model"] / "model.cbm"
    model.save_model(str(save_path))
    print(f"CatBoost сохранен: {save_path}")


def load_catboost_model(exp_manager, model_name: str, stage: str = None):
    paths = exp_manager.get_paths(model_name=model_name, stage=stage)
    load_path = paths["model"] / "model.cbm"

    if not load_path.exists():
        # Fallback
        fallback_paths = exp_manager.get_paths(model_name=model_name, stage=None)
        fallback_load_path = fallback_paths["model"] / "model.cbm"
        if fallback_load_path.exists():
            print(f"Warning: Модель найдена по старому пути: {fallback_load_path}")
            model = CatBoostClassifier()
            model.load_model(str(fallback_load_path))
            return model

        raise FileNotFoundError(f"Модель не найдена: {load_path}")

    model = CatBoostClassifier()
    model.load_model(str(load_path))
    return model


# ==========================================
# 3. PYTORCH (.pth + config)
# ==========================================


def save_pytorch_model(
    model, model_config, exp_manager, model_name: str, stage: str = None
):
    paths = exp_manager.get_paths(model_name=model_name, stage=stage)
    torch.save(model.state_dict(), paths["model"] / "weights.pth")
    with open(paths["model"] / "config.json", "w") as f:
        json.dump(model_config, f)
    print(f"PyTorch сохранен: {paths['model']}")


def load_pytorch_model(exp_manager, model_name: str, ModelClass, stage: str = None):
    paths = exp_manager.get_paths(model_name=model_name, stage=stage)
    weights_path = paths["model"] / "weights.pth"
    config_path = paths["model"] / "config.json"

    # Fallback логика для PyTorch чуть сложнее, поэтому упростим:
    if not weights_path.exists():
        fallback_paths = exp_manager.get_paths(model_name=model_name, stage=None)
        weights_path = fallback_paths["model"] / "weights.pth"
        config_path = fallback_paths["model"] / "config.json"

        if not weights_path.exists():
            raise FileNotFoundError(f"Веса не найдены: {weights_path}")
        print(f"Warning: Загружено по старому пути: {weights_path}")

    # 1. Грузим конфиг
    with open(config_path, "r") as f:
        config = json.load(f)

    # 2. Инициализируем архитектуру
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelClass(**config).to(device)

    # 3. Грузим веса
    try:
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
    except TypeError:
        # Для старых версий torch
        model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    return model
