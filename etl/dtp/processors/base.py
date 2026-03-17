import pandas as pd
import re
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, Set, List
from contextlib import contextmanager

class BaseDataProcessor(ABC):
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def validate_columns(self, df: pd.DataFrame, columns: list) -> bool:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            self.logger.warning(f"Нет колонок: {missing}. Пропуск.")
            return False
        return True

    @staticmethod
    def normalize_text(s: Any) -> str:
        if not isinstance(s, str): return ""
        s = s.lower().replace('ё', 'е')
        s = re.sub(r'[^а-яa-z0-9\s]', '', s)
        s = re.sub(r'\s+', '_', s)
        return s.strip('_')

    def safe_map_categorical(self, df: pd.DataFrame, col: str, mapping: Dict[str, str], whitelist: Set[str], default: str = 'other'):
        """
        Безопасная обработка одиночной категориальной колонки.
        1. Нормализация.
        2. Маппинг (замена редких).
        3. Проверка по WhiteList.
        4. Замена неизвестных на default и логирование.
        """
        if col not in df.columns: return df

        df[col] = df[col].apply(self.normalize_text)
        df[col] = df[col].replace(mapping)

        actual_values = set(df[col].unique())
        unknown_values = actual_values - whitelist - {default}

        if unknown_values:
            mask = df[col].isin(unknown_values)
            count = mask.sum()
            percentage = count / len(df) * 100
            values_str = ", ".join(map(str, list(unknown_values)[:5]))
            if len(unknown_values) > 5:
                values_str += ", ..."
            
            self.logger.warning(
                f"Колонка '{col}': {len(unknown_values)} новых значений вне Whitelist → заменены на '{default}'. "
                f"Примеры: [{values_str}]. "
                f"Затронуто {count} строк ({percentage:.2f}%)."
            )
            
            df.loc[mask, col] = default
        
        return df

    def safe_expand_list(self, df: pd.DataFrame, col: str, mapping: Dict[str, str], prefix: str) -> pd.DataFrame:
        """
        Безопасная обработка колонок со списками (weather, road).
        Разворачивает в флаги, ловит неизвестные значения и кидает их в _other.
        Если _other оказался пустым (все нули), колонка удаляется.
        """
        if col not in df.columns: return df

        target_flags = sorted(list(set(mapping.values())))
        if 'other' not in target_flags: target_flags.append('other')

        new_cols_data = {f"{prefix}_{flag}": [0] * len(df) for flag in target_flags}
        
        source_data = df[col].tolist()
        unknown_tags_collected = set()
        count_rows_with_other = 0

        for i, item_list in enumerate(source_data):
            if not isinstance(item_list, list): continue
            
            has_other = False
            
            for raw_val in item_list:
                clean_val = self.normalize_text(raw_val)
                found_flag = None
                
                if clean_val in mapping:
                    found_flag = mapping[clean_val]
                else:
                    for key, target in mapping.items():
                        if key in clean_val:
                            found_flag = target
                            break
                
                if found_flag:
                    new_cols_data[f"{prefix}_{found_flag}"][i] = 1
                else:
                    new_cols_data[f"{prefix}_other"][i] = 1
                    has_other = True
                    if clean_val: unknown_tags_collected.add(clean_val)
            
            if has_other:
                count_rows_with_other += 1

        if unknown_tags_collected:
            example_tags = ", ".join(list(unknown_tags_collected)[:5])  # не более 5 примеров
            if len(unknown_tags_collected) > 5:
                example_tags += ", ..."
            percentage = count_rows_with_other / len(df) * 100
            self.logger.warning(
                f"Колонка '{col}': {len(unknown_tags_collected)} неизвестных тегов → '{prefix}_other'. "
                f"Примеры: [{example_tags}]. "
                f"Затронуто {count_rows_with_other} строк ({percentage:.2f}%)."
            )

        new_df = pd.DataFrame(new_cols_data, index=df.index)
        
        df = pd.concat([df, new_df], axis=1)
        df.drop(columns=[col], inplace=True)
        return df
    
    @contextmanager
    def activity(self, description: str):
        """
        Менеджер контекста.
        Делает логи:
        >>> [НАЧАЛО] Описание...
        <<< [КОНЕЦ] Описание. Время: X сек.
        А также ловит ошибки, чтобы пайплайн не падал молча.
        """
        start_time = time.time()
        self.logger.info(f">>> [НАЧАЛО] {description}...")
        
        try:
            yield
        except Exception as e:
            self.logger.error(f"!!! [ОШИБКА] В блоке '{description}': {e}", exc_info=True)
            raise e
        finally:
            elapsed = time.time() - start_time
            self.logger.info(f"<<< [КОНЕЦ] {description}. Время: {elapsed:.2f} сек.")