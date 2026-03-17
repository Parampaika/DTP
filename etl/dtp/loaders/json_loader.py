import json
import os
import logging
import pandas as pd
import zipfile
from typing import Optional, Any

class JsonDataLoader:
    """
    Класс для загрузки данных.
    Поддерживает: .json, .geojson и .zip (содержащие json/geojson).
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _read_data(self, file_path: str) -> Any:
        """
        Внутренний метод: читает файл (обычный или zip) и возвращает dict/list.
        """
        # СЛУЧАЙ 1: ZIP-архив
        if file_path.endswith('.zip'):
            try:
                with zipfile.ZipFile(file_path, 'r') as z:
                    # Ищем подходящий файл внутри архива
                    # Берем первый попавшийся .geojson или .json
                    file_list = z.namelist()
                    target_file = next((f for f in file_list if f.endswith(('.geojson', '.json'))), None)
                    
                    if not target_file:
                        self.logger.error(f"В архиве {file_path} нет файлов .json/.geojson")
                        return None
                    
                    self.logger.info(f"Чтение из архива: {target_file}")
                    with z.open(target_file) as f:
                        return json.load(f)
            except zipfile.BadZipFile:
                self.logger.error(f"Битый ZIP-архив: {file_path}")
                return None

        # СЛУЧАЙ 2: Обычный файл
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Ошибка чтения файла {file_path}: {e}")
                return None

    def load(self, file_path: str) -> pd.DataFrame:
        """
        Основной метод загрузки и первичной нормализации.
        """
        if not os.path.exists(file_path):
            self.logger.error(f"Файл не найден: {file_path}")
            return pd.DataFrame()

        self.logger.info(f"Загрузка: {file_path}")

        # 1. Читаем данные (из zip или напрямую)
        data = self._read_data(file_path)
        if not data:
            return pd.DataFrame()

        # 2. Парсим структуру GeoJSON
        dtp_list = []
        try:
            if isinstance(data, dict) and 'features' in data:
                self.logger.info("Формат: GeoJSON.")
                dtp_list = data['features']
            elif isinstance(data, list):
                self.logger.info("Формат: JSON List.")
                dtp_list = data
            else:
                self.logger.warning("Нестандартный JSON, пробую обернуть в список.")
                dtp_list = [data]

            if not dtp_list:
                self.logger.warning("Список данных пуст.")
                return pd.DataFrame()

            # 3. Создаем DataFrame
            df = pd.json_normalize(dtp_list)

            # 4. Убираем префиксы properties.
            if any(col.startswith('properties.') for col in df.columns):
                df.columns = df.columns.str.replace('properties.', '', regex=False)
            
            # На всякий случай убираем и geometry., если проскочило
            if any(col.startswith('geometry.') for col in df.columns):
                df.columns = df.columns.str.replace('geometry.', '', regex=False)

            self.logger.info(f"Загружено строк: {len(df)}. Колонок: {len(df.columns)}")
            return df

        except Exception as e:
            self.logger.critical(f"Ошибка парсинга данных: {e}", exc_info=True)
            return pd.DataFrame()