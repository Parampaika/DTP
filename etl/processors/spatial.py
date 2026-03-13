import pandas as pd
from etl.processors.base import BaseDataProcessor

class SpatialProcessor(BaseDataProcessor):
    """
    Обработка географических данных.
    1. Преобразование координат в float.
    2. Создание уникального идентификатора локации (Область + Район).
    """

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        with self.activity("Обработка географических признаков"):

            # --- 1. Координаты ---
            if 'point.lat' in df.columns and 'point.long' in df.columns:
                df['lat'] = pd.to_numeric(df['point.lat'], errors='coerce')
                df['long'] = pd.to_numeric(df['point.long'], errors='coerce')
                
                df.drop(columns=['point.lat', 'point.long'], inplace=True)
                self.logger.info("   [i] Координаты обработаны.")

            # --- 2. Регионы ---
            if 'region' in df.columns and 'parent_region' in df.columns:
                df['region'] = df['region'].fillna('unknown')
                df['parent_region'] = df['parent_region'].fillna('unknown')

                df['region'] = df['region'].apply(self.normalize_text)
                df['parent_region'] = df['parent_region'].apply(self.normalize_text)

                df['region_id'] = df['parent_region'] + "_" + df['region']
                
                df.drop(columns=['parent_region', 'region'], inplace=True)
                self.logger.info("   [i] Регионы (дочерный и родительский) обработаны.")

            # --- 3. СХЕМА (Scheme) ---
            if 'scheme' in df.columns:
                # Заполняем пропуски
                df['scheme'] = df['scheme'].fillna('unknown').astype(str)
                df['scheme'] = df['scheme'].apply(lambda x: x.split('.')[0] if '.' in x else x)
                self.logger.info("   [i] Схемы обработаны (fillna + split).")

        return df