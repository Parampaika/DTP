import pandas as pd
import numpy as np
from etl.dtp.processors.base import BaseDataProcessor

class TemporalProcessor(BaseDataProcessor):
    """
    Обработка времени и освещения.
    """
    
    LIGHT_MAPPING = {
        'светлое_время_суток': 'day',
        'в_темное_время_суток_освещение_включено': 'night_lit',
        'сумерки': 'twilight',
        'в_темное_время_суток_освещение_отсутствует': 'night_dark',
        'в_темное_время_суток_освещение_не_включено': 'night_dark',
        'не_установлено': 'day'
    }

    ALLOWED_LIGHTS = {'day', 'night_lit', 'night_dark', 'twilight'}

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        with self.activity("Обработка времени и освещения"):

            # --- DATETIME ---
            if self.validate_columns(df, ['datetime']):
                if df['datetime'].isna().any():
                    df['datetime'] = df['datetime'].ffill()

                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                
                df['year'] = df['datetime'].dt.year 
                
                df['month_sin'] = np.sin(2 * np.pi * df['datetime'].dt.month / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)
                df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
                df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6]).astype(int)
                
                self.logger.info("   [i] Циклы времени созданы.")

            # --- LIGHT ---
            if 'light' in df.columns:
                df = self.safe_map_categorical(
                    df,
                    col='light',
                    mapping=self.LIGHT_MAPPING,
                    whitelist=self.ALLOWED_LIGHTS,
                    default='other'
                )
                
                df.rename(columns={'light': 'light_cat'}, inplace=True)
                
                self.logger.info("   [i] Свет обработан.")
            
            if 'datetime' in df.columns:
                df.drop(columns=['datetime'], inplace=True)
                self.logger.info("   [-] Колонка 'datetime' удалена.")

        return df