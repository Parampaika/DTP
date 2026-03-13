import pandas as pd
from etl.processors.base import BaseDataProcessor

class CleaningProcessor(BaseDataProcessor):
    """
    Первичная очистка: удаление мусора, дублей и явных утечек данных (Data Leakage).
    """

    FORCE_DROP = [
        'id', 
        'geometry.coordinates',
        'coordinates',
        'address'
    ]

    LEAKAGE_COLS = [
        'dead_count', 
        'injured_count',
        'participants_count'
    ]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("--- Запуск очистки данных ---")

        cols_to_drop = [c for c in self.FORCE_DROP if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            self.logger.info(f"Удален технический мусор: {cols_to_drop}")

        leakage_drop = [c for c in self.LEAKAGE_COLS if c in df.columns]
        if leakage_drop:
            df.drop(columns=leakage_drop, inplace=True)
            self.logger.info(f"Удалены поля с утечкой данных (Leakage): {leakage_drop}")

        return df
