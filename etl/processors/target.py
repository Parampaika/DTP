import pandas as pd
import numpy as np
from etl.processors.base import BaseDataProcessor

class TargetProcessor(BaseDataProcessor):
    """
    Обработчик целевой переменной (Target).
    """

    SEVERITY_MAPPING = {
        'легкий': 0,
        'тяжелый': 1,
        'с_погибшими': 2
    }

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_columns(df, ['severity']):
            return df

        with self.activity("Кодирование целевой переменной severity"):

            df['severity'] = df['severity'].apply(self.normalize_text)
            df['target'] = df['severity'].map(self.SEVERITY_MAPPING)

            nan_count = df['target'].isna().sum()
            if nan_count > 0:
                unknown_labels = df[df['target'].isna()]['severity'].unique()
                self.logger.warning(
                    f"Внимание: {nan_count} строк удалено (неизвестный статус тяжести). "
                    f"Встреченные неизвестные метки: {unknown_labels}"
                )
                df.dropna(subset=['target'], inplace=True)

            df['target'] = df['target'].astype(int)
            
            df.drop(columns=['severity'], inplace=True) 

            dist = df['target'].value_counts().sort_index().to_dict()
            self.logger.info(f"   [i] Баланс классов: {dist}")
        return df