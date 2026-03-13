import pandas as pd
from etl.processors.base import BaseDataProcessor

class ConditionsProcessor(BaseDataProcessor):
    """
    Обработка списков: Погода (weather) и Дорога (road_conditions).
    Использует универсальный безопасный метод safe_expand_list.
    """

    # Погода (ключи в snake_case)
    WEATHER_MAPPING = {
        'ясно': 'clear',
        'пасмурно': 'overcast',
        'дождь': 'rain',
        'снегопад': 'snow',
        'метель': 'snow',
        'туман': 'fog',
        'ураганный_ветер': 'wind',
        'температура_выше_30с': 'hot',
        'температура_ниже_30с': 'cold',
        'низкая_температура': 'cold'
    }

    # Дорога (ключи - корни слов для поиска подстрок)
    ROAD_MAPPING = {
        # --- Состояние покрытия ---
        'сухое': 'dry',
        'мокрое': 'wet',
        'залитое': 'wet',
        'гололедица': 'ice',
        'обледеневшее': 'ice',
        'противогололед': 'ice', 
        'заснеженное': 'snow',
        'накатом': 'snow',
        'пыльное': 'dirty',
        'загрязненное': 'dirty',
        'свежеуложен': 'roadworks',

        # --- Инфраструктура ---
        'разметк': 'bad_markings',
        'знаков': 'bad_signs',
        'реклам': 'bad_signs',
        'светофор': 'bad_lights',
        'освещени': 'no_light',
        'огражден': 'bad_fences',
        'тротуаров': 'bad_infrastructure',
        'пешеход': 'bad_infrastructure',
        'остановочн': 'bad_infrastructure',
        'железнодорож': 'bad_infrastructure',

        # --- Дефекты полотна ---
        'покрыти': 'bad_surface',
        'обочин': 'bad_surface',
        'люков': 'bad_surface',
        'колейность': 'bad_surface',
        'неровност': 'bad_surface',
        'ямочность': 'bad_surface',
        
        # --- Обслуживание ---
        'зимнего': 'bad_maintenance',
        
        # --- Препятствия и видимость ---
        'сужение': 'obstacles',
        'видимост': 'obstacles',
        'машин': 'obstacles',
        
        # --- Работы ---
        'работ': 'roadworks',
        
        # --- Игнорируем или в Other (чтобы не было Warning) ---
        'иные': 'other',
        'не_установлено': 'other' 
    }

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        with self.activity("Обработка списков погоды и дорожных условий"):
            df = self.safe_expand_list(df, 'weather', self.WEATHER_MAPPING, 'weather')
            self.logger.info("   [i] Погода обработана.")
            df = self.safe_expand_list(df, 'road_conditions', self.ROAD_MAPPING, 'road')
            self.logger.info("   [i] Дорожные условия обработаны.")
        
        return df