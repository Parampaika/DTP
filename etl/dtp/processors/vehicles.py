import pandas as pd
from etl.dtp.processors.base import BaseDataProcessor

class VehicleProcessor(BaseDataProcessor):
    """
    Обработка ТЕХНИКИ (автомобилей).
    Генерирует количественные признаки (Counters) вместо бинарных флагов.
    """

    def _classify_type(self, s: str) -> str:
        if not s or not isinstance(s, str):
            return 'other'
        if any(w in s for w in ['мото', 'вело', 'мопед', 'скутер', 'квадро', 'трицикл', 'сим']): return 'moto'
        if any(w in s for w in ['автобус', 'трамвай', 'троллейбус', 'одноэтажн', 'двухэтажн', 'электробус', 'пассажирск']): return 'bus'
        if any(w in s for w in ['спецтехник', 'трактор', 'экскаватор', 'пожарн', 'медицин', 'дорожно', 'спасательн', 'оперативно', 'бульдозер', 'полици', 'коммунальн', 'погрузчик', 'автокран', 'снегоуборочн']): return 'special'
        if any(w in s for w in ['грузов', 'фургон', 'самосвал', 'тягач', 'цистерн', 'бортов', 'шасси', 'рефрижератор', 'бетоно', 'лесовоз', 'контейнеровоз']): return 'truck'
        if any(w in s for w in ['класс', 'легков', 'минивэн', 'универсал', 'спортивн', 'седан', 'хетчбэк', 'джип', 'персональн']): return 'car'
        return 'other'

    def _classify_brand(self, s: str) -> str:
        if not s or not isinstance(s, str):
            return 'other'
        if any(x in s for x in ['ваз', 'газ', 'уаз', 'lada', 'иж', 'москвич', 'заз', 'тагаз', 'tagaz', 'камаз', 'маз', 'паз', 'лиаз', 'нефаз', 'краз', 'урал', 'зил', 'кавз']): return 'ru'
        if any(x in s for x in ['mercedes', 'bmw', 'audi', 'lexus', 'volvo', 'land_rover', 'porsche', 'infiniti', 'jaguar', 'cadillac', 'mini', 'genesis', 'tesla', 'bentley', 'jeep']): return 'premium'
        if any(x in s for x in ['chery', 'geely', 'haval', 'exeed', 'omoda', 'lifan', 'great_wall', 'changan', 'faw', 'jac', 'tank', 'voyah', 'byd', 'dongfeng', 'sitrak', 'shacman', 'foton']): return 'chinese'
        if any(x in s for x in ['man', 'scania', 'daf', 'isuzu', 'iveco', 'freightliner', 'setra', 'neoplan', 'volgabus', 'howo']): return 'commercial'
        if any(x in s for x in ['yamaha', 'kawasaki', 'harley', 'ducati', 'ktm', 'triumph', 'bajaj']): return 'moto'
        if any(x in s for x in ['hyundai', 'kia', 'volkswagen', 'renault', 'toyota', 'nissan', 'ford', 'skoda', 'chevrolet', 'mitsubishi', 'opel', 'honda', 'mazda', 'suzuki', 'peugeot', 'citroen', 'subaru', 'fiat', 'ssangyong']): return 'mass'
        return 'other'

    def _classify_color(self, s: str) -> str:
        """
        Группировка цветов по видимости на дороге.
        Возвращает: 'dark', 'light', 'colored' или None (если игнорируем).
        """
        if not isinstance(s, str): return None
        s = self.normalize_text(s) 
        if not s or s in ['не_заполнено', 'none', 'иные_цвета']: return None
        if any(x in s for x in ['черн', 'серый', 'коричнев', 'темн', 'синий', 'фиолетов', 'бурый']): return 'dark'
        if any(x in s for x in ['белый', 'серебр', 'бежев', 'светл', 'металл', 'желт', 'оранж']): return 'light'
        if any(x in s for x in ['красн', 'зелен', 'голуб', 'салат', 'розов', 'бордов', 'многоцветн']): return 'colored'
        return None


    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        with self.activity("Обработка автомобилей (Счетчики, Возраст, Взаимодействие)"):
            if 'year' in df.columns:
                current_years = df['year'].fillna(2020)
            elif 'datetime' in df.columns:
                current_years = pd.to_datetime(df['datetime'], errors='coerce').dt.year.fillna(2020)
            else:
                current_years = pd.Series([2020]*len(df))

            res = {
                'vh_count': [],
                
                'vh_is_solo': [],             # 1 машина
                'vh_is_mass': [],             # > 2 машин
                'vh_heavy_light_conflict': [],# Тяжелые vs Легкие
                'vh_age_gap': [],             # Разница в возрасте машин
                
                # Возраст
                'vh_max_age': [], 'vh_mean_age': [],
                
                # Типы
                'vh_count_car': [], 'vh_count_truck': [], 'vh_count_bus': [], 
                'vh_count_moto': [], 'vh_count_special': [],
                
                # Бренды
                'vh_count_brand_ru': [], 'vh_count_brand_premium': [], 
                'vh_count_brand_chinese': [], 'vh_count_brand_mass': [], 
                'vh_count_brand_commercial': [],
                
                # Цвета
                'vh_count_color_dark': [], 'vh_count_color_light': [], 
                'vh_count_color_colored': []
            }

            vehicles_iter = df['vehicles'].fillna("").apply(lambda x: x if isinstance(x, list) else [])

            for v_list, accident_year in zip(vehicles_iter, current_years):
                
                cnt_type = {'car': 0, 'truck': 0, 'bus': 0, 'moto': 0, 'special': 0, 'other': 0}
                cnt_brand = {'ru': 0, 'premium': 0, 'chinese': 0, 'mass': 0, 'commercial': 0, 'moto': 0, 'other': 0}
                cnt_color = {'dark': 0, 'light': 0, 'colored': 0}
                
                ages = []
                
                for car in v_list:
                    t = self._classify_type(self.normalize_text(car.get('category', '')))
                    cnt_type[t] += 1
                    
                    b = self._classify_brand(self.normalize_text(car.get('brand', '')))
                    cnt_brand[b] += 1
                    
                    c = self._classify_color(self.normalize_text(car.get('color', '')))
                    if c: cnt_color[c] += 1
                    
                    year_val = car.get('year')
                    if year_val:
                        try:
                            age = float(accident_year) - float(year_val)
                            if 0 <= age <= 60: ages.append(age)
                        except: pass

                v_count = len(v_list)
                res['vh_count'].append(v_count)
                
                res['vh_count_car'].append(cnt_type['car'])
                res['vh_count_truck'].append(cnt_type['truck'])
                res['vh_count_bus'].append(cnt_type['bus'])
                res['vh_count_moto'].append(cnt_type['moto'])
                res['vh_count_special'].append(cnt_type['special'])

                res['vh_count_brand_ru'].append(cnt_brand['ru'])
                res['vh_count_brand_premium'].append(cnt_brand['premium'])
                res['vh_count_brand_chinese'].append(cnt_brand['chinese'])
                res['vh_count_brand_mass'].append(cnt_brand['mass'])
                res['vh_count_brand_commercial'].append(cnt_brand['commercial'])

                res['vh_count_color_dark'].append(cnt_color['dark'])
                res['vh_count_color_light'].append(cnt_color['light'])
                res['vh_count_color_colored'].append(cnt_color['colored'])

                if ages:
                    res['vh_max_age'].append(max(ages))
                    res['vh_mean_age'].append(round(sum(ages) / len(ages), 1))
                    res['vh_age_gap'].append(max(ages) - min(ages) if len(ages) > 1 else 0)
                else:
                    res['vh_max_age'].append(-1)
                    res['vh_mean_age'].append(-1)
                    res['vh_age_gap'].append(-1)

                res['vh_is_solo'].append(1 if v_count == 1 else 0)
                res['vh_is_mass'].append(1 if v_count >= 3 else 0)
                
                has_heavy = (cnt_type['truck'] > 0) or (cnt_type['bus'] > 0)
                has_light = (cnt_type['car'] > 0) or (cnt_type['moto'] > 0)
                res['vh_heavy_light_conflict'].append(1 if has_heavy and has_light else 0)

            for k, v in res.items():
                df[k] = v
        return df