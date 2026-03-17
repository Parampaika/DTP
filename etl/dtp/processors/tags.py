import pandas as pd
from etl.dtp.processors.base import BaseDataProcessor

class TagsProcessor(BaseDataProcessor):
    """
    Обработка nearby с логированием тегов категории 'other'.
    """
    TARGET_CATEGORIES = [
        "residential", "pedestrian_infrastructure", "road_junctions",
        "education", "commercial", "public_transport", "railway",
        "transport_hubs", "administrative", "healthcare", 
        "road_objects", "other"
    ]

    def _classify_tag(self, s: str) -> str:
        # 1. Жилая зона
        if any(w in s for w in ["жил", "многоквартирн", "индивидуальной_застройк", "внутридворов"]): return "residential"
        # 2. Пешеходная инфраструктура
        if any(w in s for w in ["пешеходн", "тротуар", "надземн", "подземн"]): return "pedestrian_infrastructure"
        # 3. Дорожные узлы
        if any(w in s for w in ["перекресток", "кругов", "выезд", "весов", "контроля", "равнозначн", "неравнозначн"]): return "road_junctions"
        # 4. Образование 
        if any(w in s for w in ["школа", "дошкольн", "детск", "образовательн", "лицей", "гимназия", "сад", "колледж", "университет", "институт"]):
            if "переход" not in s and "автошкол" not in s: return "education"
        # 5. Коммерция
        if any(w in s for w in ["торгов", "азс", "заправ", "магазин", "кафе", "ресторан", "питания", "тяготени", "притяжения", "рынок", "супермаркет"]): return "commercial"
        # 6. Общественный транспорт
        if any(w in s for w in ["остановка", "трамвай", "автобус", "маршрутн", "троллейбус", "метро", "автостанц", "автовокзал"]): return "public_transport"
        # 7. Ж/д
        if any(w in s for w in ["жд", "железн", "поезд"]): return "railway"
        # 8. Транспортные хабы
        if any(w in s for w in ["аэропорт", "порт", "пристан", "аэродром"]): return "transport_hubs"
        # 9. Административные
        if any(w in s for w in ["административн", "мвд", "дпс", "полиц", "кпм", "суд"]): return "administrative"
        # 10. Медицина
        if any(w in s for w in ["медицинск", "лечебн", "больниц", "поликлин", "диспансер", "аптека"]): return "healthcare"
        # 11. Дорожные объекты
        if any(w in s for w in ["мост", "эстакад", "путепровод", "тоннел"]): return "road_objects"
            
        return "other"

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        with self.activity("Обработка объектов окружения (nearby)"):

            if 'nearby' not in df.columns:
                return df

            new_features = {f"near_{cat}": [0] * len(df) for cat in self.TARGET_CATEGORIES}
            nearby_counts = []
            
            other_tags_collected = set()
            count_rows_with_other = 0

            nearby_series = df['nearby'].tolist()
            
            for i, tags_raw in enumerate(nearby_series):
                if not isinstance(tags_raw, list):
                    nearby_counts.append(0)
                    continue
                    
                count = 0
                has_other = False
                
                for tag in tags_raw:
                    if not tag: continue
                    count += 1
                    
                    clean_tag = self.normalize_text(tag)
                    category = self._classify_tag(clean_tag)
                    
                    new_features[f"near_{category}"][i] = 1

                    if category == "other":
                        has_other = True
                        other_tags_collected.add(clean_tag)
                
                nearby_counts.append(count)
                if has_other:
                    count_rows_with_other += 1

            for col_name, values in new_features.items():
                df[col_name] = values
                
            df['nearby_objects_count'] = nearby_counts
            df.drop(columns=['nearby'], inplace=True)
            
            total = len(df)
            pct = (count_rows_with_other / total * 100) if total else 0
            self.logger.info(f"   [i] Строк с тегами 'other': {pct:.2f}% ({count_rows_with_other}/{total})")
            
            if other_tags_collected:
                examples = list(other_tags_collected)[:15]
                self.logger.warning(f"Примеры тегов, попавших в 'other': {examples}")
                
        return df