import pandas as pd
import re
import logging

# Настройка логгера по умолчанию, если не передан извне
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EconomicEnricher:
    def __init__(self, config: dict, logger=None):
        self.config = config
        self.logger = logger if logger else logging.getLogger(__name__)

        # Колонки, которые мы хотим забрать из экономики
        self.econ_features = [
            "Population",
            "GAP",
            "Avg_Salary",
            "Employment",
            "Street_Length",
            "Land_Area",
            "RTA_dead",
            "RTA_serious",
            "RTA_minor",
        ]

        # Ручные заплатки (Hardcoded patches)
        self.patches = {
            "республика_алтай": "алтайский",
            "северная_осетия": "северная осетия",
            "ленинградская_область": "санкт",  # Ленобласть -> Экономика Питера
            "санкт-петербург": "санкт",
            "кабардино": "кабардино",
            "карачаево": "карачаево",
            "москва": "москва",
            "московская_область": "московская",
        }

    def _clean_region_name(self, text: str) -> str:
        """Очищает название региона от мусора для сопоставления."""
        if pd.isna(text):
            return ""
        text = str(text).lower()

        garbage = [
            "автономный округ",
            "автономная область",
            "республика",
            "область",
            "край",
            "алания",
            "петербург",
            "мансийский",
            "г.",
            "ао",
            "эл",
            "югра",
            "без автономного округа",
            " — ",
            " - ",
            "-",
            "—",
        ]
        for word in garbage:
            text = text.replace(word, " ")

        text = re.sub(r"\(.*?\)", "", text)  # Убрать скобки
        text = text.replace(".", "").replace(",", "").replace("_", " ")
        text = re.sub(r"\s+", " ", text)  # Лишние пробелы
        return text.strip()

    def clean_econ_data(self, df_econ: pd.DataFrame) -> pd.DataFrame:
        """1. Очистка 'Н/Д', приведение к числам, генерация clean_name."""
        df = df_econ.copy()
        self.logger.info("--- [Enricher] 1. Очистка экономических данных ---")

        # 1. Чистка чисел
        for col in self.econ_features:
            if col in df.columns:
                # Если строки с запятыми -> точки
                if df[col].dtype == "object":
                    df[col] = df[col].astype(str).str.replace(",", ".")

                # В числа (ошибки -> NaN)
                df[col] = pd.to_numeric(df[col], errors="coerce")

                # Заполнение медианой
                nans = df[col].isna().sum()
                if nans > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    self.logger.info(
                        f"   Колонка '{col}': заполнено {nans} пропусков (median={median_val:.1f})"
                    )

        # 2. Генерация чистого имени
        df["clean_name"] = df["region_name"].apply(self._clean_region_name)
        unique_names = df["clean_name"].nunique()
        self.logger.info(f"   Уникальных регионов после очистки: {unique_names}")

        return df

    def aggregate_duplicates(self, df_econ: pd.DataFrame) -> pd.DataFrame:
        """2. Агрегация дубликатов (районы -> регион)."""
        self.logger.info("--- [Enricher] 2. Агрегация дубликатов ---")
        self.logger.info(f"   Размер ДО агрегации: {df_econ.shape}")

        # Группируем по (регион, год) и берем среднее
        df_agg = df_econ.groupby(["clean_name", "year"], as_index=False)[
            self.econ_features
        ].mean()

        self.logger.info(f"   Размер ПОСЛЕ агрегации: {df_agg.shape}")
        return df_agg

    def match_regions(self, df_dtp: pd.DataFrame, econ_names: list) -> pd.DataFrame:
        """3. Маппинг region_id из ДТП на clean_name из Экономики."""
        self.logger.info("--- [Enricher] 3. Сопоставление регионов (Matching) ---")

        # Сортируем имена от длинных к коротким для корректного поиска
        econ_names_sorted = sorted(econ_names, key=len, reverse=True)

        dtp_ids = df_dtp["region_id"].unique()
        mapping = {}

        matched_count = 0

        for r_id in dtp_ids:
            r_id_lower = str(r_id).lower()
            match_found = None

            # А. Проверка ручных заплаток
            for dtp_substr, econ_target in self.patches.items():
                if dtp_substr in r_id_lower:
                    match_found = econ_target
                    break

            # Б. Поиск подстроки
            if not match_found:
                # Заменяем _ на пробел для поиска "иркутская" в "иркутская область"
                r_search = r_id_lower.replace("_", " ").replace("-", " ")
                for name in econ_names_sorted:
                    if name in r_search:
                        match_found = name
                        break

            mapping[r_id] = match_found
            if match_found:
                matched_count += 1

        # Применяем маппинг
        df_dtp = df_dtp.copy()
        df_dtp["econ_join_key"] = df_dtp["region_id"].map(mapping)

        success_rate = matched_count / len(dtp_ids)
        self.logger.info(
            f"   Сматчилось уникальных регионов: {matched_count}/{len(dtp_ids)} ({success_rate:.1%})"
        )

        if success_rate < 1.0:
            unmatched = df_dtp[df_dtp["econ_join_key"].isna()]["region_id"].unique()[:5]
            self.logger.warning(f"   Топ-5 ненайденных регионов: {unmatched}")

        return df_dtp

    def merge(self, df_dtp: pd.DataFrame, df_econ: pd.DataFrame) -> pd.DataFrame:
        """Главный метод: запускает весь процесс и возвращает обогащенный DF."""
        self.logger.info("=== НАЧАЛО ОБОГАЩЕНИЯ ДАННЫХ ===")

        # 1. Готовим экономику
        df_econ_clean = self.clean_econ_data(df_econ)
        df_econ_agg = self.aggregate_duplicates(df_econ_clean)

        # 2. Матчим регионы в ДТП
        unique_econ_names = df_econ_agg["clean_name"].unique()
        df_dtp_ready = self.match_regions(df_dtp, unique_econ_names)

        # 3. Подготовка Годов
        self.logger.info("--- [Enricher] 4. Синхронизация годов ---")
        max_econ_year = int(df_econ_agg["year"].max())

        if "year" not in df_dtp_ready.columns:
            self.logger.error("В df_dtp нет колонки 'year'!")
            raise KeyError("Column 'year' missing in DTP dataset")

        df_dtp_ready["year"] = df_dtp_ready["year"].astype(int)
        df_econ_agg["year"] = df_econ_agg["year"].astype(int)

        # Логика: если год аварии > max_econ_year, используем max_econ_year
        df_dtp_ready["join_year"] = df_dtp_ready["year"].apply(
            lambda y: min(y, max_econ_year)
        )

        # 4. Финальный Мерж
        self.logger.info("--- [Enricher] 5. Merge и обработка пропусков ---")

        df_final = df_dtp_ready.merge(
            df_econ_agg,
            left_on=["econ_join_key", "join_year"],
            right_on=["clean_name", "year"],
            how="left",
            suffixes=("", "_econ_dup"),
        )

        # Мы НЕ удаляем clean_name, а переименовываем его в region_parent
        if "clean_name" in df_final.columns:
            df_final.rename(columns={"clean_name": "region_parent"}, inplace=True)
            self.logger.info(
                "   [Feature] Добавлена колонка 'region_parent' (Subject of Federation)"
            )

            # Заполняем пропуски в region_parent для несматченных (Чукотка и т.д.)
            # Берем первое слово из region_id в качестве заглушки
            mask_nan = df_final["region_parent"].isna()
            if mask_nan.sum() > 0:
                df_final.loc[mask_nan, "region_parent"] = df_final.loc[
                    mask_nan, "region_id"
                ].apply(lambda x: str(x).split("_")[0])
                self.logger.info(
                    f"   Заполнено {mask_nan.sum()} пропусков в region_parent из region_id"
                )

        # Удаляем технический мусор
        drop_cols = [
            "econ_join_key",
            "join_year",
            "year_econ_dup",
        ]  # clean_name убрали отсюда
        df_final.drop(
            columns=[c for c in drop_cols if c in df_final.columns], inplace=True
        )

        # 5. Заполнение пропусков для регионов-сирот (Чукотка и т.д.)
        for col in self.econ_features:
            if col in df_final.columns:
                nans_before = df_final[col].isna().sum()
                if nans_before > 0:
                    median_val = df_final[col].median()
                    df_final[col] = df_final[col].fillna(median_val)
                    self.logger.info(
                        f"   Финализация '{col}': заполнено {nans_before} пропусков медианой ({median_val:.2f})"
                    )

        self.logger.info(f"=== ГОТОВО. Итоговый размер: {df_final.shape} ===")
        return df_final
