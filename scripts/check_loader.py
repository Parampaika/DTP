import logging
import sys
import pandas as pd
from etl.processors.participants import ParticipantProcessor
from etl.processors.vehicles import VehicleProcessor
from etl.processors.conditions import ConditionsProcessor
from etl.processors.category import CategoryProcessor
from etl.processors.spatial import SpatialProcessor
from processors.nearby import TagsProcessor
from etl.processors.temporal import TemporalProcessor
from etl.processors.cleaner import CleaningProcessor
from etl.loaders.json_loader import JsonDataLoader
from etl.processors.target import TargetProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
if __name__ == "__main__":
    # file_path = 'data/raw/novgorodskaia-oblast.geojson'
    # file_path = 'data/raw/sankt-peterburg.geojson'
    file_path = 'data/test/sample_10.geojson'
    
    loader = JsonDataLoader()
    df = loader.load(file_path)

    # 1. Чистка
    cleaner = CleaningProcessor()
    df = cleaner.process(df)
    # print(f'Начальные колонки: {df.columns}')
    
    # 2. Обработка времени (datetime и light)
    temp_proc = TemporalProcessor()
    df = temp_proc.process(df)

    # 3. Обработка nearby
    tags_proc = TagsProcessor()
    df = tags_proc.process(df)

    # 4. Обработка гео
    spatial_proc = SpatialProcessor()
    df = spatial_proc.process(df)

    # 5. Обработка категории
    category_proc = CategoryProcessor()
    df = category_proc.process(df)

    # 6. Условия (погода + дорожное покрытие)
    cond_proc = ConditionsProcessor()
    df = cond_proc.process(df)

    # 7. Всё про технику
    vehicle_proc = VehicleProcessor()
    df = vehicle_proc.process(df)

    # 8. Всё про людей
    part_proc = ParticipantProcessor(config={'violations_mode': 'post_event'})
    df = part_proc.process(df)

    # . Таргет
    target_processor = TargetProcessor()
    df = target_processor.process(df)
    
    print(df.columns)
    df.to_csv('data/test/sample_10.csv')