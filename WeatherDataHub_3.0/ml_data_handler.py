import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
import pickle
import json
import os
from datetime import datetime
import logging

class DataHandler:
    """
    Класс для обработки и управления данными для машинного обучения.
    
    Attributes:
        logger: Логгер для записи операций с данными
    """
    
    def __init__(self):
        """Инициализация обработчика данных."""
        self.setup_logger()
        self.setup_directories()

    def setup_logger(self) -> None:
        """Настройка системы логирования."""
        self.logger = logging.getLogger('DataHandler')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('data_handler.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def setup_directories(self) -> None:
        """Создание необходимых директорий для хранения данных."""
        directories = ['models', 'results', 'data']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Проверка корректности данных.
        
        Args:
            df: DataFrame для проверки
            
        Returns:
            bool: True если данные корректны, False иначе
        """
        try:
            required_columns = {'date', 'temperature_day'}
            
            # Проверка наличия необходимых столбцов
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                self.logger.error(f"Отсутствуют столбцы: {missing}")
                return False
            
            # Проверка типов данных
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                try:
                    pd.to_datetime(df['date'])
                except:
                    self.logger.error("Некорректный формат даты")
                    return False
            
            # Проверка на пропущенные значения
            if df['temperature_day'].isnull().any():
                self.logger.warning("Обнаружены пропущенные значения температуры")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при валидации данных: {str(e)}")
            return False

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка данных для анализа.
        
        Args:
            df: Исходный DataFrame
            
        Returns:
            pd.DataFrame: Подготовленный DataFrame
        """
        try:
            self.logger.info("Начало подготовки данных")
            prepared_df = df.copy()
            
            # Преобразование даты
            prepared_df['date'] = pd.to_datetime(prepared_df['date'])
            prepared_df.set_index('date', inplace=True)
            
            # Сортировка по дате
            prepared_df.sort_index(inplace=True)
            
            # Обработка пропущенных значений
            if prepared_df['temperature_day'].isnull().any():
                prepared_df['temperature_day'].fillna(
                    prepared_df['temperature_day'].mean(),
                    inplace=True
                )
            
            self.logger.info("Данные успешно подготовлены")
            return prepared_df
            
        except Exception as e:
            self.logger.error(f"Ошибка при подготовке данных: {str(e)}")
            raise

    def split_data(self, df: pd.DataFrame, 
                   test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Разделение данных на обучающую и тестовую выборки.
        
        Args:
            df: Исходный DataFrame
            test_size: Размер тестовой выборки (доля от общего размера)
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Кортеж (train_data, test_data)
        """
        try:
            split_idx = int(len(df) * (1 - test_size))
            train_data = df.iloc[:split_idx].copy()
            test_data = df.iloc[split_idx:].copy()
            
            self.logger.info(
                f"Данные разделены: {len(train_data)} записей для обучения, "
                f"{len(test_data)} записей для тестирования"
            )
            
            return train_data, test_data
            
        except Exception as e:
            self.logger.error(f"Ошибка при разделении данных: {str(e)}")
            raise

    def save_model(self, model: any, filename: str, metadata: Optional[Dict] = None) -> bool:
        """
        Сохранение модели и её метаданных.
        
        Args:
            model: Модель для сохранения
            filename: Имя файла
            metadata: Дополнительные метаданные модели
            
        Returns:
            bool: True если сохранение успешно, False иначе
        """
        try:
            # Создаем полный путь
            model_path = os.path.join('models', f"{filename}.pkl")
            meta_path = os.path.join('models', f"{filename}_metadata.json")
            
            # Сохраняем модель
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Сохраняем метаданные
            if metadata:
                metadata['saved_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
            
            self.logger.info(f"Модель сохранена: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении модели: {str(e)}")
            return False

    def load_model(self, filename: str) -> Tuple[Optional[any], Optional[Dict]]:
        """
        Загрузка модели и её метаданных.
        
        Args:
            filename: Имя файла
            
        Returns:
            Tuple[Optional[any], Optional[Dict]]: Кортеж (модель, метаданные)
        """
        try:
            model_path = os.path.join('models', f"{filename}.pkl")
            meta_path = os.path.join('models', f"{filename}_metadata.json")
            
            # Загружаем модель
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Загружаем метаданные
            metadata = None
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
            
            self.logger.info(f"Модель загружена: {model_path}")
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {str(e)}")
            return None, None

    def save_results(self, results: Dict, filename: str) -> bool:
        """
        Сохранение результатов анализа.
        
        Args:
            results: Словарь с результатами
            filename: Имя файла
            
        Returns:
            bool: True если сохранение успешно, False иначе
        """
        try:
            # Добавляем временную метку
            results['saved_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Создаем полный путь
            result_path = os.path.join('results', f"{filename}.json")
            
            # Сохраняем результаты
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Результаты сохранены: {result_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении результатов: {str(e)}")
            return False
