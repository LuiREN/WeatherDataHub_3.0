import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import os
from datetime import datetime
import logging

class DataHandler:
    """
    Класс для обработки данных погоды и подготовки их к прогнозированию.
    
    Attributes:
        logger: Логгер для записи операций
    """
    
    def __init__(self):
        """Инициализация обработчика данных."""
        self.setup_logger()
        self.setup_directories()

    def setup_logger(self) -> None:
        """Настройка системы логирования."""
        self.logger = logging.getLogger('DataHandler')
        self.logger.setLevel(logging.INFO)
        
        # Создаем директорию для логов если её нет
        os.makedirs('logs', exist_ok=True)
        
        handler = logging.FileHandler(
            f'logs/data_handler_{datetime.now().strftime("%Y%m%d")}.log'
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def setup_directories(self) -> None:
        """Создание необходимых директорий."""
        directories = ['models', 'results', 'temp_data']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Проверка корректности данных погоды.
        
        Args:
            df: DataFrame для проверки
            
        Returns:
            bool: True если данные корректны
        """
        try:
            # Проверяем наличие необходимых столбцов
            required_columns = {
                'date', 'temperature_day', 'temperature_evening',
                'pressure_day', 'pressure_evening'
            }
            
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                self.logger.error(f"Отсутствуют столбцы: {missing}")
                return False
            
            # Проверяем формат даты
            try:
                pd.to_datetime(df['date'])
            except:
                self.logger.error("Некорректный формат даты")
                return False
            
            # Проверяем типы данных температуры
            numeric_columns = ['temperature_day', 'temperature_evening', 
                             'pressure_day', 'pressure_evening']
            for col in numeric_columns:
                if not pd.to_numeric(df[col], errors='coerce').notnull().all():
                    self.logger.error(f"Некорректные значения в столбце {col}")
                    return False
            
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
            
            # Создаем копию данных
            prepared_df = df.copy()
            
            # Преобразование даты
            prepared_df['date'] = pd.to_datetime(prepared_df['date'])
            prepared_df.set_index('date', inplace=True)
            
            # Сортировка по дате
            prepared_df.sort_index(inplace=True)
            
            # Заполнение пропущенных значений
            numeric_columns = ['temperature_day', 'temperature_evening', 
                             'pressure_day', 'pressure_evening']
            
            for col in numeric_columns:
                if prepared_df[col].isnull().any():
                    # Используем линейную интерполяцию для временного ряда
                    prepared_df[col] = prepared_df[col].interpolate(method='time')
            
            # Убедимся, что нет пропущенных значений в начале и конце
            prepared_df = prepared_df.fillna(method='bfill').fillna(method='ffill')
            
            # Проверяем частоту данных
            if not prepared_df.index.freq:
                prepared_df = prepared_df.asfreq('D')
            
            self.logger.info("Данные успешно подготовлены")
            return prepared_df
            
        except Exception as e:
            self.logger.error(f"Ошибка при подготовке данных: {str(e)}")
            raise

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Разделение данных на обучающую и тестовую выборки.
        
        Args:
            df: DataFrame для разделения
            test_size: Размер тестовой выборки
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Кортеж (train_data, test_data)
        """
        try:
            # Вычисляем индекс разделения
            split_idx = int(len(df) * (1 - test_size))
            
            # Разделяем данные
            train_data = df.iloc[:split_idx].copy()
            test_data = df.iloc[split_idx:].copy()
            
            self.logger.info(
                f"Данные разделены: {len(train_data)} дней для обучения, "
                f"{len(test_data)} дней для тестирования"
            )
            
            return train_data, test_data
            
        except Exception as e:
            self.logger.error(f"Ошибка при разделении данных: {str(e)}")
            raise

    def save_results(self, results: Dict, model_params: Dict) -> str:
        """
        Сохранение результатов анализа.
        
        Args:
            results: Метрики и результаты
            model_params: Параметры модели
            
        Returns:
            str: Путь к файлу с результатами
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'results/sarima_results_{timestamp}.txt'
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Результаты прогнозирования погоды (SARIMA)\n")
                f.write("=" * 50 + "\n\n")
                
                # Параметры модели
                f.write("Параметры модели:\n")
                f.write("-" * 20 + "\n")
                f.write(f"order (p,d,q): {model_params['order']}\n")
                f.write(f"seasonal_order (P,D,Q,s): {model_params['seasonal_order']}\n\n")
                
                # Метрики качества
                f.write("Метрики качества:\n")
                f.write("-" * 20 + "\n")
                for metric, value in results.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{metric}: {value:.4f}\n")
                
                f.write(f"\nДата создания: {timestamp}\n")
            
            self.logger.info(f"Результаты сохранены в {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении результатов: {str(e)}")
            raise