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
        self.data_columns = {
            'required': ['date', 'temperature_day', 'temperature_evening', 
                        'pressure_day', 'pressure_evening'],
            'binary': ['cloudiness_day_clear', 'cloudiness_day_partly_cloudy',
                      'cloudiness_day_variable', 'cloudiness_day_overcast'],
            'wind': ['wind_speed_day', 'wind_speed_evening']
        }

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
        """Улучшенная валидация данных."""
        try:
            # Проверка обязательных столбцов
            missing = self.data_columns['required'] - set(df.columns)
            if missing:
                self.logger.error(f"Отсутствуют столбцы: {missing}")
                return False
            
            # Проверка формата даты
            try:
                pd.to_datetime(df['date'])
            except:
                self.logger.error("Некорректный формат даты")
                return False
            
            # Проверка допустимых значений для бинарных признаков
            for col in self.data_columns['binary']:
                if col in df.columns:
                    if not df[col].isin([0, 1, np.nan]).all():
                        self.logger.error(f"Некорректные значения в столбце {col}")
                        return False
            
            # Проверка диапазонов для числовых признаков
            value_ranges = {
                'temperature_day': (-60, 60),
                'temperature_evening': (-60, 60),
                'pressure_day': (700, 800),
                'pressure_evening': (700, 800),
                'wind_speed_day': (0, 100),
                'wind_speed_evening': (0, 100)
            }
            
            for col, (min_val, max_val) in value_ranges.items():
                if col in df.columns:
                    valid_mask = df[col].between(min_val, max_val) | df[col].isna()
                    if not valid_mask.all():
                        self.logger.error(f"Значения вне допустимого диапазона в {col}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при валидации: {str(e)}")
            return False

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Улучшенная подготовка данных."""
        try:
            prepared_df = df.copy()
            
            # Преобразование даты и установка индекса
            prepared_df['date'] = pd.to_datetime(prepared_df['date'])
            prepared_df.set_index('date', inplace=True)
            prepared_df.sort_index(inplace=True)
            prepared_df = prepared_df.asfreq('D')
            
            # Обработка выбросов для температуры
            for col in ['temperature_day', 'temperature_evening']:
                Q1 = prepared_df[col].quantile(0.25)
                Q3 = prepared_df[col].quantile(0.75)
                IQR = Q3 - Q1
                prepared_df[col] = prepared_df[col].clip(
                    lower=Q1 - 2*IQR, 
                    upper=Q3 + 2*IQR
                )
            
            # Интерполяция пропущенных значений
            numeric_columns = ['temperature_day', 'temperature_evening', 
                             'pressure_day', 'pressure_evening']
            
            for col in numeric_columns:
                # Сначала пробуем сезонную интерполяцию
                prepared_df[col] = prepared_df[col].interpolate(
                    method='time', 
                    limit_direction='both', 
                    order=3
                )
                
                # Если остались пропуски, используем линейную интерполяцию
                if prepared_df[col].isnull().any():
                    prepared_df[col] = prepared_df[col].interpolate(
                        method='linear',
                        limit_direction='both'
                    )
            
            # Заполнение пропусков в бинарных признаках
            binary_columns = self.data_columns['binary']
            prepared_df[binary_columns] = prepared_df[binary_columns].fillna(0)
            
            # Обработка пропусков в скорости ветра
            wind_columns = self.data_columns['wind']
            for col in wind_columns:
                if col in prepared_df.columns:
                    prepared_df[col] = prepared_df[col].fillna(
                        prepared_df[col].rolling(window=3, min_periods=1).mean()
                    )
            
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