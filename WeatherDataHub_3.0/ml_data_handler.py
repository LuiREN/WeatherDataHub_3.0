import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import os
from datetime import datetime
import logging
from statsmodels.tsa.seasonal import STL
from scipy import stats
from scipy.signal import find_peaks

class DataHandler:
    """
    Улучшенный класс для обработки данных погоды и подготовки их к прогнозированию.
    
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
                      'cloudiness_day_variable', 'cloudiness_day_overcast',
                      'cloudiness_evening_clear', 'cloudiness_evening_partly_cloudy',
                      'cloudiness_evening_variable', 'cloudiness_evening_overcast'],
            'wind': ['wind_speed_day', 'wind_speed_evening',
                    'wind_direction_day_n', 'wind_direction_day_ne',
                    'wind_direction_day_e', 'wind_direction_day_se',
                    'wind_direction_day_s', 'wind_direction_day_sw',
                    'wind_direction_day_w', 'wind_direction_day_nw',
                    'wind_direction_evening_n', 'wind_direction_evening_ne',
                    'wind_direction_evening_e', 'wind_direction_evening_se',
                    'wind_direction_evening_s', 'wind_direction_evening_sw',
                    'wind_direction_evening_w', 'wind_direction_evening_nw']
        }

    def setup_logger(self) -> None:
        """Настройка системы логирования."""
        self.logger = logging.getLogger('DataHandler')
        self.logger.setLevel(logging.INFO)
        
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
        Улучшенная валидация данных с проверкой качества.
        """
        try:
            # Проверка обязательных столбцов
            missing_cols = set(self.data_columns['required']) - set(df.columns)
            if missing_cols:
                self.logger.error(f"Отсутствуют обязательные столбцы: {missing_cols}")
                return False
            
            # Проверка формата даты
            try:
                pd.to_datetime(df['date'])
            except Exception as e:
                self.logger.error(f"Некорректный формат даты: {str(e)}")
                return False
            
            # Проверка на дубликаты дат
            if df['date'].duplicated().any():
                self.logger.error("Обнаружены дубликаты дат")
                return False
            
            # Проверка диапазона температур
            temp_cols = ['temperature_day', 'temperature_evening']
            for col in temp_cols:
                if col in df.columns:
                    temps = df[col].dropna()
                    if temps.empty:
                        self.logger.error(f"Нет данных температуры в столбце {col}")
                        return False
                    if not temps.between(-60, 60).all():
                        self.logger.error(f"Температуры вне допустимого диапазона в {col}")
                        return False
            
            # Проверка давления
            pressure_cols = ['pressure_day', 'pressure_evening']
            for col in pressure_cols:
                if col in df.columns:
                    pressures = df[col].dropna()
                    if not pressures.between(700, 800).all():
                        self.logger.error(f"Давление вне допустимого диапазона в {col}")
                        return False
            
            # Проверка бинарных признаков
            for col in self.data_columns['binary']:
                if col in df.columns:
                    vals = df[col].dropna()
                    if not vals.isin([0, 1]).all():
                        self.logger.error(f"Некорректные значения в бинарном столбце {col}")
                        return False
            
            # Проверка скорости ветра
            wind_speed_cols = ['wind_speed_day', 'wind_speed_evening']
            for col in wind_speed_cols:
                if col in df.columns:
                    speeds = df[col].dropna()
                    if not speeds.between(0, 100).all():
                        self.logger.error(f"Скорость ветра вне допустимого диапазона в {col}")
                        return False
            
            # Проверка направления ветра
            wind_dir_cols = [col for col in df.columns if 'wind_direction' in col]
            for col in wind_dir_cols:
                if col in df.columns:
                    dirs = df[col].dropna()
                    if not dirs.isin([0, 1]).all():
                        self.logger.error(f"Некорректные значения направления ветра в {col}")
                        return False
            
            # Проверка временного охвата
            date_range = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
            if len(date_range) != len(df):
                self.logger.warning("Обнаружены пропуски в датах")
            
            self.logger.info("Валидация данных успешно завершена")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при валидации данных: {str(e)}")
            return False

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Улучшенная подготовка данных с обработкой выбросов и сезонности.
        """
        try:
            prepared_df = df.copy()
            
            # Преобразование даты и установка индекса
            prepared_df['date'] = pd.to_datetime(prepared_df['date'])
            prepared_df.set_index('date', inplace=True)
            prepared_df.sort_index(inplace=True)
            
            # Заполнение пропущенных дат
            full_date_range = pd.date_range(
                prepared_df.index.min(),
                prepared_df.index.max(),
                freq='D'
            )
            prepared_df = prepared_df.reindex(full_date_range)
            
            # Обработка температурных данных
            temp_columns = ['temperature_day', 'temperature_evening']
            for col in temp_columns:
                if col in prepared_df.columns:
                    # Сохраняем знаки температур
                    signs = np.sign(prepared_df[col])
                    prepared_df[col] = np.abs(prepared_df[col])
                    
                    # Логарифмическое преобразование
                    prepared_df[col] = np.log1p(prepared_df[col] + 1)
                    
                    # STL декомпозиция
                    # Заполняем пропуски для возможности декомпозиции
                    temp_filled = prepared_df[col].fillna(
                        prepared_df[col].rolling(window=7, center=True, min_periods=1).mean()
                    )
                    
                    stl = STL(temp_filled, period=365, robust=True).fit()
                    
                    # Заполняем пропуски с учетом тренда и сезонности
                    trend = stl.trend
                    seasonal = stl.seasonal
                    prepared_df[col] = prepared_df[col].fillna(trend + seasonal)
                    
                    # Обработка выбросов с учетом сезонности
                    residuals = prepared_df[col] - (trend + seasonal)
                    Q1 = residuals.quantile(0.25)
                    Q3 = residuals.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    
                    # Обрезаем выбросы в остатках
                    residuals_cleaned = residuals.clip(lower=lower_bound, upper=upper_bound)
                    prepared_df[col] = trend + seasonal + residuals_cleaned
                    
                    # Обратное преобразование
                    prepared_df[col] = (np.expm1(prepared_df[col]) - 1) * signs
            
            # Обработка давления
            pressure_columns = ['pressure_day', 'pressure_evening']
            for col in pressure_columns:
                if col in prepared_df.columns:
                    # Используем сплайн-интерполяцию для давления
                    prepared_df[col] = prepared_df[col].interpolate(
                        method='cubic', 
                        limit_direction='both'
                    )
            
            # Обработка облачности
            cloud_columns = [col for col in prepared_df.columns if 'cloudiness' in col]
            for col in cloud_columns:
                # Для каждого момента времени (день/вечер) должно быть только одно состояние
                time_period = 'day' if 'day' in col else 'evening'
                related_cols = [c for c in cloud_columns if time_period in c]
                
                # Заполняем наиболее вероятным состоянием
                if prepared_df[col].isnull().any():
                    # Находим наиболее частое состояние для каждого месяца
                    prepared_df[col] = prepared_df[col].fillna(
                        prepared_df.groupby(prepared_df.index.month)[col].transform(
                            lambda x: x.mode().iloc[0] if not x.mode().empty else 0
                        )
                    )
            
            # Обработка ветра
            wind_speed_cols = ['wind_speed_day', 'wind_speed_evening']
            for col in wind_speed_cols:
                if col in prepared_df.columns:
                    # Используем сезонное заполнение для скорости ветра
                    prepared_df[col] = prepared_df[col].fillna(
                        prepared_df.groupby(prepared_df.index.month)[col].transform('mean')
                    )
            
            # Обработка направления ветра
            wind_dir_cols = [col for col in prepared_df.columns if 'wind_direction' in col]
            for col in wind_dir_cols:
                if col in prepared_df.columns:
                    # Заполняем наиболее вероятным направлением для каждого месяца
                    prepared_df[col] = prepared_df[col].fillna(
                        prepared_df.groupby(prepared_df.index.month)[col].transform(
                            lambda x: x.mode().iloc[0] if not x.mode().empty else 0
                        )
                    )
            
            self.logger.info("Подготовка данных успешно завершена")
            return prepared_df
            
        except Exception as e:
            self.logger.error(f"Ошибка при подготовке данных: {str(e)}")
            raise

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Улучшенное разделение данных с учетом сезонности.
        """
        try:
            # Проверяем, что есть хотя бы 2 года данных
            total_days = (df.index.max() - df.index.min()).days
            if total_days < 730:  # 2 года
                self.logger.warning("Рекомендуется использовать минимум 2 года данных")
            
            # Определяем точку разделения с учетом сезонности
            split_point = int(len(df) * (1 - test_size))
            
            # Убеждаемся, что разделение происходит на границе сезона
            split_date = df.index[split_point]
            next_month_start = pd.Timestamp(
                year=split_date.year + (split_date.month == 12),
                month=split_date.month % 12 + 1,
                day=1
            )
            
            # Корректируем точку разделения
            split_idx = df.index.get_indexer([next_month_start], method='nearest')[0]
            
            # Разделяем данные
            train_data = df.iloc[:split_idx].copy()
            test_data = df.iloc[split_idx:].copy()
            
            # Проверяем размер тестовой выборки
            actual_test_size = len(test_data) / len(df)
            if abs(actual_test_size - test_size) > 0.05:
                self.logger.warning(
                    f"Фактический размер тестовой выборки ({actual_test_size:.2f}) "
                    f"отличается от запрошенного ({test_size:.2f})"
                )
            
            self.logger.info(
                f"Данные разделены: {len(train_data)} наблюдений для обучения, "
                f"{len(test_data)} наблюдений для тестирования"
            )
            
            return train_data, test_data
            
        except Exception as e:
            self.logger.error(f"Ошибка при разделении данных: {str(e)}")
            raise

    def save_results(self, results: Dict, model_params: Dict) -> str:
        """
        Расширенное сохранение результатов с дополнительной информацией.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'results/sarima_results_{timestamp}.txt'
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Результаты прогнозирования погоды (SARIMA)\n")
                f.write("=" * 50 + "\n\n")
                
                # Параметры модели
                f.write("1. Параметры модели\n")
                f.write("-" * 20 + "\n")
                f.write(f"order (p,d,q): {model_params['order']}\n")
                f.write(f"seasonal_order (P,D,Q,s): {model_params['seasonal_order']}\n\n")
                
                # Метрики качества
                f.write("2. Метрики качества\n")
                f.write("-" * 20 + "\n")
                metrics = results.get('metrics', {})
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{metric}: {value:.4f}\n")
                
                # Анализ ошибок
                f.write("\n3. Анализ ошибок\n")
                f.write("-" * 20 + "\n")
                error_analysis = results.get('error_analysis', {})
                if error_analysis:
                    f.write(f"Максимальная ошибка: {error_analysis.get('max_error', 'N/A')}\n")
                    f.write(f"Минимальная ошибка: {error_analysis.get('min_error', 'N/A')}\n")
                    f.write(f"Стандартное отклонение ошибок: {error_analysis.get('std_error', 'N/A')}\n")
                    f.write(f"Средняя ошибка: {error_analysis.get('mean_error', 'N/A')}\n")
                
                # Информация о декомпозиции
                f.write("\n4. Информация о декомпозиции\n")
                f.write("-" * 20 + "\n")
                decomposition = results.get('decomposition', {})
                if decomposition:
                    f.write(f"Сила годовой сезонности: {decomposition.get('seasonal_year_strength', 'N/A'):.4f}\n")
                    f.write(f"Сила недельной сезонности: {decomposition.get('seasonal_week_strength', 'N/A'):.4f}\n")
                    f.write(f"Сила тренда: {decomposition.get('trend_strength', 'N/A'):.4f}\n")
                
                # Дополнительная информация
                f.write("\n5. Дополнительная информация\n")
                f.write("-" * 20 + "\n")
                f.write(f"Дата создания: {timestamp}\n")
                f.write(f"Количество наблюдений: {results.get('n_observations', 'N/A')}\n")
                f.write(f"AIC: {results.get('aic', 'N/A')}\n")
                f.write(f"BIC: {results.get('bic', 'N/A')}\n")
                
                # Предупреждения и рекомендации
                warnings = results.get('warnings', [])
                if warnings:
                    f.write("\n6. Предупреждения и рекомендации\n")
                    f.write("-" * 20 + "\n")
                    for warning in warnings:
                        f.write(f"- {warning}\n")
            
            self.logger.info(f"Результаты сохранены в {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении результатов: {str(e)}")
            raise