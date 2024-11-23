from statsmodels.tsa.seasonal import STL
from scipy import stats
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import logging
import os

class WeatherModel:
    """
    Класс для прогнозирования погоды с использованием SARIMA.
    
    Attributes:
        model: Обученная модель SARIMA
        logger: Логгер для записи операций
    """
    
    def __init__(self):
        """Инициализация модели."""
        self.model = None
        self.setup_logger()
        
    def setup_logger(self) -> None:
        """Настройка системы логирования."""
        self.logger = logging.getLogger('WeatherModel')
        self.logger.setLevel(logging.INFO)
        
        # Создаем директорию для логов если её нет
        os.makedirs('logs', exist_ok=True)
        
        handler = logging.FileHandler(
            f'logs/weather_model_{datetime.now().strftime("%Y%m%d")}.log'
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def check_stationarity(self, data: pd.Series) -> Dict:
        """Улучшенная проверка стационарности"""
        try:
            self.logger.info("Начало проверки стационарности")
            
            # Добавляем дополнительные проверки данных
            if len(data) < 2:
                raise ValueError("Недостаточно данных для анализа стационарности")
            
            if data.isnull().any():
                data = data.interpolate(method='time')
            
            # Проводим тест Дики-Фуллера с оптимизированными параметрами
            result = adfuller(data, regression='ct', autolag='AIC')
            
            # Расширенный анализ результатов
            stationarity_results = {
                'test_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05,
                'lags_used': result[2],
                'nobs': result[3],
                'test_used': 'Augmented Dickey-Fuller'
            }
            
            # Добавляем анализ тренда
            trend = np.polyfit(range(len(data)), data, 1)[0]
            stationarity_results['trend_coefficient'] = trend
            
            self.logger.info(f"Результаты проверки стационарности:\n"
                           f"Тест-статистика: {result[0]:.4f}\n"
                           f"p-значение: {result[1]:.4f}\n"
                           f"Тренд: {trend:.4f}\n"
                           f"Ряд {'стационарен' if result[1] < 0.05 else 'не стационарен'}")
            
            return stationarity_results
            
        except Exception as e:
            self.logger.error(f"Ошибка при проверке стационарности: {str(e)}")
            return {'error': str(e)}

    def analyze_autocorrelation(self, data: pd.Series, nlags: int = 40) -> Dict:
        """Улучшенный анализ автокорреляции"""
        try:
            self.logger.info(f"Начало анализа автокорреляции (лаги: {nlags})")
            
            # Предварительная обработка
            if data.isnull().any():
                data = data.interpolate(method='time')
            
            # Вычисляем автокорреляцию с confidence intervals
            acf_values = acf(data, nlags=nlags, fft=True)
            confidence_interval = 1.96/np.sqrt(len(data))
            
            # Находим значимые лаги
            significant_lags = []
            for i, value in enumerate(acf_values):
                if abs(value) > confidence_interval:
                    significant_lags.append((i, value))
            
            # Определяем периодичность
            potential_periods = []
            for i in range(1, len(acf_values)-1):
                if acf_values[i-1] < acf_values[i] > acf_values[i+1]:
                    potential_periods.append(i)
            
            results = {
                'acf_values': acf_values.tolist(),
                'significant_lags': significant_lags,
                'confidence_interval': confidence_interval,
                'potential_periods': potential_periods,
                'strongest_correlation': max(abs(acf_values[1:])),  # Исключаем лаг 0
                'number_significant': len(significant_lags)
            }
            
            self.logger.info(f"Найдено {len(significant_lags)} значимых лагов\n"
                           f"Потенциальные периоды: {potential_periods}\n"
                           f"Максимальная корреляция: {results['strongest_correlation']:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе автокорреляции: {str(e)}")
            return {'error': str(e)}

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Предобработка данных с учетом сезонности и стабилизации дисперсии.
    
        Args:
            df: DataFrame с исходными данными
        
        Returns:
            DataFrame: Обработанные данные
        """
        try:
            self.logger.info("Начало предобработки данных")
            df_copy = df.copy()
        
            if 'date' in df_copy.columns:
                df_copy['date'] = pd.to_datetime(df_copy['date'])
                df_copy.set_index('date', inplace=True)
        
            df_copy.sort_index(inplace=True)
            df_copy = df_copy.asfreq('D')
        
            # Работаем с температурными данными
            temp_columns = ['temperature_day', 'temperature_evening']
            for col in temp_columns:
                if col in df_copy.columns:
                    # Сохраняем знаки температур
                    signs = np.sign(df_copy[col])
                
                    # Преобразуем в положительные значения для логарифмирования
                    df_copy[col] = np.abs(df_copy[col]) + 1  # +1 для избежания log(0)
                
                    # Логарифмируем для стабилизации дисперсии
                    df_copy[col] = np.log1p(df_copy[col])
                
                    # STL декомпозиция для определения тренда и сезонности
                    from statsmodels.tsa.seasonal import STL
                
                    # Заполняем пропуски для возможности декомпозиции
                    temp_filled = df_copy[col].fillna(method='ffill').fillna(method='bfill')
                
                    # Выполняем декомпозицию с годовой и недельной сезонностью
                    stl_year = STL(temp_filled, period=365, robust=True).fit()
                    stl_week = STL(stl_year.resid, period=7, robust=True).fit()
                
                    # Получаем тренд и сезонные компоненты
                    trend = stl_year.trend
                    seasonal_year = stl_year.seasonal
                    seasonal_week = stl_week.seasonal
                
                    # Заполняем пропуски с учетом тренда и сезонности
                    df_copy[col] = df_copy[col].fillna(trend + seasonal_year + seasonal_week)
                
                    # Обрабатываем выбросы
                    Q1 = df_copy[col].quantile(0.25)
                    Q3 = df_copy[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR  # Используем 3*IQR вместо 1.5*IQR для большей гибкости
                    upper_bound = Q3 + 3 * IQR
                
                    # Обрезаем выбросы с учетом сезонности
                    df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
                
                    # Возвращаем исходный масштаб
                    df_copy[col] = (np.expm1(df_copy[col]) - 1) * signs
        
            # Обработка давления
            pressure_columns = ['pressure_day', 'pressure_evening']
            for col in pressure_columns:
                if col in df_copy.columns:
                    # Заполняем пропуски с учетом тренда
                    df_copy[col] = df_copy[col].interpolate(method='time')
                    df_copy[col] = df_copy[col].fillna(method='ffill').fillna(method='bfill')
        
            # Обработка облачности (бинарные признаки)
            cloud_columns = [col for col in df_copy.columns if 'cloudiness' in col]
            for col in cloud_columns:
                df_copy[col] = df_copy[col].fillna(0)
        
            # Обработка ветра
            wind_columns = [col for col in df_copy.columns if 'wind' in col]
            for col in wind_columns:
                if 'speed' in col:
                    # Для скорости ветра используем скользящее среднее
                    df_copy[col] = df_copy[col].fillna(
                        df_copy[col].rolling(window=7, min_periods=1).mean()
                    )
                else:
                    # Для направления ветра используем моду
                    df_copy[col] = df_copy[col].fillna(
                        df_copy[col].mode()[0] if not df_copy[col].mode().empty else 0
                    )
        
            self.logger.info("Предобработка данных завершена успешно")
            return df_copy
        
        except Exception as e:
            self.logger.error(f"Ошибка при предобработке данных: {str(e)}")
            raise

    def analyze_time_series(self, data: pd.Series) -> Dict:
        """
        Расширенный анализ временного ряда с определением оптимальных параметров SARIMA.
    
        Args:
            data: Временной ряд для анализа
        
        Returns:
            Dict: Результаты анализа
        """
        try:
            self.logger.info("Начало расширенного анализа временного ряда")
        
            results = {}
        
            # Базовая статистика
            data_cleaned = data.dropna()
            results['statistics'] = {
                'n_observations': len(data_cleaned),
                'mean': data_cleaned.mean(),
                'std': data_cleaned.std(),
                'min': data_cleaned.min(),
                'max': data_cleaned.max(),
                'skewness': stats.skew(data_cleaned),
                'kurtosis': stats.kurtosis(data_cleaned)
            }
        
            # Проверка стационарности
            stationarity = self.check_stationarity(data_cleaned)
            results['stationarity'] = stationarity
        
            # Расширенный анализ автокорреляции
            from statsmodels.tsa.stattools import acf, pacf
        
            # Вычисляем ACF и PACF для разных лагов
            max_lags = min(len(data_cleaned) - 1, 365)  # максимум год или длина ряда
            acf_values = acf(data_cleaned, nlags=max_lags, fft=True)
            pacf_values = pacf(data_cleaned, nlags=max_lags)
        
            # Находим значимые лаги
            confidence_interval = 1.96/np.sqrt(len(data_cleaned))
            significant_lags_acf = [
                (lag, val) for lag, val in enumerate(acf_values) 
                if abs(val) > confidence_interval and lag > 0
            ]
            significant_lags_pacf = [
                (lag, val) for lag, val in enumerate(pacf_values) 
                if abs(val) > confidence_interval and lag > 0
            ]
        
            # Определяем сезонные периоды
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(acf_values, height=confidence_interval)
            seasonal_periods = [
                peak for peak in peaks 
                if peak in [7, 14, 30, 90, 365, 180]  # проверяем известные периоды
            ]
        
            results['autocorrelation'] = {
                'acf_values': acf_values.tolist(),
                'pacf_values': pacf_values.tolist(),
                'significant_lags_acf': significant_lags_acf,
                'significant_lags_pacf': significant_lags_pacf,
                'seasonal_periods': seasonal_periods,
                'confidence_interval': confidence_interval
            }
        
            # Определяем оптимальные параметры SARIMA
            # p - на основе значимых лагов PACF
            p = min(len([lag for lag, val in significant_lags_pacf if lag <= 5]), 3)
        
            # q - на основе значимых лагов ACF
            q = min(len([lag for lag, val in significant_lags_acf if lag <= 5]), 3)
        
            # d - на основе теста стационарности
            d = 0 if stationarity['is_stationary'] else 1
        
            # Сезонные параметры
            if seasonal_periods:
                primary_season = min(seasonal_periods)  # выбираем минимальный период
                P = 1
                D = 1
                Q = 1
            else:
                primary_season = 7  # недельная сезонность по умолчанию
                P = 0
                D = 0
                Q = 0
        
            results['suggested_parameters'] = {
                'order': (p, d, q),
                'seasonal_order': (P, D, Q, primary_season),
                'alternative_seasons': seasonal_periods
            }
        
            # Декомпозиция временного ряда
            for period in [primary_season] + [s for s in seasonal_periods if s != primary_season]:
                try:
                    decomposition = STL(
                        data_cleaned,
                        period=period,
                        robust=True
                    ).fit()
                
                    strength_seasonal = 1 - np.var(decomposition.resid)/np.var(decomposition.seasonal + decomposition.resid)
                    strength_trend = 1 - np.var(decomposition.resid)/np.var(decomposition.trend + decomposition.resid)
                
                    results[f'decomposition_period_{period}'] = {
                        'trend_strength': strength_trend,
                        'seasonal_strength': strength_seasonal,
                        'residual_std': np.std(decomposition.resid)
                    }
                except Exception as e:
                    self.logger.warning(f"Ошибка при декомпозиции для периода {period}: {str(e)}")
        
            # Сохранение отчета
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'results/time_series_analysis_{timestamp}.txt'
        
            os.makedirs('results', exist_ok=True)
        
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Расширенный анализ временного ряда температуры\n")
                f.write("=" * 50 + "\n\n")
            
                # Записываем результаты анализа
                self._write_analysis_report(f, results)
        
            results['report_file'] = filename
            self.logger.info(f"Анализ временного ряда завершен, результаты сохранены в {filename}")
        
            return results
        
        except Exception as e:
            self.logger.error(f"Ошибка при анализе временного ряда: {str(e)}")
            return {
                'error': str(e),
                'stationarity': {'error': 'Анализ не выполнен'},
                'autocorrelation': {'error': 'Анализ не выполнен'}
            }

        
    def train(self, train_data: pd.Series, order: Tuple[int, int, int], 
          seasonal_order: Tuple[int, int, int, int]) -> Dict:
        """
        Обучение модели SARIMA с множественной сезонностью и робастной оптимизацией.
    
        Args:
            train_data: Временной ряд температуры для обучения
            order: Параметры (p,d,q)
            seasonal_order: Сезонные параметры (P,D,Q,s)
    
        Returns:
            Dict: Результаты обучения
        """
        try:
            self.logger.info(f"Начало обучения модели\n"
                            f"Параметры: order={order}, seasonal_order={seasonal_order}")
    
            if train_data is None or len(train_data) == 0:
                raise ValueError("Пустые данные для обучения")
        
            # Предварительная обработка данных
            train_clean = train_data.copy()
            if train_clean.isnull().any():
                self.logger.info("Обнаружены пропущенные значения, выполняется заполнение...")
                train_clean = train_clean.interpolate(method='time')
                train_clean = train_clean.ffill().bfill()
        
            # Устанавливаем частоту
            if not train_clean.index.freq:
                train_clean = train_clean.asfreq('D')
        
            self.logger.info(f"Размер данных: {len(train_clean)}")
        
            # Сохраняем знаки температур
            signs = np.sign(train_clean)
        
            # Преобразуем в положительные значения и применяем логарифмирование
            train_transformed = np.log1p(np.abs(train_clean) + 1)
        
            # Выполняем декомпозицию с множественной сезонностью
            self.logger.info("Выполняется декомпозиция с множественной сезонностью...")
        
            # Годовая сезонность
            stl_year = STL(
                train_transformed,
                period=365,
                robust=True
            ).fit()
        
            # Недельная сезонность на остатках
            stl_week = STL(
                stl_year.resid,
                period=7,
                robust=True
            ).fit()
        
            # Извлекаем компоненты
            trend = stl_year.trend
            seasonal_year = stl_year.seasonal
            seasonal_week = stl_week.seasonal
            residuals = stl_week.resid
        
            # Нормализация остатков
            residual_mean = residuals.mean()
            residual_std = residuals.std()
            residuals_normalized = (residuals - residual_mean) / residual_std
        
            self.logger.info("Создание и обучение модели SARIMA...")
        
            # Создание модели на нормализованных остатках
            model = SARIMAX(
                residuals_normalized,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                initialization='approximate_diffuse'
            )
        
            # Обучение модели с робастной оптимизацией
            try:
                self.model = model.fit(
                    disp=False,
                    method='powell',  # Более надежный метод оптимизации
                    maxiter=1000,
                    cov_type='robust',
                    optim_score='harvey',
                    optim_complex_step=True,
                    optim_hessian='cs'
                )
            except Exception as e:
                self.logger.warning(f"Первая попытка обучения не удалась: {str(e)}")
                # Вторая попытка с другими параметрами
                self.model = model.fit(
                    disp=False,
                    method='lbfgs',
                    maxiter=2000,
                    cov_type='robust'
                )
        
            # Получение прогноза для остатков
            predicted_residuals = self.model.fittedvalues
        
            # Обратное преобразование остатков
            predicted_residuals = (predicted_residuals * residual_std) + residual_mean
        
            # Восстановление прогноза
            predicted_values = pd.Series(
                predicted_residuals + seasonal_year + seasonal_week + trend,
                index=train_clean.index
            )
        
            # Обратное преобразование из лог-шкалы
            predicted_values = (np.expm1(predicted_values) - 1) * signs
        
            # Синхронизация индексов
            valid_idx = train_clean.index.intersection(predicted_values.index)
            actual_values = train_clean[valid_idx]
            predicted_values = predicted_values[valid_idx]
        
            # Проверка на NaN
            if predicted_values.isnull().any() or actual_values.isnull().any():
                self.logger.warning("Обнаружены NaN в результатах, выполняется финальная очистка...")
                mask = ~(predicted_values.isnull() | actual_values.isnull())
                actual_values = actual_values[mask]
                predicted_values = predicted_values[mask]
        
            # Расчет метрик
            errors = actual_values - predicted_values
            abs_errors = np.abs(errors)
        
            # Безопасный расчет MAPE
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs(errors / actual_values)) * 100
            
            metrics = {
                'mse': mean_squared_error(actual_values, predicted_values),
                'rmse': np.sqrt(mean_squared_error(actual_values, predicted_values)),
                'mae': mean_absolute_error(actual_values, predicted_values),
                'mape': mape,
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'max_error': np.max(abs_errors),
                'min_error': np.min(abs_errors)
            }
        
            # Информация о декомпозиции
            decomposition_info = {
                'seasonal_year_strength': 1 - np.var(residuals)/np.var(seasonal_year + residuals),
                'seasonal_week_strength': 1 - np.var(stl_week.resid)/np.var(seasonal_week + stl_week.resid),
                'trend_strength': 1 - np.var(residuals)/np.var(trend + residuals),
                'residual_mean': float(residual_mean),
                'residual_std': float(residual_std)
            }
        
            self.logger.info(
                f"Результаты обучения:\n"
                f"MSE: {metrics['mse']:.4f}\n"
                f"RMSE: {metrics['rmse']:.4f}\n"
                f"MAE: {metrics['mae']:.4f}\n"
                f"MAPE: {metrics['mape']:.2f}%"
            )
        
            return {
                'status': 'success',
                'metrics': metrics,
                'parameters': {
                    'order': order,
                    'seasonal_order': seasonal_order
                },
                'decomposition': decomposition_info,
                'model_info': {
                    'nobs': len(actual_values),
                    'aic': self.model.aic,
                    'bic': self.model.bic
                }
            }
    
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def tune_parameters(self) -> Dict:
        """
        Подбор оптимальных параметров модели SARIMA для погодных данных.
    
        Returns:
            Dict: Лучшие найденные параметры и результаты
        """
        try:
            self.logger.info("Начало подбора параметров...")
        
            # Параметры оптимизированы для температурных данных
            parameter_combinations = [
                # (p, d, q, P, D, Q, s)
                (1, 1, 1, 1, 1, 1, 7),    # базовая недельная модель
                (2, 1, 2, 0, 1, 1, 7),    # более сложная недельная модель
                (2, 0, 2, 1, 1, 1, 7),    # недельная без разности
                (1, 1, 1, 1, 1, 1, 30),   # месячная сезонность
                (2, 1, 2, 1, 1, 1, 30),   # сложная месячная модель
                (2, 1, 2, 1, 1, 1, 90),   # квартальная сезонность
                (1, 1, 2, 1, 1, 1, 7),    # асимметричная недельная модель
                (3, 1, 1, 1, 1, 1, 7)     # с увеличенным AR
            ]
        
            results = []
            best_result = None
            best_rmse = float('inf')
        
            total_combinations = len(parameter_combinations)
            self.logger.info(f"Всего комбинаций для проверки: {total_combinations}")
        
            for i, params in enumerate(parameter_combinations, 1):
                try:
                    self.logger.info(f"Проверка комбинации {i}/{total_combinations}: {params}")
                    order = params[:3]
                    seasonal_order = params[3:]
                
                    # Пробуем обучить модель с текущими параметрами
                    result = self.train(
                        self.train_data['temperature_day'],
                        order,
                        seasonal_order
                    )
                
                    if result['status'] == 'success':
                        results.append(result)
                        current_rmse = result['metrics']['rmse']
                    
                        if current_rmse < best_rmse:
                            best_rmse = current_rmse
                            best_result = result.copy()
                            best_result['parameters']['combination_index'] = i
                        
                            self.logger.info(
                                f"Найдена лучшая модель:\n"
                                f"RMSE: {best_rmse:.4f}\n"
                                f"Параметры: order={order}, seasonal_order={seasonal_order}"
                            )
                        
                except Exception as e:
                    self.logger.warning(
                        f"Ошибка для параметров {params}: {str(e)}\n"
                        "Продолжаем с следующей комбинацией..."
                    )
                    continue
        
            if not best_result:
                raise ValueError("Не удалось найти работающую комбинацию параметров")
            
            # Сохраняем отчет о подборе параметров
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f'results/parameter_tuning_{timestamp}.txt'
        
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("Результаты подбора параметров SARIMA\n")
                f.write("=" * 50 + "\n\n")
            
                # Лучшие параметры
                f.write("Лучшие параметры:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Order (p,d,q): {best_result['parameters']['order']}\n")
                f.write(f"Seasonal Order (P,D,Q,s): {best_result['parameters']['seasonal_order']}\n")
                f.write(f"RMSE: {best_result['metrics']['rmse']:.4f}\n")
                f.write(f"MSE: {best_result['metrics']['mse']:.4f}\n")
                f.write(f"MAE: {best_result['metrics']['mae']:.4f}\n")
                f.write(f"MAPE: {best_result['metrics']['mape']:.2f}%\n\n")
            
                # Все результаты
                f.write("Все проверенные комбинации:\n")
                f.write("-" * 20 + "\n")
                for result in results:
                    f.write(f"\nOrder: {result['parameters']['order']}\n")
                    f.write(f"Seasonal Order: {result['parameters']['seasonal_order']}\n")
                    f.write(f"RMSE: {result['metrics']['rmse']:.4f}\n")
                    f.write(f"MAPE: {result['metrics']['mape']:.2f}%\n")
                    if 'decomposition' in result:
                        f.write(f"Сила сезонности: {result['decomposition']['seasonal_strength']:.4f}\n")
                    f.write("-" * 20 + "\n")
            
                # Дополнительная информация
                f.write(f"\nВсего проверено комбинаций: {len(results)}\n")
                f.write(f"Дата создания отчета: {timestamp}")
        
            self.logger.info(f"Результаты подбора параметров сохранены в {results_file}")
            return best_result
        
        except Exception as e:
            self.logger.error(f"Ошибка при подборе параметров: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, steps: int) -> Optional[np.ndarray]:
        """
        Прогнозирование на указанное количество шагов.
        
        Args:
            steps: Количество шагов для прогноза
            
        Returns:
            Optional[np.ndarray]: Прогнозные значения или None при ошибке
        """
        try:
            if self.model is None:
                raise ValueError("Модель не обучена")
                
            self.logger.info(f"Прогнозирование на {steps} шагов вперед")
            predictions = self.model.forecast(steps)
            return predictions
            
        except Exception as e:
            self.logger.error(f"Ошибка при прогнозировании: {str(e)}")
            return None
    
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Расширенная оценка качества модели.
    
        Args:
            actual: Фактические значения
            predicted: Прогнозные значения
        
        Returns:
            Dict[str, float]: Расширенные метрики качества
        """
        try:
            # Базовые метрики
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, predicted)
        
            # Расчет MAPE с обработкой нулевых значений
            # Используем только ненулевые значения для расчета MAPE
            non_zero_mask = actual != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / 
                                     actual[non_zero_mask])) * 100
            else:
                mape = np.nan
            
            # Дополнительные метрики
            errors = actual - predicted
            mean_error = np.mean(errors)
            std_error = np.std(errors)
        
            # Собираем все метрики
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape if not np.isinf(mape) else np.nan,  # Проверка на inf
                'mean_error': mean_error,
                'std_error': std_error,
            }
        
            # Записываем в лог
            self.logger.info("Оценка качества модели:")
            for metric, value in metrics.items():
                if np.isnan(value):
                    self.logger.info(f"{metric}: Не удалось рассчитать")
                else:
                    self.logger.info(f"{metric}: {value:.4f}")
        
            return metrics
        
        except Exception as e:
            self.logger.error(f"Ошибка при расчете метрик: {str(e)}")
            return {}

    def analyze_forecast(self, actual: np.ndarray, predicted: np.ndarray, dates: pd.DatetimeIndex) -> Dict:
        """
        Подробный анализ прогноза.
    
        Args:
            actual: Фактические значения
            predicted: Прогнозные значения
            dates: Индекс дат
        
        Returns:
            Dict: Результаты анализа
        """
        try:
            # Получаем базовые метрики
            metrics = self.evaluate(actual, predicted)
        
            # Анализ ошибок по времени
            errors = actual - predicted
            error_analysis = {
                'max_overpredict': float(min(errors)),
                'max_underpredict': float(max(errors)),
                'date_max_error': str(dates[np.argmax(np.abs(errors))]),
                'error_std': float(np.std(errors)),
            }
        
            # Анализ тренда
            trend_analysis = {
                'actual_trend': float(np.polyfit(range(len(actual)), actual, 1)[0]),
                'predicted_trend': float(np.polyfit(range(len(predicted)), predicted, 1)[0]),
            }
        
            # Собираем полный анализ
            analysis_results = {
                'metrics': metrics,
                'error_analysis': error_analysis,
                'trend_analysis': trend_analysis,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
            return analysis_results
        
        except Exception as e:
            self.logger.error(f"Ошибка при анализе прогноза: {str(e)}")
            return {}

    def save_forecast_report(self, analysis_results: Dict, filename: str) -> None:
        """
        Сохранение подробного отчета о прогнозе.
    
        Args:
            analysis_results: Результаты анализа
            filename: Имя файла для сохранения
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Отчет о прогнозировании температуры\n")
                f.write("=" * 50 + "\n\n")
            
                # Основные метрики
                f.write("1. Метрики качества прогноза\n")
                f.write("-" * 30 + "\n")
                for metric, value in analysis_results['metrics'].items():
                    f.write(f"{metric}: {value:.4f}\n")
                f.write("\n")
            
                # Анализ ошибок
                f.write("2. Анализ ошибок прогноза\n")
                f.write("-" * 30 + "\n")
                f.write(f"Максимальное завышение прогноза: {analysis_results['error_analysis']['max_overpredict']:.2f}°C\n")
                f.write(f"Максимальное занижение прогноза: {analysis_results['error_analysis']['max_underpredict']:.2f}°C\n")
                f.write(f"Дата максимальной ошибки: {analysis_results['error_analysis']['date_max_error']}\n")
                f.write(f"Стандартное отклонение ошибок: {analysis_results['error_analysis']['error_std']:.2f}°C\n")
                f.write("\n")
            
                # Анализ тренда
                f.write("3. Анализ тренда\n")
                f.write("-" * 30 + "\n")
                f.write(f"Тренд фактических значений: {analysis_results['trend_analysis']['actual_trend']:.4f}\n")
                f.write(f"Тренд прогнозных значений: {analysis_results['trend_analysis']['predicted_trend']:.4f}\n")
                f.write("\n")
            
                # Общая информация
                f.write("4. Дополнительная информация\n")
                f.write("-" * 30 + "\n")
                f.write(f"Дата создания отчета: {analysis_results['timestamp']}\n")
            
            self.logger.info(f"Отчет сохранен в файл: {filename}")
        
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении отчета: {str(e)}")

    def _write_analysis_report(self, f, results: Dict) -> None:
        """
        Запись результатов анализа в файл.
    
        Args:
            f: Файловый объект для записи
            results: Словарь с результатами анализа
        """
        try:
            # Базовая статистика
            f.write("1. Общая статистика\n")
            f.write("-" * 30 + "\n")
            stats = results['statistics']
            f.write(f"Количество наблюдений: {stats['n_observations']}\n")
            f.write(f"Среднее значение: {stats['mean']:.2f}\n")
            f.write(f"Стандартное отклонение: {stats['std']:.2f}\n")
            f.write(f"Минимум: {stats['min']:.2f}\n")
            f.write(f"Максимум: {stats['max']:.2f}\n")
            f.write(f"Асимметрия: {stats['skewness']:.2f}\n")
            f.write(f"Эксцесс: {stats['kurtosis']:.2f}\n\n")
        
            # Стационарность
            f.write("2. Анализ стационарности\n")
            f.write("-" * 30 + "\n")
            stationarity = results['stationarity']
            f.write(f"Тест-статистика: {stationarity['test_statistic']:.4f}\n")
            f.write(f"p-значение: {stationarity['p_value']:.4f}\n")
            f.write(f"Ряд {'стационарен' if stationarity['is_stationary'] else 'не стационарен'}\n\n")
        
            # Автокорреляция
            f.write("3. Анализ автокорреляции\n")
            f.write("-" * 30 + "\n")
            autocorr = results['autocorrelation']
            f.write(f"Количество значимых лагов ACF: {len(autocorr['significant_lags_acf'])}\n")
            f.write(f"Количество значимых лагов PACF: {len(autocorr['significant_lags_pacf'])}\n")
            f.write("Обнаруженные сезонные периоды: " + 
                    ", ".join(map(str, autocorr['seasonal_periods'])) + "\n\n")
        
            # Рекомендуемые параметры
            f.write("4. Рекомендуемые параметры SARIMA\n")
            f.write("-" * 30 + "\n")
            params = results['suggested_parameters']
            f.write(f"order (p,d,q): {params['order']}\n")
            f.write(f"seasonal_order (P,D,Q,s): {params['seasonal_order']}\n")
            f.write("Альтернативные сезонные периоды: " + 
                    ", ".join(map(str, params['alternative_seasons'])) + "\n\n")
        
            # Декомпозиция
            f.write("5. Анализ декомпозиции\n")
            f.write("-" * 30 + "\n")
            for key, value in results.items():
                if key.startswith('decomposition_period_'):
                    period = key.split('_')[-1]
                    f.write(f"\nПериод {period}:\n")
                    f.write(f"Сила тренда: {value['trend_strength']:.4f}\n")
