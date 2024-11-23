import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from statsmodels.tsa.stattools import adfuller, acf
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
        """Комплексный анализ временного ряда."""
        try:
            self.logger.info("Начало комплексного анализа временного ряда")
            self.logger.info(f"Размер входных данных: {len(data)}")
        
            # Предварительная обработка данных
            # Удаляем NaN значения
            data_cleaned = data.dropna()
        
            if len(data_cleaned) < 2:
                raise ValueError("Недостаточно данных для анализа после удаления NaN значений")
            
            self.logger.info(f"Количество данных после очистки: {len(data_cleaned)}")
        
            results = {}
        
            # Проверяем стационарность
            self.logger.info("Проверка стационарности...")
            stationarity_results = self.check_stationarity(data_cleaned)
            if stationarity_results is not None:
                results['stationarity'] = stationarity_results
            else:
                self.logger.warning("Не удалось выполнить проверку стационарности")
                results['stationarity'] = {
                    'test_statistic': None,
                    'p_value': None,
                    'is_stationary': None,
                    'error': 'Ошибка при проверке стационарности'
                }
        
            # Анализируем автокорреляцию
            self.logger.info("Анализ автокорреляции...")
            autocorr_results = self.analyze_autocorrelation(data_cleaned)
            if autocorr_results is not None:
                results['autocorrelation'] = autocorr_results
            else:
                self.logger.warning("Не удалось выполнить анализ автокорреляции")
                results['autocorrelation'] = {
                    'significant_lags': [],
                    'error': 'Ошибка при анализе автокорреляции'
                }
        
            # Создаем отчет
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'results/time_series_analysis_{timestamp}.txt'
        
            os.makedirs('results', exist_ok=True)
        
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Анализ временного ряда температуры\n")
                f.write("=" * 50 + "\n\n")
            
                # Общая информация
                f.write("1. Общая информация\n")
                f.write("-" * 30 + "\n")
                f.write(f"Исходное количество наблюдений: {len(data)}\n")
                f.write(f"Количество наблюдений после очистки: {len(data_cleaned)}\n")
                f.write(f"Удалено пропущенных значений: {len(data) - len(data_cleaned)}\n")
                f.write(f"Среднее значение: {data_cleaned.mean():.2f}\n")
                f.write(f"Стандартное отклонение: {data_cleaned.std():.2f}\n")
                f.write(f"Минимум: {data_cleaned.min():.2f}\n")
                f.write(f"Максимум: {data_cleaned.max():.2f}\n\n")
            
                # Стационарность
                f.write("2. Анализ стационарности\n")
                f.write("-" * 30 + "\n")
                if 'error' in results['stationarity']:
                    f.write(f"Ошибка: {results['stationarity']['error']}\n")
                else:
                    f.write(f"Тест-статистика: {results['stationarity']['test_statistic']:.4f}\n")
                    f.write(f"p-значение: {results['stationarity']['p_value']:.4f}\n")
                    f.write(f"Ряд {'стационарен' if results['stationarity']['is_stationary'] else 'не стационарен'}\n")
                f.write("\n")
            
                # Автокорреляция
                f.write("3. Анализ автокорреляции\n")
                f.write("-" * 30 + "\n")
                if 'error' in results['autocorrelation']:
                    f.write(f"Ошибка: {results['autocorrelation']['error']}\n")
                else:
                    f.write(f"Количество значимых лагов: {len(results['autocorrelation']['significant_lags'])}\n")
                    if results['autocorrelation']['significant_lags']:
                        f.write("Значимые лаги:\n")
                        for lag, value in results['autocorrelation']['significant_lags'][:5]:
                            f.write(f"Лаг {lag}: {value:.4f}\n")
            
                f.write(f"\nДата создания отчета: {timestamp}")
        
            self.logger.info(f"Анализ завершен, результаты сохранены в {filename}")
        
            # Добавляем путь к файлу и статистику в результаты
            results['report_file'] = filename
            results['statistics'] = {
                'n_observations': len(data_cleaned),
                'mean': data_cleaned.mean(),
                'std': data_cleaned.std(),
                'min': data_cleaned.min(),
                'max': data_cleaned.max()
            }
        
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
        Обучение модели SARIMA с декомпозицией временного ряда.
    
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
            
            # Проверяем на NaN перед началом обработки
            if train_data.isnull().any():
                self.logger.info("Обнаружены пропущенные значения, выполняется заполнение...")
                train_data = train_data.interpolate(method='time')
                train_data = train_data.ffill().bfill()
            
            # Установка частоты
            if not train_data.index.freq:
                train_data = train_data.asfreq('D')
            
            self.logger.info(f"Размер данных: {len(train_data)}")
            self.logger.info("Выполняется декомпозиция временного ряда...")
        
            # Декомпозиция временного ряда
            decomposition = seasonal_decompose(train_data, period=seasonal_order[3])
        
            # Обработка компонентов
            seasonal = decomposition.seasonal.interpolate(method='time')
            seasonal = seasonal.ffill().bfill()
        
            trend = decomposition.trend.interpolate(method='time')
            trend = trend.ffill().bfill()
        
            residual = decomposition.resid.interpolate(method='time')
            residual = residual.ffill().bfill()
        
            # Нормализация остатков
            residual_mean = residual.mean()
            residual_std = residual.std()
            residual_normalized = (residual - residual_mean) / residual_std
        
            self.logger.info("Создание и обучение модели SARIMA...")
        
            # Создание модели на нормализованных остатках
            model = SARIMAX(
                residual_normalized,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                simple_differencing=True,
                initialization='approximate_diffuse'
            )
        
            # Обучение модели с обработкой проблем сходимости
            try:
                self.model = model.fit(
                    disp=False,
                    maxiter=1000,
                    method='lbfgs',
                    cov_type='robust',
                    optim_score='harvey',
                    optim_complex_step=True
                )
            except Exception as e:
                self.logger.warning(f"Первая попытка обучения не удалась: {str(e)}")
                # Вторая попытка с другими параметрами
                self.model = model.fit(
                    disp=False,
                    maxiter=2000,
                    method='powell',
                    cov_type='robust'
                )
        
            self.logger.info("Получение прогноза...")
        
            # Получение прогноза и обратное преобразование
            predicted_residuals = self.model.fittedvalues
            predicted_residuals = (predicted_residuals * residual_std) + residual_mean
        
            # Восстановление прогноза с сезонностью и трендом
            predicted_values = predicted_residuals + seasonal + trend
        
            # Синхронизация индексов
            valid_idx = train_data.index.intersection(predicted_values.index)
            actual_values = train_data[valid_idx]
            predicted_values = predicted_values[valid_idx]
        
            # Проверка на NaN после всех преобразований
            if predicted_values.isnull().any() or actual_values.isnull().any():
                self.logger.warning("Обнаружены NaN в результатах, выполняется финальная очистка...")
                mask = ~(predicted_values.isnull() | actual_values.isnull())
                actual_values = actual_values[mask]
                predicted_values = predicted_values[mask]
        
            self.logger.info("Расчет метрик...")
        
            # Расчет ошибок и метрик
            errors = actual_values - predicted_values
            abs_errors = np.abs(errors)
        
            # Безопасный расчет MAPE
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_errors = abs_errors / np.abs(actual_values)
                mape = np.nanmean(rel_errors) * 100
        
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
                'seasonal_strength': 1 - np.var(residual)/np.var(seasonal + residual),
                'trend_strength': 1 - np.var(residual)/np.var(trend + residual),
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
