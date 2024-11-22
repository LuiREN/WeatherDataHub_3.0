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
        """Улучшенная предобработка данных"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # Обработка категориальных признаков
        categorical_columns = [col for col in df.columns if 'cloudiness' in col or 'wind_direction' in col]
        df[categorical_columns] = df[categorical_columns].fillna(0)
        
        # Обработка числовых признаков
        numeric_columns = ['temperature_day', 'temperature_evening', 'pressure_day', 
                         'pressure_evening', 'wind_speed_day', 'wind_speed_evening']
        
        for col in numeric_columns:
            if col in df.columns:
                # Обработка выбросов
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 2 * IQR
                upper_bound = Q3 + 2 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
                
                # Заполнение пропусков
                if 'temperature' in col:
                    # Для температуры используем сезонную интерполяцию
                    df[col] = df[col].interpolate(method='time', limit_direction='both', order=3)
                elif 'pressure' in col:
                    # Для давления используем линейную интерполяцию
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
                else:
                    # Для остальных числовых признаков используем скользящее среднее
                    df[col] = df[col].fillna(df[col].rolling(window=3, min_periods=1).mean())
        
        return df

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
        """Улучшенное обучение модели SARIMA"""
        try:
            self.logger.info(f"Начало обучения модели\n"
                           f"Параметры: order={order}, seasonal_order={seasonal_order}")
            
            # Проверка и подготовка данных
            if train_data.isnull().any():
                raise ValueError("Обнаружены пропущенные значения в данных")
            
            # Декомпозиция временного ряда
            decomposition = seasonal_decompose(train_data, period=seasonal_order[3])
            seasonal = decomposition.seasonal
            resid = decomposition.resid
            
            # Создание модели с оптимизированными параметрами
            model = SARIMAX(
                train_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                initialization='approximate_diffuse',
                hamilton_representation=True,
                concentrate_scale=True
            )
            
            # Обучение с оптимизированными параметрами
            self.model = model.fit(
                disp=False,
                maxiter=1000,
                method='lbfgs',
                optim_score='harvey',
                cov_type='robust'
            )
            
            # Получение прогноза на тренировочном наборе
            train_predictions = self.model.get_prediction(start=0)
            predicted_values = train_predictions.predicted_mean
            
            # Расчет расширенных метрик
            metrics = {
                'mse': mean_squared_error(train_data, predicted_values),
                'rmse': np.sqrt(mean_squared_error(train_data, predicted_values)),
                'mae': mean_absolute_error(train_data, predicted_values),
                'aic': self.model.aic,
                'bic': self.model.bic,
                'seasonal_strength': 1 - np.var(resid)/np.var(seasonal + resid)
            }
            
            return {
                'status': 'success',
                'metrics': metrics,
                'parameters': {
                    'order': order,
                    'seasonal_order': seasonal_order
                },
                'model_info': {
                    'nobs': self.model.nobs,
                    'seasonal_periods': seasonal_order[3]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def tune_parameters(self) -> Dict:
        """
        Перебор гиперпараметров модели.
        Проверяет различные комбинации параметров и записывает результаты.
    
        Returns:
            Dict: Результаты подбора параметров
        """
        try:
            # Определяем диапазоны параметров
            parameter_ranges = {
                'p': [0, 1, 2],
                'd': [0, 1],
                'q': [0, 1, 2],
                'P': [0, 1],
                'D': [0, 1],
                'Q': [0, 1],
                's': [7, 12]  # недельная и годовая сезонность
            }
        
            results = []
            best_result = None
            best_rmse = float('inf')
        
            self.logger.info("Начало подбора гиперпараметров")
        
            # Основные параметры (p, d, q)
            for p in parameter_ranges['p']:
                for d in parameter_ranges['d']:
                    for q in parameter_ranges['q']:
                        # Сезонные параметры (P, D, Q, s)
                        for P in parameter_ranges['P']:
                            for D in parameter_ranges['D']:
                                for Q in parameter_ranges['Q']:
                                    for s in parameter_ranges['s']:
                                        # Формируем параметры
                                        order = (p, d, q)
                                        seasonal_order = (P, D, Q, s)
                                    
                                        try:
                                            # Обучаем модель
                                            train_result = self.train(
                                                self.train_data,
                                                order,
                                                seasonal_order
                                            )
                                        
                                            if train_result['status'] == 'success':
                                                # Записываем результаты
                                                result = {
                                                    'parameters': {
                                                        'order': order,
                                                        'seasonal_order': seasonal_order
                                                    },
                                                    'metrics': train_result['metrics'],
                                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                                }
                                            
                                                results.append(result)
                                            
                                                # Проверяем, лучший ли это результат
                                                current_rmse = train_result['metrics']['rmse']
                                                if current_rmse < best_rmse:
                                                    best_rmse = current_rmse
                                                    best_result = result
                                                
                                                self.logger.info(
                                                    f"Проверена комбинация: order={order}, "
                                                    f"seasonal_order={seasonal_order}, "
                                                    f"RMSE={current_rmse:.4f}"
                                                )
                                            
                                        except Exception as e:
                                            self.logger.warning(
                                                f"Ошибка при проверке комбинации "
                                                f"order={order}, seasonal_order={seasonal_order}: {str(e)}"
                                            )
                                            continue
        
            # Сохраняем результаты
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f'results/parameter_tuning_{timestamp}.txt'
        
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("Результаты подбора гиперпараметров SARIMA\n")
                f.write("=" * 50 + "\n\n")
            
                # Лучший результат
                f.write("Лучшая комбинация параметров:\n")
                f.write("-" * 30 + "\n")
                f.write(f"order: {best_result['parameters']['order']}\n")
                f.write(f"seasonal_order: {best_result['parameters']['seasonal_order']}\n")
                f.write(f"RMSE: {best_result['metrics']['rmse']:.4f}\n")
                f.write(f"MSE: {best_result['metrics']['mse']:.4f}\n")
                f.write(f"MAPE: {best_result['metrics']['mape']:.2f}%\n\n")
            
                # Все результаты
                f.write("Все проверенные комбинации:\n")
                f.write("-" * 30 + "\n")
                for result in results:
                    f.write(f"\norder: {result['parameters']['order']}\n")
                    f.write(f"seasonal_order: {result['parameters']['seasonal_order']}\n")
                    f.write(f"RMSE: {result['metrics']['rmse']:.4f}\n")
                    f.write(f"MSE: {result['metrics']['mse']:.4f}\n")
                    f.write(f"MAPE: {result['metrics']['mape']:.2f}%\n")
                    f.write("-" * 20 + "\n")
                
            return {
                'best_result': best_result,
                'all_results': results,
                'results_file': results_file
            }
        
        except Exception as e:
            self.logger.error(f"Ошибка при подборе параметров: {str(e)}")
            return {'error': str(e)}
    
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
