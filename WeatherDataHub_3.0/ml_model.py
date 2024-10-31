import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import logging

class ModelMetrics:
    """
    Класс для расчета и анализа метрик качества модели.
    
    Methods:
        mse: Расчет средней квадратичной ошибки
        r2: Расчет коэффициента детерминации
        mae: Расчет средней абсолютной ошибки
        evaluate: Расчет всех метрик для модели
    """
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Расчет средней квадратичной ошибки (Mean Squared Error).
        
        Args:
            y_true: Реальные значения
            y_pred: Предсказанные значения
            
        Returns:
            float: Значение MSE
        """
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Расчет коэффициента детерминации (R²).
        
        Args:
            y_true: Реальные значения
            y_pred: Предсказанные значения
            
        Returns:
            float: Значение R² в диапазоне [0, 1]
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Расчет средней абсолютной ошибки (Mean Absolute Error).
        
        Args:
            y_true: Реальные значения
            y_pred: Предсказанные значения
            
        Returns:
            float: Значение MAE
        """
        return np.mean(np.abs(y_true - y_pred))

    @classmethod
    def evaluate(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Расчет всех метрик качества.
        
        Args:
            y_true: Реальные значения
            y_pred: Предсказанные значения
            
        Returns:
            Dict[str, float]: Словарь с метриками
        """
        return {
            'mse': cls.mse(y_true, y_pred),
            'r2': cls.r2(y_true, y_pred),
            'mae': cls.mae(y_true, y_pred)
        }

class WeatherModel:
    """
    Класс для работы с моделью прогнозирования погоды.
    
    Attributes:
        model: Обученная модель SARIMA
        metrics: Экземпляр класса для расчета метрик
        best_score: Лучший показатель качества
        best_params: Лучшие параметры модели
        logger: Логгер для записи процесса работы
    """
    
    def __init__(self):
        """Инициализация модели."""
        self.model = None
        self.metrics = ModelMetrics()
        self.best_score = float('inf')
        self.best_params = None
        self.setup_logger()

    def setup_logger(self) -> None:
        """Настройка системы логирования."""
        self.logger = logging.getLogger('WeatherModel')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('weather_model.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def train(self, train_data: pd.DataFrame, 
              order: Tuple[int, int, int], 
              seasonal_order: Tuple[int, int, int, int]) -> Dict:
        """
        Обучение модели SARIMA.
        
        Args:
            train_data: DataFrame с данными для обучения
            order: Параметры (p,d,q) для несезонной части
            seasonal_order: Параметры (P,D,Q,s) для сезонной части
            
        Returns:
            Dict: Результаты обучения
        """
        try:
            self.logger.info(f"Начало обучения модели с параметрами: order={order}, seasonal_order={seasonal_order}")
            model = SARIMAX(train_data['temperature_day'], order=order, seasonal_order=seasonal_order)
            self.model = model.fit(disp=False)
            self.logger.info("Модель успешно обучена")
            
            return {
                'status': 'success',
                'model': self.model,
                'order': order,
                'seasonal_order': seasonal_order,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def predict(self, steps: int) -> Optional[np.ndarray]:
        """
        Прогнозирование на указанное количество шагов вперед.
        
        Args:
            steps: Количество шагов для прогноза
            
        Returns:
            Optional[np.ndarray]: Массив с прогнозами или None в случае ошибки
        """
        try:
            if self.model is None:
                raise ValueError("Модель не обучена")
            self.logger.info(f"Прогнозирование на {steps} шагов вперед")
            return self.model.forecast(steps)
        except Exception as e:
            self.logger.error(f"Ошибка при прогнозировании: {str(e)}")
            return None

    def evaluate_model(self, test_data: pd.DataFrame, predictions: np.ndarray) -> Dict[str, float]:
        """
        Оценка качества модели на тестовых данных.
        
        Args:
            test_data: DataFrame с тестовыми данными
            predictions: Прогнозы модели
            
        Returns:
            Dict[str, float]: Метрики качества
        """
        try:
            metrics = self.metrics.evaluate(
                test_data['temperature_day'].values,
                predictions
            )
            self.logger.info(f"Оценка модели: MSE={metrics['mse']:.4f}, R2={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
            
            if metrics['mse'] < self.best_score:
                self.best_score = metrics['mse']
                self.best_params = {
                    'order': self.model.specification['order'],
                    'seasonal_order': self.model.specification['seasonal_order']
                }
            return metrics
        except Exception as e:
            self.logger.error(f"Ошибка при оценке модели: {str(e)}")
            return {}

    def tune_parameters(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """
        Подбор оптимальных параметров модели.
        
        Args:
            train_data: Данные для обучения
            test_data: Данные для тестирования
            
        Returns:
            Dict: Результаты лучшей модели
        """
        try:
            self.logger.info("Начало подбора параметров")
            parameter_grid = {
                'p': range(0, 3), 'd': range(0, 2), 'q': range(0, 3),
                'P': range(0, 2), 'D': range(0, 2), 'Q': range(0, 2),
                's': [12]
            }
            
            best_result = None
            results = []
            
            for p in parameter_grid['p']:
                for d in parameter_grid['d']:
                    for q in parameter_grid['q']:
                        for P in parameter_grid['P']:
                            for D in parameter_grid['D']:
                                for Q in parameter_grid['Q']:
                                    for s in parameter_grid['s']:
                                        try:
                                            train_result = self.train(
                                                train_data,
                                                order=(p, d, q),
                                                seasonal_order=(P, D, Q, s)
                                            )
                                            
                                            if train_result['status'] == 'success':
                                                predictions = self.predict(len(test_data))
                                                if predictions is not None:
                                                    metrics = self.evaluate_model(test_data, predictions)
                                                    result = {
                                                        'order': (p, d, q),
                                                        'seasonal_order': (P, D, Q, s),
                                                        **metrics
                                                    }
                                                    results.append(result)
                                                    if best_result is None or metrics['mse'] < best_result['mse']:
                                                        best_result = result
                                        except Exception as e:
                                            self.logger.warning(f"Пропуск комбинации параметров из-за ошибки: {str(e)}")
            
            self.logger.info("Подбор параметров завершен")
            return {
                'best_result': best_result,
                'all_results': results,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            self.logger.error(f"Ошибка при подборе параметров: {str(e)}")
            return {'status': 'error', 'message': str(e)}