import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
from datetime import datetime
import logging

class ModelVisualizer:
    """
    Класс для визуализации результатов анализа и прогнозирования.
    
    Attributes:
        logger: Логгер для записи операций визуализации
        style: Стиль графиков matplotlib
        figsize: Размер графиков по умолчанию
    """
    
    def __init__(self):
        """Инициализация визуализатора."""
        self.figsize = (12, 6)
        # Используем встроенный стиль matplotlib
        plt.style.use('classic')
        self.setup_logger()
        os.makedirs('plots', exist_ok=True)

    def setup_logger(self) -> None:
        """Настройка системы логирования."""
        self.logger = logging.getLogger('ModelVisualizer')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('visualization.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def plot_forecast(self, actual: pd.Series, predicted: np.ndarray, 
                     dates: pd.DatetimeIndex, title: str = 'Прогноз температуры') -> Figure:
        """
        Построение графика прогноза.
        
        Args:
            actual: Фактические значения
            predicted: Предсказанные значения
            dates: Индекс дат
            title: Заголовок графика
            
        Returns:
            Figure: Объект графика matplotlib
        """
        try:
            fig = Figure(figsize=self.figsize)
            ax = fig.add_subplot(111)
            
            # Построение фактических значений
            ax.plot(dates, actual, label='Фактические значения', 
                   color='blue', linewidth=2)
            
            # Построение прогноза
            ax.plot(dates, predicted, label='Прогноз', 
                   color='red', linestyle='--', linewidth=2)
            
            # Настройка графика
            ax.set_title(title, fontsize=14, pad=20)
            ax.set_xlabel('Дата', fontsize=12)
            ax.set_ylabel('Температура', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Поворот подписей дат
            fig.autofmt_xdate()
            
            # Добавляем отступы
            fig.tight_layout()
            
            # Сохранение графика
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig.savefig(f'plots/forecast_{timestamp}.png', bbox_inches='tight', dpi=300)
            
            self.logger.info(f"График прогноза сохранен: forecast_{timestamp}.png")
            return fig
            
        except Exception as e:
            self.logger.error(f"Ошибка при построении графика прогноза: {str(e)}")
            raise

    def plot_metrics(self, metrics_history: List[Dict[str, float]], 
                    metric_name: str = 'mse') -> Figure:
        """
        Построение графика изменения метрик.
        
        Args:
            metrics_history: История изменения метрик
            metric_name: Название метрики для отображения
            
        Returns:
            Figure: Объект графика matplotlib
        """
        try:
            fig = Figure(figsize=self.figsize)
            ax = fig.add_subplot(111)
            
            values = [m[metric_name] for m in metrics_history]
            iterations = range(1, len(values) + 1)
            
            ax.plot(iterations, values, marker='o', 
                   linestyle='-', linewidth=2, markersize=8)
            
            ax.set_title(f'Динамика метрики {metric_name.upper()}', 
                        fontsize=14, pad=20)
            ax.set_xlabel('Итерация', fontsize=12)
            ax.set_ylabel(metric_name.upper(), fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Сохранение графика
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig.savefig(f'plots/metrics_{metric_name}_{timestamp}.png', 
                       bbox_inches='tight', dpi=300)
            
            self.logger.info(f"График метрик сохранен: metrics_{metric_name}_{timestamp}.png")
            return fig
            
        except Exception as e:
            self.logger.error(f"Ошибка при построении графика метрик: {str(e)}")
            raise

    def plot_residuals(self, residuals: np.ndarray) -> Figure:
        """
        Построение графика остатков модели.
        
        Args:
            residuals: Массив остатков
            
        Returns:
            Figure: Объект графика matplotlib
        """
        try:
            fig = Figure(figsize=(12, 8))
            
            # График распределения остатков
            ax = fig.add_subplot(111)
            
            # Гистограмма
            n, bins, patches = ax.hist(residuals, bins=30, 
                                     density=True, alpha=0.7)
            
            # Добавляем среднее и стандартное отклонение
            mean = np.mean(residuals)
            std = np.std(residuals)
            ax.axvline(mean, color='red', linestyle='dashed', 
                      label=f'Среднее: {mean:.2f}')
            
            ax.set_title('Распределение остатков', fontsize=14, pad=20)
            ax.set_xlabel('Значение остатка', fontsize=12)
            ax.set_ylabel('Частота', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            fig.tight_layout()
            
            # Сохранение графика
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig.savefig(f'plots/residuals_{timestamp}.png', 
                       bbox_inches='tight', dpi=300)
            
            self.logger.info(f"График остатков сохранен: residuals_{timestamp}.png")
            return fig
            
        except Exception as e:
            self.logger.error(f"Ошибка при построении графика остатков: {str(e)}")
            raise

    def create_results_dashboard(self, model_results: Dict) -> Figure:
        """
        Создание информационной панели с результатами модели.
        
        Args:
            model_results: Словарь с результатами модели
            
        Returns:
            Figure: Объект графика matplotlib
        """
        try:
            fig = Figure(figsize=(15, 10))
            
            # График прогноза
            ax1 = fig.add_subplot(221)
            ax1.plot(model_results['dates'], model_results['actual'], 
                    label='Фактические значения')
            ax1.plot(model_results['dates'], model_results['predicted'], 
                    label='Прогноз', linestyle='--')
            ax1.set_title('Прогноз температуры', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # График метрик
            ax2 = fig.add_subplot(222)
            metrics = ['mse', 'r2', 'mae']
            values = [model_results[m] for m in metrics]
            ax2.bar(metrics, values)
            ax2.set_title('Метрики качества', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # График остатков
            ax3 = fig.add_subplot(223)
            residuals = model_results['actual'] - model_results['predicted']
            ax3.hist(residuals, bins=30, density=True)
            ax3.set_title('Распределение остатков', fontsize=12)
            ax3.grid(True, alpha=0.3)
            
            # Текстовая информация
            ax4 = fig.add_subplot(224)
            ax4.axis('off')
            info_text = (
                f"Параметры модели:\n"
                f"Order: {model_results['order']}\n"
                f"Seasonal Order: {model_results['seasonal_order']}\n\n"
                f"Метрики:\n"
                f"MSE: {model_results['mse']:.4f}\n"
                f"R²: {model_results['r2']:.4f}\n"
                f"MAE: {model_results['mae']:.4f}"
            )
            ax4.text(0.1, 0.5, info_text, fontsize=10,
                    verticalalignment='center')
            
            fig.tight_layout()
            
            # Сохранение дашборда
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig.savefig(f'plots/dashboard_{timestamp}.png',
                       bbox_inches='tight', dpi=300)
            
            self.logger.info(f"Дашборд сохранен: dashboard_{timestamp}.png")
            return fig
            
        except Exception as e:
            self.logger.error(f"Ошибка при создании дашборда: {str(e)}")
            raise