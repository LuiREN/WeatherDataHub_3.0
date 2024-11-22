import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import os
from datetime import datetime
import logging

class ModelVisualizer:
    """
    Класс для визуализации результатов прогнозирования погоды.
    
    Attributes:
        logger: Логгер для записи операций
    """
    
    def __init__(self):
        """Инициализация визуализатора."""
        self.setup_logger()
        os.makedirs('plots', exist_ok=True)
        plt.style.use('seaborn')  # Используем стиль для лучшей визуализации

    def setup_logger(self) -> None:
        """Настройка системы логирования."""
        self.logger = logging.getLogger('ModelVisualizer')
        self.logger.setLevel(logging.INFO)
        
        os.makedirs('logs', exist_ok=True)
        handler = logging.FileHandler(
            f'logs/visualizer_{datetime.now().strftime("%Y%m%d")}.log'
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def plot_forecast(self, actual: pd.Series, predicted: np.ndarray) -> Figure:
        """
        Построение графика прогноза температуры.
        
        Args:
            actual: Фактические значения
            predicted: Прогнозируемые значения
            
        Returns:
            Figure: График сравнения
        """
        try:
            fig = Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            
            # Построение графиков
            ax.plot(actual.index, actual.values, 
                   label='Фактическая температура', 
                   color='blue', linewidth=2)
            
            ax.plot(actual.index, predicted, 
                   label='Прогноз', 
                   color='red', 
                   linestyle='--', 
                   linewidth=2)
            
            # Настройка внешнего вида
            ax.set_title('Прогноз температуры', fontsize=14, pad=20)
            ax.set_xlabel('Дата', fontsize=12)
            ax.set_ylabel('Температура (°C)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Поворот подписей дат
            fig.autofmt_xdate()
            
            # Добавляем отступы
            fig.tight_layout()
            
            # Сохраняем график
            self.save_plot(fig, 'forecast')
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Ошибка при построении графика прогноза: {str(e)}")
            raise

    def plot_error_analysis(self, actual: pd.Series, predicted: np.ndarray) -> Figure:
        """
        Построение графика анализа ошибок.
        
        Args:
            actual: Фактические значения
            predicted: Прогнозируемые значения
            
        Returns:
            Figure: График анализа ошибок
        """
        try:
            errors = actual - predicted
            
            fig = Figure(figsize=(12, 8))
            
            # График ошибок во времени
            ax1 = fig.add_subplot(211)
            ax1.plot(actual.index, errors, color='green', marker='o', linestyle='-', 
                    markersize=4, alpha=0.6)
            ax1.axhline(y=0, color='r', linestyle='--')
            ax1.set_title('Ошибки прогноза во времени', fontsize=12)
            ax1.set_xlabel('Дата')
            ax1.set_ylabel('Ошибка (°C)')
            ax1.grid(True, alpha=0.3)
            
            # Гистограмма ошибок
            ax2 = fig.add_subplot(212)
            ax2.hist(errors, bins=30, color='blue', alpha=0.6, density=True)
            ax2.axvline(errors.mean(), color='r', linestyle='--', 
                       label=f'Среднее: {errors.mean():.2f}')
            ax2.set_title('Распределение ошибок', fontsize=12)
            ax2.set_xlabel('Ошибка (°C)')
            ax2.set_ylabel('Частота')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            fig.tight_layout()
            
            # Сохраняем график
            self.save_plot(fig, 'error_analysis')
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Ошибка при построении графика анализа ошибок: {str(e)}")
            raise

    def save_plot(self, fig: Figure, plot_type: str) -> None:
        """
        Сохранение графика.
        
        Args:
            fig: График для сохранения
            plot_type: Тип графика
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'plots/{plot_type}_{timestamp}.png'
            
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"График сохранен: {filename}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении графика: {str(e)}")
            raise