import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from typing import Optional, Tuple, Dict
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
        self.setup_logger()
        os.makedirs('plots', exist_ok=True)
        plt.style.use('seaborn')
        self.set_plot_style()

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

    def set_plot_style(self):
        """Установка улучшенного стиля графиков"""
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.grid'] = True

    def plot_forecast(self, actual: pd.Series, predicted: np.ndarray) -> Figure:
        """Улучшенная визуализация прогноза"""
        try:
            fig = Figure(figsize=(12, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
            
            # Основной график
            ax1 = fig.add_subplot(gs[0])
            
            # Строим графики с доверительными интервалами
            ax1.plot(actual.index, actual.values, 
                    label='Фактическая температура', 
                    color='#2ecc71', linewidth=2)
            
            ax1.plot(actual.index, predicted, 
                    label='Прогноз', 
                    color='#e74c3c', 
                    linestyle='--', 
                    linewidth=2)
            
            # Добавляем доверительный интервал
            std_err = np.std(actual.values - predicted)
            ax1.fill_between(actual.index,
                           predicted - 1.96*std_err,
                           predicted + 1.96*std_err,
                           color='#e74c3c', alpha=0.2,
                           label='95% доверительный интервал')
            
            ax1.set_title('Прогноз температуры', fontsize=14, pad=20)
            ax1.set_xlabel('')
            ax1.set_ylabel('Температура (°C)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10, loc='upper right')
            
            # График ошибок
            ax2 = fig.add_subplot(gs[1])
            errors = actual.values - predicted
            ax2.plot(actual.index, errors, color='#3498db', marker='o', 
                    linestyle='-', markersize=4, alpha=0.6)
            ax2.axhline(y=0, color='#e74c3c', linestyle='--')
            ax2.set_title('Ошибки прогноза', fontsize=12)
            ax2.set_xlabel('Дата', fontsize=10)
            ax2.set_ylabel('Ошибка (°C)', fontsize=10)
            
            # Поворот подписей дат
            for ax in [ax1, ax2]:
                ax.tick_params(axis='x', rotation=45)
            
            fig.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Ошибка при построении графика прогноза: {str(e)}")
            raise

    def plot_error_analysis(self, actual: pd.Series, predicted: np.ndarray) -> Figure:
        """Расширенный анализ ошибок"""
        try:
            errors = actual - predicted
            fig = Figure(figsize=(12, 10))
            gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.4, wspace=0.3)
            
            # 1. График ошибок во времени
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(actual.index, errors, color='#3498db', marker='o', 
                    linestyle='-', markersize=4, alpha=0.6)
            ax1.axhline(y=0, color='#e74c3c', linestyle='--')
            ax1.set_title('Ошибки прогноза во времени', fontsize=12)
            ax1.set_xlabel('Дата')
            ax1.set_ylabel('Ошибка (°C)')
            
            # 2. QQ-plot
            ax2 = fig.add_subplot(gs[1, 0])
            stats.probplot(errors, dist="norm", plot=ax2)
            ax2.set_title('Q-Q график ошибок')
            
            # 3. Гистограмма ошибок
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.hist(errors, bins=30, color='#3498db', alpha=0.6, density=True)
            mu, std = stats.norm.fit(errors)
            xmin, xmax = ax3.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, mu, std)
            ax3.plot(x, p, '#e74c3c', linewidth=2)
            ax3.set_title('Распределение ошибок')
            
            # 4. Автокорреляция ошибок
            ax4 = fig.add_subplot(gs[2, 0])
            pd.plotting.autocorrelation_plot(errors, ax=ax4)
            ax4.set_title('Автокорреляция ошибок')
            
            # 5. Зависимость ошибок от predicted
            ax5 = fig.add_subplot(gs[2, 1])
            ax5.scatter(predicted, errors, alpha=0.5)
            ax5.axhline(y=0, color='#e74c3c', linestyle='--')
            ax5.set_title('Ошибки vs Прогноз')
            ax5.set_xlabel('Прогноз')
            ax5.set_ylabel('Ошибка')
            
            # Статистика ошибок
            stats_text = (
                f'MSE: {np.mean(errors**2):.2f}\n'
                f'RMSE: {np.sqrt(np.mean(errors**2)):.2f}\n'
                f'MAE: {np.mean(np.abs(errors)):.2f}\n'
                f'Std: {np.std(errors):.2f}'
            )
            fig.text(0.02, 0.02, stats_text, fontsize=10)
            
            fig.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Ошибка при построении графика анализа ошибок: {str(e)}")
            raise

    def plot_seasonal_decomposition(self, data: pd.Series, period: int) -> Figure:
        """Декомпозиция временного ряда"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Выполняем декомпозицию
            decomposition = seasonal_decompose(data, period=period)
            
            fig = Figure(figsize=(12, 12))
            gs = fig.add_gridspec(4, 1, height_ratios=[1.5, 1, 1, 1], hspace=0.4)
            
            # Исходные данные
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(data.index, data.values, color='#2ecc71')
            ax1.set_title('Исходный ряд')
            
            # Тренд
            ax2 = fig.add_subplot(gs[1])
            ax2.plot(data.index, decomposition.trend, color='#e74c3c')
            ax2.set_title('Тренд')
            
            # Сезонность
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(data.index, decomposition.seasonal, color='#3498db')
            ax3.set_title('Сезонная компонента')
            
            # Остатки
            ax4 = fig.add_subplot(gs[3])
            ax4.plot(data.index, decomposition.resid, color='#95a5a6')
            ax4.set_title('Остатки')
            
            for ax in [ax1, ax2, ax3, ax4]:
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Ошибка при построении декомпозиции: {str(e)}")
            raise

    def save_plot(self, fig: Figure, plot_type: str) -> str:
        """Сохранение графика с улучшенным качеством"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'plots/{plot_type}_{timestamp}.png'
            
            # Сохраняем с высоким DPI и оптимальным форматом
            fig.savefig(filename, dpi=300, bbox_inches='tight', 
                       format='png', facecolor='white', edgecolor='none')
            
            self.logger.info(f"График сохранен: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении графика: {str(e)}")
            raise