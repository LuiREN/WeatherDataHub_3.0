from typing import Optional, Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QGroupBox, QMessageBox, QSpinBox,
    QFrame, QFileDialog, QProgressBar
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import logging
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from optimized_table import OptimizedTableWidget
import os
import pickle
import json

class MLTab(QWidget):
    """
    Вкладка для машинного обучения и прогнозирования временных рядов.
    
    Attributes:
        df (Optional[pd.DataFrame]): Текущий датафрейм с данными
        current_file (Optional[str]): Путь к текущему файлу
        train_data (Optional[pd.DataFrame]): Данные для обучения
        test_data (Optional[pd.DataFrame]): Данные для тестирования
        best_model: Лучшая модель после обучения
        best_score (float): Лучший показатель качества модели
        model_results (List[Dict]): История результатов обучения
    """
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Инициализация вкладки машинного обучения.

        Args:
            parent: Родительский виджет
        """
        super().__init__(parent)
        self.df: Optional[pd.DataFrame] = None
        self.current_file: Optional[str] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.best_model = None
        self.best_score: float = float('inf')
        self.model_results: List[Dict] = []
        
        # Настройка логирования и интерфейса
        self.setup_logging()
        self.init_ui()
        self.create_directories()

    def setup_logging(self) -> None:
        """Настройка системы логирования."""
        self.logger = logging.getLogger('MLTab')
        self.logger.setLevel(logging.INFO)
        
        # Создаем форматтер для логов
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Обработчик для файла
        fh = logging.FileHandler('ml_training.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def create_directories(self) -> None:
        """Создание необходимых директорий."""
        directories = ['ml_results', 'ml_models', 'ml_plots']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def init_ui(self) -> None:
        """Инициализация пользовательского интерфейса."""
        main_layout = QHBoxLayout()
        
        # Создаем панели
        left_panel = self.create_left_panel()
        right_panel = self.create_right_panel()
        
        # Добавляем панели в главный layout
        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(right_panel, stretch=2)
        
        self.setLayout(main_layout)

    def create_left_panel(self) -> QWidget:
        """
        Создание левой панели с элементами управления.
        
        Returns:
            QWidget: Виджет левой панели
        """
        left_panel = QWidget()
        layout = QVBoxLayout()
        
        # Группа подготовки данных
        data_group = self.create_data_preparation_group()
        layout.addWidget(data_group)
        
        # Группа параметров SARIMA
        sarima_group = self.create_sarima_group()
        layout.addWidget(sarima_group)
        
        # Группа управления моделью
        model_group = self.create_model_control_group()
        layout.addWidget(model_group)
        
        # Индикатор прогресса
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        left_panel.setLayout(layout)
        return left_panel

    def create_data_preparation_group(self) -> QGroupBox:
        """Создание группы подготовки данных."""
        group = QGroupBox("1. Подготовка данных")
        layout = QVBoxLayout()
        
        # Размер тестовой выборки
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Размер тестовой выборки:"))
        self.test_size_spin = QSpinBox()
        self.test_size_spin.setRange(10, 40)
        self.test_size_spin.setValue(20)
        self.test_size_spin.setSuffix("%")
        size_layout.addWidget(self.test_size_spin)
        layout.addLayout(size_layout)
        
        # Кнопки
        self.prepare_data_btn = QPushButton("Подготовить данные")
        self.prepare_data_btn.setEnabled(False)
        self.prepare_data_btn.clicked.connect(self.prepare_data)
        layout.addWidget(self.prepare_data_btn)
        
        self.split_data_btn = QPushButton("Разделить данные")
        self.split_data_btn.setEnabled(False)
        self.split_data_btn.clicked.connect(self.split_data)
        layout.addWidget(self.split_data_btn)
        
        group.setLayout(layout)
        return group

    def create_sarima_group(self) -> QGroupBox:
        """Создание группы параметров SARIMA."""
        group = QGroupBox("2. Параметры SARIMA")
        layout = QVBoxLayout()
        
        # Параметры p, d, q
        pdq_layout = QHBoxLayout()
        pdq_layout.addWidget(QLabel("p:"))
        self.p_spin = QSpinBox()
        self.p_spin.setRange(0, 3)
        self.p_spin.setValue(1)
        pdq_layout.addWidget(self.p_spin)
        
        pdq_layout.addWidget(QLabel("d:"))
        self.d_spin = QSpinBox()
        self.d_spin.setRange(0, 2)
        self.d_spin.setValue(1)
        pdq_layout.addWidget(self.d_spin)
        
        pdq_layout.addWidget(QLabel("q:"))
        self.q_spin = QSpinBox()
        self.q_spin.setRange(0, 3)
        self.q_spin.setValue(1)
        pdq_layout.addWidget(self.q_spin)
        
        layout.addLayout(pdq_layout)
        
        # Сезонные параметры
        seasonal_layout = QHBoxLayout()
        seasonal_layout.addWidget(QLabel("P:"))
        self.P_spin = QSpinBox()
        self.P_spin.setRange(0, 2)
        self.P_spin.setValue(1)
        seasonal_layout.addWidget(self.P_spin)
        
        seasonal_layout.addWidget(QLabel("D:"))
        self.D_spin = QSpinBox()
        self.D_spin.setRange(0, 1)
        self.D_spin.setValue(1)
        seasonal_layout.addWidget(self.D_spin)
        
        seasonal_layout.addWidget(QLabel("Q:"))
        self.Q_spin = QSpinBox()
        self.Q_spin.setRange(0, 2)
        self.Q_spin.setValue(1)
        seasonal_layout.addWidget(self.Q_spin)
        
        seasonal_layout.addWidget(QLabel("s:"))
        self.s_spin = QSpinBox()
        self.s_spin.setRange(1, 24)
        self.s_spin.setValue(12)
        seasonal_layout.addWidget(self.s_spin)
        
        layout.addLayout(seasonal_layout)
        group.setLayout(layout)
        return group

    def create_model_control_group(self) -> QGroupBox:
        """Создание группы управления моделью."""
        group = QGroupBox("3. Управление моделью")
        layout = QVBoxLayout()
        
        # Кнопки управления
        self.train_btn = QPushButton("Обучить модель")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self.train_model)
        layout.addWidget(self.train_btn)
        
        self.tune_btn = QPushButton("Подобрать гиперпараметры")
        self.tune_btn.setEnabled(False)
        self.tune_btn.clicked.connect(self.tune_hyperparameters)
        layout.addWidget(self.tune_btn)
        
        self.save_model_btn = QPushButton("Сохранить лучшую модель")
        self.save_model_btn.setEnabled(False)
        self.save_model_btn.clicked.connect(self.save_model)
        layout.addWidget(self.save_model_btn)
        
        self.load_model_btn = QPushButton("Загрузить модель")
        self.load_model_btn.clicked.connect(self.load_model)
        layout.addWidget(self.load_model_btn)
        
        group.setLayout(layout)
        return group

    def create_right_panel(self) -> QWidget:
        """Создание правой панели для отображения результатов."""
        right_panel = QWidget()
        layout = QVBoxLayout()
        
        # Информационная метка
        self.info_label = QLabel("Загрузите данные для начала работы")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.info_label)
        
        # Таблица для предпросмотра данных
        self.data_preview = OptimizedTableWidget()
        layout.addWidget(self.data_preview)
        
        right_panel.setLayout(layout)
        self.right_panel = right_panel
        return right_panel

    def load_data(self, df: pd.DataFrame, file_path: Optional[str] = None) -> None:
        """
        Загрузка данных для анализа.
        
        Args:
            df: DataFrame с данными
            file_path: Путь к файлу данных
        """
        try:
            self.df = df.copy()
            self.current_file = file_path
            
            if self.validate_data(df):
                self.data_preview.load_data(df)
                self.prepare_data_btn.setEnabled(True)
                
                self.info_label.setText(
                    f"Данные загружены: {len(df)} записей\n"
                    f"Период: с {df['date'].min()} по {df['date'].max()}"
                )
                
                self.logger.info(f"Загружены данные: {len(df)} записей")
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при загрузке данных: {str(e)}")

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Проверка корректности данных.
        
        Args:
            df: DataFrame для проверки
            
        Returns:
            bool: True если данные корректны
        """
        if df is None:
            return False
            
        required_columns = {'date', 'temperature_day'}
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            QMessageBox.warning(
                self,
                "Ошибка данных",
                f"Отсутствуют столбцы: {', '.join(missing_columns)}"
            )
            return False
            
        return True

    def prepare_data(self) -> None:
        """Подготовка данных для анализа."""
        if self.df is None:
            return
            
        try:
            df = self.df.copy()
            
            # Преобразуем дату
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Сортируем по дате
            df.sort_index(inplace=True)
            
            # Обработка пропусков
            if df['temperature_day'].isnull().any():
                df['temperature_day'].fillna(
                    df['temperature_day'].mean(),
                    inplace=True
                )
            
            self.df = df
            self.data_preview.load_data(df.reset_index())
            self.split_data_btn.setEnabled(True)
            
            self.logger.info("Данные подготовлены")
            self.info_label.setText("Данные подготовлены к анализу")
            
        except Exception as e:
            self.logger.error(f"Ошибка при подготовке данных: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при подготовке данных: {str(e)}")

    def split_data(self) -> None:
        """Разделение данных на обучающую и тестовую выборки."""
        if self.df is None:
            return
            
        try:
            test_size = self.test_size_spin.value() / 100
            split_idx = int(len(self.df) * (1 - test_size))
            
            self.train_data = self.df.iloc[:split_idx].copy()
            self.test_data = self.df.iloc[split_idx:].copy()
            
            self.data_preview.load_data(self.train_data.reset_index())
            
            self.train_btn.setEnabled(True)
            self.tune_btn.setEnabled(True)
            
            self.info_label.setText(
                f"Данные разделены:\n"
                f"Обучающая выборка: {len(self.train_data)} записей\n"
                f"Тестовая выборка: {len(self.test_data)} записей"
            )
            
            self.logger.info(
                f"Данные разделены: обучающая - {len(self.train_data)}, "
                f"тестовая - {len(self.test_data)}"
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка при разделении данных: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при разделении данных: {str(e)}")

    def get_current_params(self) -> Tuple[tuple, tuple]:
        """
        Получение текущих параметров модели.
        
        Returns:
            Tuple[tuple, tuple]: Кортеж с параметрами (order, seasonal_order)
        """
        order = (
            self.p_spin.value(),
            self.d_spin.value(),
            self.q_spin.value()
        )
        
        seasonal_order = (
            self.P_spin.value(),
            self.D_spin.value(),
            self.Q_spin.value(),
            self.s_spin.value()
        )
        
        return order, seasonal_order

    def train_model(self) -> None:
        """Обучение модели с текущими параметрами."""
        if self.train_data is None or self.test_data is None:
            QMessageBox.warning(self, "Ошибка", "Сначала разделите данные")
            return
            
        try:
            order, seasonal_order = self.get_current_params()
            
            self.logger.info(f"Начало обучения модели с параметрами: {order}, {seasonal_order}")
            
            # Создаем и обучаем модель
            model = SARIMAX(
                self.train_data['temperature_day'],
                order=order,
                seasonal_order=seasonal_order
            )
            
            fitted_model = model.fit(disp=False)
            
            # Делаем прогноз
            predictions = fitted_model.get_forecast(len(self.test_data))
            predicted_values = predictions.predicted_mean
            
            # Оцениваем качество
            mse = mean_squared_error(self.test_data['temperature_day'], predicted_values)
            r2 = r2_score(self.test_data['temperature_day'], predicted_values)
            
            # Сохраняем результаты
            result = {
                'order': order,
                'seasonal_order': seasonal_order,
                'mse': mse,
                'r2': r2,
                'model': fitted_model,
                'predictions': predicted_values,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.model_results.append(result)
            
            # Обновляем лучшую модель если нужно
            if mse < self.best_score:
                self.best_score = mse
                self.best_model = fitted_model
                self.save_model_btn.setEnabled(True)
            
            # Строим график
            self.plot_predictions(predicted_values)
            
            # Сохраняем результаты
            self.save_results()
            
            self.info_label.setText(
                f"Модель обучена:\nMSE: {mse:.4f}\nR2: {r2:.4f}"
            )
            
            self.logger.info(f"Модель обучена: MSE={mse:.4f}, R2={r2:.4f}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при обучении модели: {str(e)}")

    def tune_hyperparameters(self) -> None:
        """Подбор оптимальных гиперпараметров."""
        if self.train_data is None or self.test_data is None:
            QMessageBox.warning(self, "Ошибка", "Сначала разделите данные")
            return
            
        try:
            # Параметры для перебора
            params_grid = {
                'p': range(0, 3),
                'd': range(0, 2),
                'q': range(0, 3),
                'P': range(0, 2),
                'D': range(0, 2),
                'Q': range(0, 2),
                's': [12]  # Фиксированная сезонность
            }
            
            total_combinations = (
                len(params_grid['p']) * len(params_grid['d']) * len(params_grid['q']) *
                len(params_grid['P']) * len(params_grid['D']) * len(params_grid['Q']) *
                len(params_grid['s'])
            )
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(total_combinations)
            current_combination = 0
            
            self.logger.info(f"Начало подбора параметров. Всего комбинаций: {total_combinations}")
            
            for p in params_grid['p']:
                for d in params_grid['d']:
                    for q in params_grid['q']:
                        for P in params_grid['P']:
                            for D in params_grid['D']:
                                for Q in params_grid['Q']:
                                    for s in params_grid['s']:
                                        order = (p, d, q)
                                        seasonal_order = (P, D, Q, s)
                                        
                                        try:
                                            # Создаем и обучаем модель
                                            model = SARIMAX(
                                                self.train_data['temperature_day'],
                                                order=order,
                                                seasonal_order=seasonal_order
                                            )
                                            
                                            fitted_model = model.fit(disp=False)
                                            
                                            # Делаем прогноз
                                            predictions = fitted_model.get_forecast(len(self.test_data))
                                            predicted_values = predictions.predicted_mean
                                            
                                            # Оцениваем качество
                                            mse = mean_squared_error(
                                                self.test_data['temperature_day'],
                                                predicted_values
                                            )
                                            r2 = r2_score(
                                                self.test_data['temperature_day'],
                                                predicted_values
                                            )
                                            
                                            # Сохраняем результаты
                                            result = {
                                                'order': order,
                                                'seasonal_order': seasonal_order,
                                                'mse': mse,
                                                'r2': r2,
                                                'model': fitted_model,
                                                'predictions': predicted_values,
                                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            }
                                            
                                            self.model_results.append(result)
                                            
                                            # Обновляем лучшую модель
                                            if mse < self.best_score:
                                                self.best_score = mse
                                                self.best_model = fitted_model
                                                self.save_model_btn.setEnabled(True)
                                                
                                                # Обновляем значения в интерфейсе
                                                self.p_spin.setValue(p)
                                                self.d_spin.setValue(d)
                                                self.q_spin.setValue(q)
                                                self.P_spin.setValue(P)
                                                self.D_spin.setValue(D)
                                                self.Q_spin.setValue(Q)
                                                self.s_spin.setValue(s)
                                                
                                        except Exception as e:
                                            self.logger.warning(
                                                f"Пропуск комбинации {order}, {seasonal_order}: {str(e)}"
                                            )
                                            
                                        current_combination += 1
                                        self.progress_bar.setValue(current_combination)
                                        
            # Сохраняем подробные результаты
            self.save_detailed_results()
            
            # Строим график для лучшей модели
            best_result = min(self.model_results, key=lambda x: x['mse'])
            self.plot_predictions(best_result['predictions'])
            
            self.progress_bar.setVisible(False)
            
            self.info_label.setText(
                f"Лучшая модель:\n"
                f"Order: {best_result['order']}\n"
                f"Seasonal Order: {best_result['seasonal_order']}\n"
                f"MSE: {best_result['mse']:.4f}\n"
                f"R2: {best_result['r2']:.4f}"
            )
            
            self.logger.info("Подбор параметров завершен")
            
        except Exception as e:
            self.logger.error(f"Ошибка при подборе параметров: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при подборе параметров: {str(e)}")
            self.progress_bar.setVisible(False)

    def plot_predictions(self, predictions: np.ndarray) -> None:
        """
        Построение графика прогноза.
        
        Args:
            predictions: Массив предсказанных значений
        """
        try:
            # Очищаем предыдущий график
            for i in reversed(range(self.right_panel.layout().count())): 
                widget = self.right_panel.layout().itemAt(i).widget()
                if isinstance(widget, FigureCanvas):
                    widget.deleteLater()
            
            # Создаем новый график
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Строим фактические значения
            ax.plot(
                self.test_data.index,
                self.test_data['temperature_day'],
                label='Фактические значения',
                color='blue'
            )
            
            # Строим предсказанные значения
            ax.plot(
                self.test_data.index,
                predictions,
                label='Прогноз',
                color='red',
                linestyle='--'
            )
            
            ax.set_title('Прогноз температуры')
            ax.set_xlabel('Дата')
            ax.set_ylabel('Температура')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Создаем виджет с графиком
            canvas = FigureCanvas(fig)
            self.right_panel.layout().addWidget(canvas)
            
            # Сохраняем график
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'ml_plots/prediction_plot_{timestamp}.png')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Ошибка при построении графика: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при построении графика: {str(e)}")

    def save_model(self) -> None:
        """Сохранение лучшей модели."""
        if self.best_model is None:
            QMessageBox.warning(self, "Ошибка", "Нет обученной модели для сохранения")
            return
            
        try:
            # Получаем путь для сохранения
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'ml_models/sarima_model_{timestamp}.pkl'
            
            # Сохраняем модель
            with open(filename, 'wb') as f:
                pickle.dump(self.best_model, f)
            
            # Сохраняем метаданные
            metadata = {
                'timestamp': timestamp,
                'mse': self.best_score,
                'order': self.get_current_params()[0],
                'seasonal_order': self.get_current_params()[1],
                'training_size': len(self.train_data),
                'test_size': len(self.test_data)
            }
            
            with open(f'ml_models/sarima_model_{timestamp}_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)
            
            self.logger.info(f"Модель сохранена в {filename}")
            QMessageBox.information(self, "Успех", f"Модель сохранена в {filename}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении модели: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при сохранении модели: {str(e)}")

    def load_model(self) -> None:
        """Загрузка сохраненной модели."""
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Загрузить модель",
                "ml_models",
                "Pickle files (*.pkl)"
            )
            
            if filename:
                with open(filename, 'rb') as f:
                    self.best_model = pickle.load(f)
                
                # Попробуем загрузить метаданные
                metadata_file = filename.replace('.pkl', '_metadata.json')
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        self.info_label.setText(
                            f"Загружена модель:\n"
                            f"MSE: {metadata['mse']:.4f}\n"
                            f"Order: {metadata['order']}\n"
                            f"Seasonal Order: {metadata['seasonal_order']}"
                        )
                
                self.logger.info(f"Загружена модель из {filename}")
                QMessageBox.information(self, "Успех", "Модель успешно загружена")
                
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при загрузке модели: {str(e)}")

    def save_detailed_results(self) -> None:
        """
        Сохранение подробных результатов обучения и тестирования моделей.
        Создает текстовый файл с описанием влияния различных гиперпараметров
        на качество модели и общей статистикой по всем экспериментам.
        """
        try:
            # Создаем директорию для результатов, если её нет
            os.makedirs('ml_results', exist_ok=True)
            
            # Формируем имя файла с текущей датой и временем
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'ml_results/training_results_{timestamp}.txt'
            
            with open(filename, 'w', encoding='utf-8') as f:
                # Заголовок отчета
                f.write("ОТЧЕТ О ПОДБОРЕ ПАРАМЕТРОВ МОДЕЛИ SARIMA\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("1. ИНФОРМАЦИЯ О ДАННЫХ\n")
                f.write("-" * 60 + "\n")
                f.write(f"Размер обучающей выборки: {len(self.train_data)}\n")
                f.write(f"Размер тестовой выборки: {len(self.test_data)}\n")
                f.write(f"Период данных: с {self.train_data.index.min()} по {self.test_data.index.max()}\n\n")
                
                # Результаты всех моделей
                f.write("2. РЕЗУЛЬТАТЫ ВСЕХ МОДЕЛЕЙ\n")
                f.write("-" * 60 + "\n")
                
                # Сортируем результаты по MSE
                sorted_results = sorted(self.model_results, key=lambda x: x['mse'])
                
                for i, result in enumerate(sorted_results, 1):
                    f.write(f"\nМодель {i}:\n")
                    f.write(f"Время обучения: {result['timestamp']}\n")
                    f.write(f"Параметры (p,d,q): {result['order']}\n")
                    f.write(f"Сезонные параметры (P,D,Q,s): {result['seasonal_order']}\n")
                    f.write(f"MSE: {result['mse']:.4f}\n")
                    f.write(f"R2: {result['r2']:.4f}\n")
                
                # Анализ влияния параметров
                f.write("\n3. АНАЛИЗ ВЛИЯНИЯ ПАРАМЕТРОВ\n")
                f.write("-" * 60 + "\n")
                
                # Анализ несезонных параметров
                param_names = ['p', 'd', 'q']
                for param_idx, param_name in enumerate(param_names):
                    f.write(f"\nВлияние параметра {param_name.upper()}:\n")
                    param_values = sorted(set(r['order'][param_idx] for r in self.model_results))
                    
                    for value in param_values:
                        results = [r for r in self.model_results if r['order'][param_idx] == value]
                        avg_mse = np.mean([r['mse'] for r in results])
                        avg_r2 = np.mean([r['r2'] for r in results])
                        std_mse = np.std([r['mse'] for r in results])
                        
                        f.write(f"\n{param_name} = {value}:\n")
                        f.write(f"  Количество моделей: {len(results)}\n")
                        f.write(f"  Средняя MSE: {avg_mse:.4f} (±{std_mse:.4f})\n")
                        f.write(f"  Средний R2: {avg_r2:.4f}\n")
                
                # Анализ сезонных параметров
                seasonal_param_names = ['P', 'D', 'Q', 's']
                for param_idx, param_name in enumerate(seasonal_param_names):
                    f.write(f"\nВлияние сезонного параметра {param_name.upper()}:\n")
                    param_values = sorted(set(r['seasonal_order'][param_idx] for r in self.model_results))
                    
                    for value in param_values:
                        results = [r for r in self.model_results if r['seasonal_order'][param_idx] == value]
                        avg_mse = np.mean([r['mse'] for r in results])
                        avg_r2 = np.mean([r['r2'] for r in results])
                        std_mse = np.std([r['mse'] for r in results])
                        
                        f.write(f"\n{param_name} = {value}:\n")
                        f.write(f"  Количество моделей: {len(results)}\n")
                        f.write(f"  Средняя MSE: {avg_mse:.4f} (±{std_mse:.4f})\n")
                        f.write(f"  Средний R2: {avg_r2:.4f}\n")
                
                # Общие выводы
                f.write("\n4. ОБЩИЕ ВЫВОДЫ\n")
                f.write("-" * 60 + "\n")
                
                # Лучшая модель
                best_model = min(self.model_results, key=lambda x: x['mse'])
                f.write("\nЛучшая модель:\n")
                f.write(f"Параметры (p,d,q): {best_model['order']}\n")
                f.write(f"Сезонные параметры (P,D,Q,s): {best_model['seasonal_order']}\n")
                f.write(f"MSE: {best_model['mse']:.4f}\n")
                f.write(f"R2: {best_model['r2']:.4f}\n")
                
                # Статистика по всем моделям
                all_mse = [r['mse'] for r in self.model_results]
                all_r2 = [r['r2'] for r in self.model_results]
                
                f.write("\nСтатистика по всем моделям:\n")
                f.write(f"Количество обученных моделей: {len(self.model_results)}\n")
                f.write(f"Средняя MSE: {np.mean(all_mse):.4f}\n")
                f.write(f"Стандартное отклонение MSE: {np.std(all_mse):.4f}\n")
                f.write(f"Минимальная MSE: {np.min(all_mse):.4f}\n")
                f.write(f"Максимальная MSE: {np.max(all_mse):.4f}\n")
                f.write(f"Средний R2: {np.mean(all_r2):.4f}\n")
                f.write(f"Минимальный R2: {np.min(all_r2):.4f}\n")
                f.write(f"Максимальный R2: {np.max(all_r2):.4f}\n")
                
                # Рекомендации
                f.write("\n5. РЕКОМЕНДАЦИИ\n")
                f.write("-" * 60 + "\n")
                f.write("На основе проведенного анализа:\n")
                
                # Находим наиболее стабильные параметры
                for param_idx, param_name in enumerate(param_names + seasonal_param_names):
                    if param_name in param_names:
                        results_by_param = [(value, [r['mse'] for r in self.model_results if r['order'][param_idx] == value]) 
                                          for value in sorted(set(r['order'][param_idx] for r in self.model_results))]
                    else:
                        seasonal_idx = param_idx - len(param_names)
                        results_by_param = [(value, [r['mse'] for r in self.model_results if r['seasonal_order'][seasonal_idx] == value]) 
                                          for value in sorted(set(r['seasonal_order'][seasonal_idx] for r in self.model_results))]
                    
                    best_value = min(results_by_param, key=lambda x: np.mean(x[1]))
                    f.write(f"- Рекомендуемое значение {param_name}: {best_value[0]}\n")

            self.logger.info(f"Подробные результаты сохранены в {filename}")
            QMessageBox.information(
                self,
                "Результаты сохранены",
                f"Подробный отчет сохранен в:\n{filename}"
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении результатов: {str(e)}")
            QMessageBox.warning(
                self,
                "Ошибка",
                f"Ошибка при сохранении результатов: {str(e)}"
            )

if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = MLTab()
    window.show()
    sys.exit(app.exec())