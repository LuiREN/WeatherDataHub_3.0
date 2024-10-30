from typing import Optional, Tuple
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QGroupBox, QMessageBox, QSpinBox
)
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os
from datetime import datetime
from PyQt6.QtCore import Qt
import logging
from optimized_table import OptimizedTableWidget

class MLTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.df: Optional[pd.DataFrame] = None
        self.current_file: Optional[str] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.best_model = None
        self.best_score = float('inf')
        self.model_results = []

        # Настройка логирования
        self.setup_logging()
        
        # Инициализация интерфейса
        self.init_ui()

    def setup_logging(self) -> None:
        """Настройка системы логирования."""
        self.logger = logging.getLogger('MLTab')
        self.logger.setLevel(logging.INFO)
        
        # Создаем обработчик для записи в файл
        fh = logging.FileHandler('ml_training.log')
        fh.setLevel(logging.INFO)
        
        # Создаем форматтер
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        
        # Добавляем обработчик к логгеру
        self.logger.addHandler(fh)

    def init_ui(self) -> None:
        """Инициализация пользовательского интерфейса вкладки."""
        main_layout = QHBoxLayout()
        
        # Создаем левую панель с контролами
        left_panel = self.create_left_panel()
        
        # Создаем правую панель для отображения результатов
        right_panel = self.create_right_panel()
        
        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(right_panel, stretch=2)
        
        self.setLayout(main_layout)

    def create_left_panel(self) -> QWidget:
        """Создание левой панели с элементами управления."""
        left_panel = QWidget()
        layout = QVBoxLayout()
        
        # Группа подготовки данных
        data_group = QGroupBox("Подготовка данных")
        data_layout = QVBoxLayout()
        
        # Выбор размера тестовой выборки
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Размер тестовой выборки:"))
        self.test_size_spin = QSpinBox()
        self.test_size_spin.setRange(10, 40)
        self.test_size_spin.setValue(20)
        self.test_size_spin.setSuffix("%")
        size_layout.addWidget(self.test_size_spin)
        data_layout.addLayout(size_layout)
        
        # Кнопки для подготовки данных
        self.prepare_data_btn = QPushButton("Подготовить данные")
        self.prepare_data_btn.setEnabled(False)
        self.prepare_data_btn.clicked.connect(self.prepare_data)
        data_layout.addWidget(self.prepare_data_btn)
        
        self.split_data_btn = QPushButton("Разделить данные")
        self.split_data_btn.setEnabled(False)
        self.split_data_btn.clicked.connect(self.split_data)
        data_layout.addWidget(self.split_data_btn)

        # Группа настройки SARIMA
        sarima_group = QGroupBox("Настройка SARIMA")
        sarima_layout = QVBoxLayout()
        
        # Параметры p, d, q
        pdq_layout = QHBoxLayout()
        pdq_layout.addWidget(QLabel("p:"))
        self.p_spin = QSpinBox()
        self.p_spin.setRange(0, 5)
        self.p_spin.setValue(1)
        pdq_layout.addWidget(self.p_spin)
        
        pdq_layout.addWidget(QLabel("d:"))
        self.d_spin = QSpinBox()
        self.d_spin.setRange(0, 2)
        self.d_spin.setValue(1)
        pdq_layout.addWidget(self.d_spin)
        
        pdq_layout.addWidget(QLabel("q:"))
        self.q_spin = QSpinBox()
        self.q_spin.setRange(0, 5)
        self.q_spin.setValue(1)
        pdq_layout.addWidget(self.q_spin)
        
        sarima_layout.addLayout(pdq_layout)
        
        # Кнопка обучения
        self.train_btn = QPushButton("Обучить модель")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self.train_model)
        sarima_layout.addWidget(self.train_btn)

        # Сезонные параметры P, D, Q, s
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
        self.s_spin.setValue(12)  # 12 месяцев по умолчанию
        seasonal_layout.addWidget(self.s_spin)
        
        sarima_layout.addLayout(seasonal_layout)

        # Кнопка подбора гиперпараметров
        self.tune_btn = QPushButton("Подобрать гиперпараметры")
        self.tune_btn.setEnabled(False)
        self.tune_btn.clicked.connect(self.tune_hyperparameters)
        sarima_layout.addWidget(self.tune_btn)
        
        sarima_group.setLayout(sarima_layout)
        layout.addWidget(sarima_group)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        layout.addStretch()
        left_panel.setLayout(layout)
        return left_panel

    def create_right_panel(self) -> QWidget:
        """Создание правой панели для отображения результатов."""
        right_panel = QWidget()
        layout = QVBoxLayout()
        
        self.info_label = QLabel("Загрузите данные для начала работы")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)
        
        self.data_preview = OptimizedTableWidget()
        layout.addWidget(self.data_preview)
        
        right_panel.setLayout(layout)
        return right_panel

    def load_data(self, df: pd.DataFrame, file_path: Optional[str] = None) -> None:
        """Загрузка данных для анализа и прогнозирования."""
        try:
            self.df = df.copy()
            self.current_file = file_path
            
            if self.validate_data(df):
                self.data_preview.load_data(df)
                self.prepare_data_btn.setEnabled(True)
                self.info_label.setText(f"Данные загружены: {len(df)} записей")
                self.logger.info(f"Загружены данные: {len(df)} записей")
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при загрузке данных: {str(e)}")

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Проверка корректности данных для анализа."""
        if df is None:
            self.logger.warning("Нет данных для проверки")
            return False
            
        # Проверяем наличие необходимых столбцов
        required_columns = {'date', 'temperature_day'} 
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            self.logger.warning(f"Отсутствуют столбцы: {missing_columns}")
            QMessageBox.warning(
                self,
                "Ошибка данных",
                f"Отсутствуют необходимые столбцы: {', '.join(missing_columns)}\n"
                "Выполните предобработку данных на вкладке анализа."
            )
            return False
        
        return True

    def prepare_data(self) -> None:
        """Подготовка данных для анализа временных рядов."""
        if self.df is None:
            return
            
        try:
            # Создаем копию данных
            df = self.df.copy()
            
            # Преобразуем дату в индекс
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Сортируем по дате
            df.sort_index(inplace=True)
            
            # Проверяем на пропущенные значения
            if df['temperature_day'].isnull().any():
                # Заполняем пропуски средним значением
                df['temperature_day'].fillna(df['temperature_day'].mean(), inplace=True)
                self.logger.info("Заполнены пропущенные значения температуры")
            
            self.df = df
            self.data_preview.load_data(df.reset_index())
            self.split_data_btn.setEnabled(True)
            self.info_label.setText("Данные подготовлены к анализу")
            self.logger.info("Данные успешно подготовлены")
            
        except Exception as e:
            self.logger.error(f"Ошибка при подготовке данных: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при подготовке данных: {str(e)}")

    def split_data(self) -> None:
        """Разделение данных на обучающую и тестовую выборки."""
        if self.df is None:
            return
            
        try:
            # Получаем размер тестовой выборки
            test_size = self.test_size_spin.value() / 100
            
            # Определяем точку разделения
            split_idx = int(len(self.df) * (1 - test_size))
            
            # Разделяем данные
            self.train_data = self.df.iloc[:split_idx].copy()
            self.test_data = self.df.iloc[split_idx:].copy()
            
            # Отображаем обучающую выборку
            self.data_preview.load_data(self.train_data.reset_index())
            
            # Обновляем информацию
            self.info_label.setText(
                f"Данные разделены:\n"
                f"Обучающая выборка: {len(self.train_data)} записей\n"
                f"Тестовая выборка: {len(self.test_data)} записей"
            )
            
            self.logger.info(
                f"Данные разделены на выборки: "
                f"обучающая - {len(self.train_data)}, "
                f"тестовая - {len(self.test_data)}"
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка при разделении данных: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при разделении данных: {str(e)}")

    def train_model(self) -> None:
        """Обучение модели SARIMA с текущими параметрами."""
        if self.train_data is None or self.test_data is None:
            QMessageBox.warning(self, "Ошибка", "Сначала разделите данные")
            return
            
        try:
            # Получаем параметры модели
            order = (
                self.p_spin.value(),
                self.d_spin.value(),
                self.q_spin.value()
            )
            
            # Создаем и обучаем модель
            model = SARIMAX(
                self.train_data['temperature_day'],
                order=order,
                seasonal_order=(0, 0, 0, 0)  # Пока без сезонности
            )
            
            self.logger.info(f"Начало обучения модели с параметрами {order}")
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
                'mse': mse,
                'r2': r2,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.model_results.append(result)
            
            # Обновляем лучшую модель если нужно
            if mse < self.best_score:
                self.best_score = mse
                self.best_model = fitted_model
                self.logger.info(f"Новая лучшая модель: MSE={mse:.4f}, R2={r2:.4f}")
            
            # Строим график
            self.plot_predictions(predicted_values)
            
            # Выводим результаты
            self.info_label.setText(
                f"Модель обучена:\n"
                f"MSE: {mse:.4f}\n"
                f"R2: {r2:.4f}"
            )
            
            # Создаем директорию для результатов если её нет
            os.makedirs('ml_results', exist_ok=True)
            
            # Сохраняем результаты в файл
            self.save_results()
            
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при обучении модели: {str(e)}")

    def plot_predictions(self, predictions) -> None:
        """
        Построение графика прогноза.
        
        Args:
            predictions: Предсказанные значения
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
                label='Фактические значения'
            )
            
            # Строим предсказанные значения
            ax.plot(
                self.test_data.index,
                predictions,
                label='Прогноз',
                linestyle='--'
            )
            
            ax.set_title('Прогноз температуры')
            ax.set_xlabel('Дата')
            ax.set_ylabel('Температура')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Создаем виджет с графиком
            canvas = FigureCanvas(fig)
            self.right_panel.layout().addWidget(canvas)
            
            # Сохраняем график
            plt.savefig('ml_results/prediction_plot.png')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Ошибка при построении графика: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при построении графика: {str(e)}")

    def save_results(self) -> None:
        """Сохранение результатов обучения в файл."""
        try:
            with open('ml_results/training_results.txt', 'w', encoding='utf-8') as f:
                f.write("Результаты обучения моделей SARIMA\n")
                f.write("=" * 50 + "\n\n")
                
                for result in self.model_results:
                    f.write(f"Время: {result['timestamp']}\n")
                    f.write(f"Параметры (p,d,q): {result['order']}\n")
                    f.write(f"MSE: {result['mse']:.4f}\n")
                    f.write(f"R2: {result['r2']:.4f}\n")
                    f.write("-" * 50 + "\n\n")
                    
            self.logger.info("Результаты сохранены в файл")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении результатов: {str(e)}")

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
            
            self.info_label.setText(
                f"Данные разделены:\n"
                f"Обучающая выборка: {len(self.train_data)} записей\n"
                f"Тестовая выборка: {len(self.test_data)} записей"
            )
            
            # Активируем кнопку обучения
            self.train_btn.setEnabled(True)
            
            self.logger.info(
                f"Данные разделены на выборки: "
                f"обучающая - {len(self.train_data)}, "
                f"тестовая - {len(self.test_data)}"
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка при разделении данных: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при разделении данных: {str(e)}")