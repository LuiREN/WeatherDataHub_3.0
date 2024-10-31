from typing import Optional, Dict, List, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QGroupBox, QMessageBox, QSpinBox,
    QFrame, QFileDialog, QProgressBar,QTableWidget,QTableWidgetItem
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import pandas as pd
import logging
from ml_model import WeatherModel
from ml_data_handler import DataHandler
from ml_visualization import ModelVisualizer
import os
from datetime import datetime

class MLTab(QWidget):
    """
    Вкладка машинного обучения для прогнозирования погоды.
    
    Attributes:
        data_handler: Обработчик данных
        model: Модель прогнозирования
        visualizer: Визуализатор результатов
        logger: Логгер для записи операций
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Инициализация вкладки машинного обучения."""
        super().__init__(parent)
        
        # Инициализация компонентов
        self.data_handler = DataHandler()
        self.model = WeatherModel()
        self.visualizer = ModelVisualizer()
        
        # Инициализация данных
        self.train_data = None
        self.test_data = None
        self.current_predictions = None
        
        # Настройка интерфейса и логирования
        self.setup_logger()
        self.init_ui()

    def setup_logger(self) -> None:
        """Настройка системы логирования."""
        self.logger = logging.getLogger('MLInterface')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('ml_interface.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def init_ui(self) -> None:
        """Инициализация пользовательского интерфейса."""
        layout = QHBoxLayout(self)
        
        # Создание панелей
        left_panel = self.create_left_panel()
        right_panel = self.create_right_panel()
        
        # Добавление панелей в главный layout
        layout.addWidget(left_panel, stretch=1)  # 40% ширины
        layout.addWidget(right_panel, stretch=2)  # 60% ширины

    def create_left_panel(self) -> QFrame:
        """
        Создание левой панели с элементами управления.
        
        Returns:
            QFrame: Виджет левой панели
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Группа подготовки данных
        data_group = self.create_data_group()
        layout.addWidget(data_group)
        
        # Группа параметров модели
        model_group = self.create_model_group()
        layout.addWidget(model_group)
        
        # Группа обучения
        training_group = self.create_training_group()
        layout.addWidget(training_group)
        
        # Индикатор прогресса
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        return panel

    def create_data_group(self) -> QGroupBox:
        """
        Создание группы элементов для работы с данными.
        
        Returns:
            QGroupBox: Группа элементов для данных
        """
        group = QGroupBox("Подготовка данных")
        layout = QVBoxLayout(group)
        
        # Размер тестовой выборки
        test_size_layout = QHBoxLayout()
        test_size_layout.addWidget(QLabel("Размер тестовой выборки:"))
        self.test_size_spin = QSpinBox()
        self.test_size_spin.setRange(10, 40)
        self.test_size_spin.setValue(20)
        self.test_size_spin.setSuffix("%")
        test_size_layout.addWidget(self.test_size_spin)
        layout.addLayout(test_size_layout)
        
        # Кнопки для работы с данными
        self.prepare_btn = QPushButton("Подготовить данные")
        self.prepare_btn.clicked.connect(self.prepare_data)
        self.prepare_btn.setEnabled(False)
        layout.addWidget(self.prepare_btn)
        
        self.split_btn = QPushButton("Разделить данные")
        self.split_btn.clicked.connect(self.split_data)
        self.split_btn.setEnabled(False)
        layout.addWidget(self.split_btn)
        
        return group

    def create_model_group(self) -> QGroupBox:
        """
        Создание группы элементов для настройки модели.
        
        Returns:
            QGroupBox: Группа элементов настройки модели
        """
        group = QGroupBox("Параметры SARIMA")
        layout = QVBoxLayout(group)
        
        # Параметры p, d, q
        pdq_layout = QHBoxLayout()
        
        # p parameter
        pdq_layout.addWidget(QLabel("p:"))
        self.p_spin = QSpinBox()
        self.p_spin.setRange(0, 3)
        self.p_spin.setValue(1)
        pdq_layout.addWidget(self.p_spin)
        
        # d parameter
        pdq_layout.addWidget(QLabel("d:"))
        self.d_spin = QSpinBox()
        self.d_spin.setRange(0, 2)
        self.d_spin.setValue(1)
        pdq_layout.addWidget(self.d_spin)
        
        # q parameter
        pdq_layout.addWidget(QLabel("q:"))
        self.q_spin = QSpinBox()
        self.q_spin.setRange(0, 3)
        self.q_spin.setValue(1)
        pdq_layout.addWidget(self.q_spin)
        
        layout.addLayout(pdq_layout)
        
        # Сезонные параметры P, D, Q, s
        seasonal_layout = QHBoxLayout()
        
        seasonal_layout.addWidget(QLabel("P:"))
        self.P_spin = QSpinBox()
        self.P_spin.setRange(0, 2)
        self.P_spin.setValue(1)
        seasonal_layout.addWidget(self.P_spin)
        
        seasonal_layout.addWidget(QLabel("D:"))
        self.D_spin = QSpinBox()
        self.D_spin.setRange(0, 2)
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
        
        return group

    def create_training_group(self) -> QGroupBox:
        """
        Создание группы элементов для обучения модели.
        
        Returns:
            QGroupBox: Группа элементов обучения
        """
        group = QGroupBox("Обучение и оценка")
        layout = QVBoxLayout(group)
        
        self.train_btn = QPushButton("Обучить модель")
        self.train_btn.clicked.connect(self.train_model)
        self.train_btn.setEnabled(False)
        layout.addWidget(self.train_btn)
        
        self.tune_btn = QPushButton("Подобрать параметры")
        self.tune_btn.clicked.connect(self.tune_parameters)
        self.tune_btn.setEnabled(False)
        layout.addWidget(self.tune_btn)
        
        self.save_btn = QPushButton("Сохранить модель")
        self.save_btn.clicked.connect(self.save_model)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)
        
        return group

    def create_right_panel(self) -> QFrame:
        """Создание правой панели для отображения результатов."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
    
        # Информационная метка
        self.info_label = QLabel("Загрузите данные для начала работы")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.info_label)
    
        # Таблица для отображения результатов
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table)
    
        # Область для графиков
        self.plot_frame = QFrame()
        self.plot_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        self.plot_frame.setMinimumHeight(400)
        layout.addWidget(self.plot_frame)
    
        return panel

    def update_results_table(self, results: Dict[str, Any]) -> None:
        """Обновление таблицы результатов."""
        self.results_table.setRowCount(0)
        for param, value in results.items():
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(str(param)))
            self.results_table.setItem(row, 1, QTableWidgetItem(str(value)))

    def load_data(self, df: pd.DataFrame, filename: Optional[str] = None) -> None:
        """
        Загрузка данных для анализа.
    
        Args:
            df: DataFrame с данными
            filename: Имя файла (опционально)
        """
        try:
            self.df = df.copy()
            self.prepare_btn.setEnabled(True)
            self.info_label.setText("Данные загружены. Нажмите 'Подготовить данные' для начала анализа")
            self.logger.info("Данные загружены успешно")
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке данных: {str(e)}")

    def prepare_data(self) -> None:
        """Подготовка данных для анализа."""
        try:
            if not self.df is None:
                # Преобразование даты и установка частоты
                df = self.df.copy()
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.asfreq('D')  # Установка дневной частоты
            
                # Обработка пропущенных значений
                if df['temperature_day'].isnull().any():
                    df['temperature_day'].interpolate(method='time', inplace=True)
            
                self.prepared_data = df
                self.split_btn.setEnabled(True)
            
                # Обновляем таблицу результатов
                results = {
                    'Количество записей': len(df),
                    'Период данных': f"с {df.index.min().date()} по {df.index.max().date()}",
                    'Пропущенные значения': df.isnull().sum().to_dict()
                }
                self.update_results_table(results)
            
                self.info_label.setText("Данные подготовлены успешно")
                self.logger.info("Данные подготовлены")
            
        except Exception as e:
            self.logger.error(f"Ошибка при подготовке данных: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при подготовке данных: {str(e)}")

    def split_data(self) -> None:
        """Разделение данных на обучающую и тестовую выборки."""
        try:
            test_size = self.test_size_spin.value() / 100
            split_idx = int(len(self.prepared_data) * (1 - test_size))
        
            self.train_data = self.prepared_data.iloc[:split_idx].copy()
            self.test_data = self.prepared_data.iloc[split_idx:].copy()
        
            # Обновляем таблицу результатов
            results = {
                'Размер обучающей выборки': len(self.train_data),
                'Размер тестовой выборки': len(self.test_data),
                'Период обучающей выборки': f"с {self.train_data.index.min().date()} по {self.train_data.index.max().date()}",
                'Период тестовой выборки': f"с {self.test_data.index.min().date()} по {self.test_data.index.max().date()}"
            }
            self.update_results_table(results)
        
            self.train_btn.setEnabled(True)
            self.tune_btn.setEnabled(True)
        
            self.info_label.setText("Данные успешно разделены")
            self.logger.info("Данные разделены на выборки")
        
        except Exception as e:
            self.logger.error(f"Ошибка при разделении данных: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при разделении данных: {str(e)}")

    def train_model(self) -> None:
        """Обучение модели с текущими параметрами."""
        try:
            # Получение параметров
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
            
            # Обучение модели
            result = self.model.train(self.train_data, order, seasonal_order)
            
            if result['status'] == 'success':
                # Получение прогноза
                predictions = self.model.predict(len(self.test_data))
                if predictions is not None:
                    self.current_predictions = predictions
                    
                    # Оценка качества
                    metrics = self.model.evaluate_model(self.test_data, predictions)
                    
                    # Визуализация результатов
                    fig = self.visualizer.plot_forecast(
                        self.test_data['temperature_day'],
                        predictions,
                        self.test_data.index
                    )
                    
                    # Обновление интерфейса
                    self.update_plot(fig)
                    self.save_btn.setEnabled(True)
                    self.info_label.setText(
                        f"Модель обучена:\n"
                        f"MSE: {metrics['mse']:.4f}\n"
                        f"R2: {metrics['r2']:.4f}\n"
                        f"MAE: {metrics['mae']:.4f}"
                    )
                    
                    self.logger.info("Модель успешно обучена")
                    
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при обучении модели: {str(e)}")

    def tune_parameters(self) -> None:
        """Подбор оптимальных параметров модели."""
        try:
            self.progress_bar.setVisible(True)
            
            # Подбор параметров
            results = self.model.tune_parameters(self.train_data, self.test_data)
            
            if results['best_result']:
                # Обновление параметров в интерфейсе
                order = results['best_result']['order']
                seasonal_order = results['best_result']['seasonal_order']
                
                self.p_spin.setValue(order[0])
                self.d_spin.setValue(order[1])
                self.q_spin.setValue(order[2])
                self.P_spin.setValue(seasonal_order[0])
                self.D_spin.setValue(seasonal_order[1])
                self.Q_spin.setValue(seasonal_order[2])
                self.s_spin.setValue(seasonal_order[3])
                
                # Визуализация результатов
                fig = self.visualizer.create_results_dashboard(results['best_result'])
                self.update_plot(fig)
                
                # Создание отчета о влиянии параметров
                self.create_parameter_analysis(results['all_results'])
                
                self.info_label.setText(
                    f"Найдены оптимальные параметры:\n"
                    f"Order: {order}\n"
                    f"Seasonal Order: {seasonal_order}\n"
                    f"MSE: {results['best_result']['mse']:.4f}"
                )
                
                self.logger.info("Подбор параметров завершен успешно")
                
            else:
                QMessageBox.warning(
                    self,
                    "Предупреждение",
                    "Не удалось найти оптимальные параметры"
                )
                
            self.progress_bar.setVisible(False)
            
        except Exception as e:
            self.logger.error(f"Ошибка при подборе параметров: {str(e)}")
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Ошибка при подборе параметров: {str(e)}"
            )
            self.progress_bar.setVisible(False)

    def create_parameter_analysis(self, results: List[Dict]) -> None:
        """
        Создание анализа влияния параметров.
        
        Args:
            results: Список результатов с разными параметрами
        """
        try:
            # Создание графиков для каждого параметра
            parameters = ['p', 'd', 'q', 'P', 'D', 'Q']
            for param in parameters:
                fig = self.visualizer.plot_parameter_influence(results, param)
                
                # Сохранение графика
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'parameter_analysis_{param}_{timestamp}.png'
                fig.savefig(os.path.join('plots', filename))
            
            self.logger.info("Создан анализ влияния параметров")
            
        except Exception as e:
            self.logger.error(f"Ошибка при создании анализа параметров: {str(e)}")

    def update_plot(self, fig) -> None:
        """
        Обновление области с графиком.
        
        Args:
            fig: Объект графика matplotlib
        """
        try:
            # Очистка предыдущего графика
            for i in reversed(range(self.plot_frame.layout().count())): 
                self.plot_frame.layout().itemAt(i).widget().deleteLater()
            
            # Создание нового canvas
            canvas = FigureCanvas(fig)
            
            # Добавление canvas в layout
            if not self.plot_frame.layout():
                self.plot_frame.setLayout(QVBoxLayout())
            self.plot_frame.layout().addWidget(canvas)
            
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении графика: {str(e)}")

    def save_model(self) -> None:
        """Сохранение текущей модели."""
        try:
            # Получение пути для сохранения
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить модель",
                "",
                "Model Files (*.pkl)"
            )
            
            if filename:
                # Подготовка метаданных
                metadata = {
                    'order': (
                        self.p_spin.value(),
                        self.d_spin.value(),
                        self.q_spin.value()
                    ),
                    'seasonal_order': (
                        self.P_spin.value(),
                        self.D_spin.value(),
                        self.Q_spin.value(),
                        self.s_spin.value()
                    ),
                    'train_size': len(self.train_data),
                    'test_size': len(self.test_data),
                    'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Сохранение модели
                if self.data_handler.save_model(self.model, filename, metadata):
                    QMessageBox.information(
                        self,
                        "Успех",
                        "Модель успешно сохранена"
                    )
                    self.logger.info(f"Модель сохранена в {filename}")
                else:
                    QMessageBox.warning(
                        self,
                        "Предупреждение",
                        "Не удалось сохранить модель"
                    )
                    
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении модели: {str(e)}")
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Ошибка при сохранении модели: {str(e)}"
            )

    def closeEvent(self, event) -> None:
        """
        Обработка закрытия вкладки.
        
        Args:
            event: Событие закрытия
        """
        try:
            # Сохранение результатов если есть
            if hasattr(self, 'model') and self.model is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.data_handler.save_results(
                    {
                        'order': (
                            self.p_spin.value(),
                            self.d_spin.value(),
                            self.q_spin.value()
                        ),
                        'seasonal_order': (
                            self.P_spin.value(),
                            self.D_spin.value(),
                            self.Q_spin.value(),
                            self.s_spin.value()
                        ),
                        'metrics': self.model.evaluate_model(
                            self.test_data,
                            self.current_predictions
                        ) if self.current_predictions is not None else None
                    },
                    f'final_results_{timestamp}'
                )
            
            self.logger.info("Вкладка закрыта")
            event.accept()
            
        except Exception as e:
            self.logger.error(f"Ошибка при закрытии вкладки: {str(e)}")
            event.accept()


