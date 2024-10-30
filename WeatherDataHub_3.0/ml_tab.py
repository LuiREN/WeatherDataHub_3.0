from typing import Optional, Tuple
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QGroupBox, QMessageBox, QSpinBox
)
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