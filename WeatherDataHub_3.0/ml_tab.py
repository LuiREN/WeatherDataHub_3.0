from typing import Optional
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt
from optimized_table import OptimizedTableWidget

class MLTab(QWidget):
    """
    Вкладка для машинного обучения и прогнозирования временных рядов.
    
    Attributes:
        df (Optional[pd.DataFrame]): Текущий датафрейм с данными
        current_file (Optional[str]): Путь к текущему файлу
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
        
        # Инициализация интерфейса
        self.init_ui()

    def init_ui(self) -> None:
        """Инициализация пользовательского интерфейса вкладки."""
        main_layout = QHBoxLayout()
        
        # Создаем левую панель с контролами
        left_panel = self.create_left_panel()
        
        # Создаем правую панель для отображения результатов
        right_panel = self.create_right_panel()
        
        main_layout.addWidget(left_panel, stretch=1)  # 30% ширины
        main_layout.addWidget(right_panel, stretch=2)  # 70% ширины
        
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
        data_group = QGroupBox("Подготовка данных")
        data_layout = QVBoxLayout()
        
        self.prepare_data_btn = QPushButton("Подготовить данные")
        self.prepare_data_btn.setEnabled(False)
        data_layout.addWidget(self.prepare_data_btn)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Добавляем растягивающийся спейсер
        layout.addStretch()
        
        left_panel.setLayout(layout)
        return left_panel

    def create_right_panel(self) -> QWidget:
        """
        Создание правой панели для отображения результатов.

        Returns:
            QWidget: Виджет правой панели
        """
        right_panel = QWidget()
        layout = QVBoxLayout()
        
        # Метка для информации
        self.info_label = QLabel("Загрузите данные для начала работы")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)
        
        # Таблица для отображения данных
        self.data_preview = OptimizedTableWidget()
        layout.addWidget(self.data_preview)
        
        right_panel.setLayout(layout)
        return right_panel

    def load_data(self, df: pd.DataFrame, file_path: Optional[str] = None) -> None:
        """
        Загрузка данных для анализа и прогнозирования.

        Args:
            df: DataFrame с данными
            file_path: Путь к файлу данных (опционально)
        """
        try:
            self.df = df.copy()
            self.current_file = file_path
            
            # Отображаем данные
            self.data_preview.load_data(df)
            
            # Активируем кнопки
            self.prepare_data_btn.setEnabled(True)
            
            # Обновляем информацию
            self.info_label.setText(f"Данные загружены: {len(df)} записей")
            
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка при загрузке данных: {str(e)}")