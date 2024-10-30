from PyQt6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget
from typing import Optional


class DateDataWidget(QWidget):
    """
    Виджет для ввода даты и получения данных.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Инициализирует виджет DateDataWidget.

        Args:
            parent (Optional[QWidget]): Родительский виджет.
        """
        super().__init__(parent)
        self.init_ui()

    def init_ui(self) -> None:
        """
        Инициализирует пользовательский интерфейс виджета.
        """
        self.layout = QVBoxLayout()
        
        date_layout = self.create_date_layout()
        self.layout.addLayout(date_layout)
        
        self.setLayout(self.layout)

    def create_date_layout(self) -> QHBoxLayout:
        """
        Создает горизонтальный layout с полем ввода даты и кнопкой.

        Returns:
            QHBoxLayout: Горизонтальный layout с виджетами для ввода даты.
        """
        date_layout = QHBoxLayout()
        
        self.date_input = QLineEdit()
        self.date_input.setPlaceholderText("ГГГГ-ММ-ДД")
        
        self.get_data_button = QPushButton("Получить данные")
        
        date_layout.addWidget(self.date_input)
        date_layout.addWidget(self.get_data_button)
        
        return date_layout


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    widget = DateDataWidget()
    widget.show()
    sys.exit(app.exec())