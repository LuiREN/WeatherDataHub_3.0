def get_styles():
    return """
    QMainWindow, QWidget {
        background-color: #f5f5f5;
        color: #333333;
    }

    QPushButton {
        background-color: #8A2BE2;
        color: white;
        padding: 8px 16px;
        font-size: 14px;
        border: none;
        border-radius: 10px;
        min-width: 180px;
        font-family: Arial, sans-serif;
    }

    QPushButton:hover {
        background-color: #9932CC;
    }

    QPushButton:disabled {
        background-color: #cccccc;
        color: #666666;
    }

    QTableWidget {
        background-color: white;
        gridline-color: #e0e0e0;
        border: 1px solid #e0e0e0;
        color: #333333;
    }

    QHeaderView::section {
        background-color: #f0f0f0;
        padding: 4px;
        border: 1px solid #e0e0e0;
        font-weight: bold;
        font-family: Arial, sans-serif;
        color: #333333;
    }

    QLabel {
        color: #333333;
        font-size: 14px;
        font-family: Arial, sans-serif;
    }

    QLineEdit {
        border: 1px solid #c0c0c0;
        border-radius: 4px;
        padding: 5px;
        font-family: Arial, sans-serif;
        background-color: white;
        color: #333333;
    }

    QProgressBar {
        border: 1px solid #c0c0c0;
        border-radius: 5px;
        text-align: center;
        background-color: #f0f0f0;
    }

    QProgressBar::chunk {
        background-color: #8A2BE2;
    }

    QTabWidget::pane {
        border: 1px solid #c0c0c0;
        background-color: #f5f5f5;
    }

    QTabBar::tab {
        background-color: #e0e0e0;
        color: #333333;
        padding: 8px 16px;
        border: 1px solid #c0c0c0;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }

    QTabBar::tab:selected {
        background-color: #f5f5f5;
        border-bottom: none;
    }

    QGroupBox {
        border: 1px solid #c0c0c0;
        border-radius: 4px;
        margin-top: 1em;
        padding-top: 1em;
        background-color: #ffffff;
    }

    QGroupBox::title {
        color: #333333;
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 3px;
        background-color: transparent;
    }

    QComboBox {
        background-color: white;
        border: 1px solid #c0c0c0;
        border-radius: 4px;
        padding: 5px;
        color: #333333;
    }

    QComboBox::drop-down {
        border: none;
    }

    QComboBox::down-arrow {
        image: none;
        border: none;
    }

    QMessageBox {
        background-color: #f5f5f5;
        color: #333333;
    }

    QMessageBox QPushButton {
        min-width: 100px;
    }

    QDialog {
        background-color: #f5f5f5;
        color: #333333;
    }

    /* Стили для скролл-баров */
    QScrollBar:vertical {
        border: none;
        background-color: #f0f0f0;
        width: 10px;
        margin: 0;
    }

    QScrollBar::handle:vertical {
        background-color: #c0c0c0;
        min-height: 20px;
        border-radius: 5px;
    }

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        border: none;
        background: none;
    }

    QScrollBar:horizontal {
        border: none;
        background-color: #f0f0f0;
        height: 10px;
        margin: 0;
    }

    QScrollBar::handle:horizontal {
        background-color: #c0c0c0;
        min-width: 20px;
        border-radius: 5px;
    }

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        border: none;
        background: none;
    }
    """