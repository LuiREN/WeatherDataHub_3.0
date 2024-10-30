import logging
import os
from datetime import datetime
from typing import Optional

class WeatherLogger:
    """Класс для логирования операций и ошибок при сборе данных о погоде."""
    
    def __init__(self, base_filename: str):
        """
        Инициализация логгера.
        
        Args:
            base_filename (str): Базовое имя файла для логов (без расширения)
        """
        self.logs_folder = 'weather_logs'
        if not os.path.exists(self.logs_folder):
            os.makedirs(self.logs_folder)
            
        # Создаем имя лог-файла на основе базового имени
        log_filename = os.path.splitext(os.path.basename(base_filename))[0]
        self.log_path = os.path.join(self.logs_folder, f"{log_filename}.log")
        
        # Настраиваем логгер
        self.logger = logging.getLogger(f'weather_logger_{log_filename}')
        self.logger.setLevel(logging.INFO)
        
        # Создаем форматтер для логов
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Добавляем обработчик для файла
        file_handler = logging.FileHandler(self.log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Добавляем обработчик для консоли
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_missing_data(self, year: int, month: int, reason: str) -> None:
        """
        Логирует информацию о ненайденных данных.
        
        Args:
            year (int): Год
            month (int): Месяц
            reason (str): Причина отсутствия данных
        """
        self.logger.warning(
            f"Данные не найдены: {month:02d}.{year} | Причина: {reason}"
        )

    def log_scraping_start(self, start_date: str, end_date: str) -> None:
        """
        Логирует начало процесса сбора данных.
        
        Args:
            start_date (str): Начальная дата
            end_date (str): Конечная дата
        """
        self.logger.info(
            f"Начало сбора данных за период {start_date} - {end_date}"
        )

    def log_scraping_end(self, total_records: int) -> None:
        """
        Логирует завершение процесса сбора данных.
        
        Args:
            total_records (int): Общее количество собранных записей
        """
        self.logger.info(
            f"Сбор данных завершен. Собрано записей: {total_records}"
        )

    def log_error(self, error_msg: str) -> None:
        """
        Логирует ошибки.
        
        Args:
            error_msg (str): Сообщение об ошибке
        """
        self.logger.error(error_msg)

    def log_request_error(self, url: str, error: Exception) -> None:
        """
        Логирует ошибки HTTP-запросов.
        
        Args:
            url (str): URL, вызвавший ошибку
            error (Exception): Объект исключения
        """
        self.logger.error(
            f"Ошибка при запросе URL {url}: {str(error)}"
        )

    def get_log_summary(self) -> dict:
        """
        Возвращает сводку по логам.
        
        Returns:
            dict: Статистика логирования
        """
        with open(self.log_path, 'r', encoding='utf-8') as f:
            logs = f.readlines()
        
        return {
            'total_logs': len(logs),
            'errors': sum(1 for log in logs if 'ERROR' in log),
            'warnings': sum(1 for log in logs if 'WARNING' in log),
            'info': sum(1 for log in logs if 'INFO' in log)
        }

