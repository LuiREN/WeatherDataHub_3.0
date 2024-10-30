import requests
from bs4 import BeautifulSoup
import csv
import os
from datetime import datetime, timedelta
from weather_logger import WeatherLogger

CLOUDINESS = {
    'sun.png': 'Ясно',
    'sunc.png': 'Малооблачно',
    'suncl.png': 'Переменная облачность',
    'dull.png': 'Пасмурно'
}

class WeatherScraper:
    def __init__(self):
        self.logger: Optional[WeatherLogger] = None

    def get_weather_data(self, year: int, month: int) -> list:
        url = f"https://www.gismeteo.ru/diary/4618/{year}/{month:02d}/"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
        except requests.RequestException as e:
            if self.logger:
                self.logger.log_request_error(url, e)
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', attrs={"align": "center", "valign": "top", "border": "0"})
        
        if not table:
            if self.logger:
                self.logger.log_missing_data(
                    year, month, 
                    "Таблица с данными не найдена на странице"
                )
            return []

        data = []
        rows = table.find_all('tr')[2:]
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 11:
                date = f"{year}-{month:02d}-{cols[0].text.strip()}"
                temp_day = cols[1].text.strip()
                pressure_day = cols[2].text.strip()
                cloudiness_day = self.get_cloudiness(cols[3])
                wind_day = cols[5].text.strip().split('\n')[-1]
                temp_evening = cols[6].text.strip()
                pressure_evening = cols[7].text.strip()
                cloudiness_evening = self.get_cloudiness(cols[8])
                wind_evening = cols[10].text.strip().split('\n')[-1]
                data.append([
                    date, temp_day, pressure_day, cloudiness_day, wind_day,
                    temp_evening, pressure_evening, cloudiness_evening, wind_evening
                ])

        return data

    def get_cloudiness(self, cell):
        img = cell.find('img', class_='screen_icon')
        if img and 'src' in img.attrs:
            src = img['src'].split('/')[-1]
            return CLOUDINESS.get(src, 'Неизвестно')
        return 'Нет данных'

    def run(self, start_date: str, end_date: str, progress_callback, status_callback) -> str:
        start_date = datetime.strptime(start_date, "%m.%Y")
        end_date = datetime.strptime(end_date, "%m.%Y")
        
        filename = f'samara_weather_{start_date.strftime("%Y%m")}-{end_date.strftime("%Y%m")}.csv'
        self.logger = WeatherLogger(filename)
        
        self.logger.log_scraping_start(start_date.strftime("%m.%Y"), end_date.strftime("%m.%Y"))
        
        all_data = []
        current_date = start_date
        total_months = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1
        processed_months = 0
        
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            status_callback.emit(f"Получение данных за {month:02d}.{year}")
            
            month_data = self.get_weather_data(year, month)
            all_data.extend(month_data)
            
            current_date += timedelta(days=32)
            current_date = current_date.replace(day=1)
            
            processed_months += 1
            progress = int((processed_months / total_months) * 100)
            progress_callback.emit(progress)

        self.save_to_csv(all_data, filename)
        self.logger.log_scraping_end(len(all_data))
        
        # Получаем и логируем сводку
        summary = self.logger.get_log_summary()
        self.logger.logger.info(
            f"Итоги сбора данных:\n"
            f"Всего логов: {summary['total_logs']}\n"
            f"Ошибок: {summary['errors']}\n"
            f"Предупреждений: {summary['warnings']}\n"
            f"Информационных сообщений: {summary['info']}"
        )
        
        return filename

    def save_missing_data(self, csv_filename):
        # Создаем папку для логов, если её нет
        logs_folder = 'missing_data_logs'
        if not os.path.exists(logs_folder):
            os.makedirs(logs_folder)
        
        # Создаем имя файла на основе имени CSV файла
        base_name = os.path.splitext(csv_filename)[0]
        log_filename = os.path.join(logs_folder, f"{base_name}_missing_data.txt")
        
        # Записываем данные в файл
        with open(log_filename, 'w', encoding='utf-8') as f:
            f.write("Список ненайденных данных:\n")
            f.write("=" * 50 + "\n")
            for item in self.missing_data:
                f.write(f"{item}\n")

    def save_to_csv(self, data, filename):
        dataset_folder = 'dataset'
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
        
        filepath = os.path.join(dataset_folder, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Дата', 'Температура (день)', 'Давление (день)', 'Облачность (день)', 'Ветер (день)',
                'Температура (вечер)', 'Давление (вечер)', 'Облачность (вечер)', 'Ветер (вечер)'
            ])
            writer.writerows(data)