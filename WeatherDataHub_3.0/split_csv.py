import pandas as pd
import os
from typing import Optional

def split_csv(input_file: str) -> Optional[str]:
    """
    Разделяет исходный CSV файл на X.csv (даты) и Y.csv (данные).
    
    Args:
        input_file (str): Путь к исходному CSV файлу.
    
    Returns:
        Optional[str]: Путь к папке с результатами или None в случае ошибки.
    """
    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден.")
        return None
        
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    output_folder = os.path.join('dataset', 'split_csv', file_name)
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        df: pd.DataFrame = pd.read_csv(input_file)
        if not pd.to_datetime(df.iloc[:, 0], format='%Y-%m-%d', errors='coerce').notna().all():
            print("Первый столбец не содержит корректные даты в формате ISO 8601.")
            return None
            
        X: pd.Series = df.iloc[:, 0]
        Y: pd.DataFrame = df.iloc[:, 1:]
        
        X.to_csv(os.path.join(output_folder, 'X.csv'), index=False, header=['Date'])
        Y.to_csv(os.path.join(output_folder, 'Y.csv'), index=False)
        
        print(f"Файлы X.csv и Y.csv успешно созданы в папке {output_folder}.")
        return output_folder
    except Exception as e:
        print(f"Ошибка при разделении файла: {str(e)}")
        return None

def split_by_week(input_file: str) -> Optional[str]:
    """
    Разделяет исходный CSV файл на отдельные файлы по неделям.
    
    Args:
        input_file (str): Путь к исходному CSV файлу.
    
    Returns:
        Optional[str]: Путь к папке с результатами или None в случае ошибки.
    """
    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден.")
        return None
        
    try:
        df: pd.DataFrame = pd.read_csv(input_file, parse_dates=['Дата'])
        df['Week'] = df['Дата'].dt.to_period('W')
        grouped: pd.DataFrameGroupBy = df.groupby('Week')
        
        file_name: str = os.path.splitext(os.path.basename(input_file))[0]
        output_folder: str = os.path.join('dataset', 'weekly_data', file_name)
        os.makedirs(output_folder, exist_ok=True)
        
        for week, group in grouped:
            start_date: str = group['Дата'].min().strftime('%Y%m%d')
            end_date: str = group['Дата'].max().strftime('%Y%m%d')
            filename: str = f'{start_date}_{end_date}.csv'
            filepath: str = os.path.join(output_folder, filename)
            group.drop('Week', axis=1).to_csv(filepath, index=False)
            print(f"Создан файл: {filename}")
            
        print(f"Файлы по неделям созданы в папке {output_folder}.")
        return output_folder
    except Exception as e:
        print(f"Ошибка при разделении файла: {str(e)}")
        return None

def split_by_year(input_file: str) -> Optional[str]:
    """
    Разделяет исходный CSV файл на отдельные файлы по годам.
    
    Args:
        input_file (str): Путь к исходному CSV файлу.
    
    Returns:
        Optional[str]: Путь к папке с результатами или None в случае ошибки.
    """
    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден.")
        return None
        
    try:
        df: pd.DataFrame = pd.read_csv(input_file, parse_dates=['Дата'])
        grouped: pd.DataFrameGroupBy = df.groupby(df['Дата'].dt.year)
        
        file_name: str = os.path.splitext(os.path.basename(input_file))[0]
        output_folder: str = os.path.join('dataset', 'yearly_data', file_name)
        os.makedirs(output_folder, exist_ok=True)
        
        for year, group in grouped:
            start_date: str = group['Дата'].min().strftime('%Y%m%d')
            end_date: str = group['Дата'].max().strftime('%Y%m%d')
            filename: str = f'{start_date}_{end_date}.csv'
            filepath: str = os.path.join(output_folder, filename)
            group.to_csv(filepath, index=False)
            print(f"Создан файл: {filename}")
            
        print(f"Файлы по годам созданы в папке {output_folder}.")
        return output_folder
    except Exception as e:
        print(f"Ошибка при разделении файла: {str(e)}")
        return None

if __name__ == "__main__":
    input_file: str = input("Введите путь к исходному CSV файлу: ")
    split_csv(input_file)