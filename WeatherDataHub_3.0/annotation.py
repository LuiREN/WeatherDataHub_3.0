import pandas as pd
import csv
from typing import List, Tuple


def create_annotation_file(file_path: str, output_path: str) -> None:
    """
    Создает файл аннотации для заданного CSV файла.

    Args:
        file_path (str): Путь к исходному CSV файлу.
        output_path (str): Путь для сохранения файла аннотации.
    """
    df = pd.read_csv(file_path)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        write_general_info(writer, file_path, df)
        write_column_info(writer, df)


def write_general_info(writer: csv.writer, file_path: str, df: pd.DataFrame) -> None:
    """
    Записывает общую информацию о датасете в файл аннотации.

    Args:
        writer (csv.writer): Объект для записи в CSV.
        file_path (str): Путь к исходному CSV файлу.
        df (pd.DataFrame): DataFrame с данными.
    """
    general_info: List[Tuple[str, str]] = [
        ("Параметр", "Значение"),
        ("Имя файла", file_path.split("/")[-1]),
        ("Количество строк", str(len(df))),
        ("Количество столбцов", str(len(df.columns))),
        ("Начальная дата", str(df['Дата'].min())),
        ("Конечная дата", str(df['Дата'].max()))
    ]
    writer.writerows(general_info)


def write_column_info(writer: csv.writer, df: pd.DataFrame) -> None:
    """
    Записывает информацию о каждом столбце датасета в файл аннотации.

    Args:
        writer (csv.writer): Объект для записи в CSV.
        df (pd.DataFrame): DataFrame с данными.
    """
    for col in df.columns:
        col_info = get_column_info(df, col)
        writer.writerow([col, col_info])


def get_column_info(df: pd.DataFrame, col: str) -> str:
    """
    Формирует строку с информацией о заданном столбце.

    Args:
        df (pd.DataFrame): DataFrame с данными.
        col (str): Имя столбца.

    Returns:
        str: Строка с информацией о столбце.
    """
    dtype = df[col].dtype
    unique_count = df[col].nunique()
    samples = ', '.join(map(str, df[col].sample(min(5, unique_count)).tolist()))
    return f"Тип: {dtype}, Уникальных значений: {unique_count}, Примеры: {samples}"


def read_annotation_file(file_path: str) -> pd.DataFrame:
    """
    Читает файл аннотации и возвращает его содержимое в виде DataFrame.

    Args:
        file_path (str): Путь к файлу аннотации.

    Returns:
        pd.DataFrame: DataFrame с данными из файла аннотации.
    """
    return pd.read_csv(file_path, encoding='utf-8')


if __name__ == "__main__":
    input_file = input("Введите путь к исходному CSV файлу: ")
    output_file = input("Введите путь для сохранения файла аннотации: ")
    create_annotation_file(input_file, output_file)
    print(f"Файл аннотации создан: {output_file}")