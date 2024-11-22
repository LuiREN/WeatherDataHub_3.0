from typing import Optional, Dict, List, Tuple
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QSpinBox, QGroupBox, QMessageBox, 
    QTableWidget, QTableWidgetItem, QProgressBar, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from ml_model import WeatherModel

class MLTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        """Инициализация вкладки машинного обучения."""
        super().__init__(parent)
        self.df = None
        self.model = WeatherModel()
        self.train_data = None
        self.test_data = None
        
        # Инициализация интерфейса
        self.setup_logger()
        self.init_ui()

    def setup_logger(self) -> None:
        """Настройка логирования."""
        self.logger = logging.getLogger('MLInterface')
        self.logger.setLevel(logging.INFO)
        
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        handler = logging.FileHandler('logs/ml_interface.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def init_ui(self) -> None:
        """Инициализация пользовательского интерфейса."""
        layout = QHBoxLayout()
        
        # Создаем панели
        left_panel = self.create_left_panel()
        right_panel = self.create_right_panel()
        
        # Добавляем панели в главный layout
        layout.addWidget(left_panel, stretch=2)  # 40%
        layout.addWidget(right_panel, stretch=3)  # 60%
        
        self.setLayout(layout)

    def create_left_panel(self) -> QFrame:
        """Создание левой панели управления."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout()
    
        # 1. Группа подготовки данных
        data_group = QGroupBox("Подготовка данных")
        data_layout = QVBoxLayout()
    
        # Размер тестовой выборки
        test_size_layout = QHBoxLayout()
        test_size_layout.addWidget(QLabel("Тестовая выборка:"))
        self.test_size_spin = QSpinBox()
        self.test_size_spin.setRange(10, 40)
        self.test_size_spin.setValue(20)
        self.test_size_spin.setSuffix("%")
        test_size_layout.addWidget(self.test_size_spin)
        data_layout.addLayout(test_size_layout)
    
        # Кнопка подготовки данных
        self.prepare_btn = QPushButton("Подготовить данные")
        self.prepare_btn.clicked.connect(self.prepare_data)
        self.prepare_btn.setEnabled(False)
        data_layout.addWidget(self.prepare_btn)
    
        data_group.setLayout(data_layout)

        # 2. Группа анализа временного ряда
        analysis_group = QGroupBox("Анализ временного ряда")
        analysis_layout = QVBoxLayout()
    
        # Кнопка анализа
        self.analyze_btn = QPushButton("Анализировать данные")
        self.analyze_btn.clicked.connect(self.analyze_time_series)
        self.analyze_btn.setEnabled(False)
        analysis_layout.addWidget(self.analyze_btn)
    
        analysis_group.setLayout(analysis_layout)
    
        # 3. Группа параметров SARIMA
        model_group = QGroupBox("Параметры SARIMA")
        model_layout = QVBoxLayout()
    
        # Параметры p, d, q
        pdq_layout = QHBoxLayout()
    
        pdq_layout.addWidget(QLabel("p:"))
        self.p_spin = QSpinBox()
        self.p_spin.setRange(0, 3)
        self.p_spin.setValue(1)
        pdq_layout.addWidget(self.p_spin)
    
        pdq_layout.addWidget(QLabel("d:"))
        self.d_spin = QSpinBox()
        self.d_spin.setRange(0, 2)
        self.d_spin.setValue(1)
        pdq_layout.addWidget(self.d_spin)
    
        pdq_layout.addWidget(QLabel("q:"))
        self.q_spin = QSpinBox()
        self.q_spin.setRange(0, 3)
        self.q_spin.setValue(1)
        pdq_layout.addWidget(self.q_spin)
    
        model_layout.addLayout(pdq_layout)
    
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
        self.s_spin.setValue(7)
        seasonal_layout.addWidget(self.s_spin)
    
        model_layout.addLayout(seasonal_layout)
        model_group.setLayout(model_layout)
    
        # 4. Группа кнопок управления
        control_group = QGroupBox("Управление")
        control_layout = QVBoxLayout()
    
        # Кнопка обучения
        self.train_btn = QPushButton("Обучить модель")
        self.train_btn.clicked.connect(self.train_model)
        self.train_btn.setEnabled(False)
        control_layout.addWidget(self.train_btn)
    
        # Кнопка подбора параметров
        self.tune_btn = QPushButton("Подобрать параметры")
        self.tune_btn.clicked.connect(self.tune_parameters)
        self.tune_btn.setEnabled(False)
        control_layout.addWidget(self.tune_btn)
    
        # Кнопка сохранения
        self.save_btn = QPushButton("Сохранить модель")
        self.save_btn.clicked.connect(self.save_model)
        self.save_btn.setEnabled(False)
        control_layout.addWidget(self.save_btn)
    
        control_group.setLayout(control_layout)
    
        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
    
        # Добавляем все группы в основной layout
        layout.addWidget(data_group)
        layout.addWidget(analysis_group)
        layout.addWidget(model_group)
        layout.addWidget(control_group)
        layout.addWidget(self.progress_bar)
        layout.addStretch()
    
        panel.setLayout(layout)
        return panel

    def create_right_panel(self) -> QFrame:
        """Создание правой панели визуализации."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout()
        
        # Информационная метка
        self.info_label = QLabel("Загрузите данные для начала работы")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.info_label)
        
        # Таблица результатов
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Метрика", "Значение"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table)
        
        # Область для графика
        self.plot_frame = QFrame()
        self.plot_frame.setLayout(QVBoxLayout())
        layout.addWidget(self.plot_frame)
        
        panel.setLayout(layout)
        return panel

  
    def load_data(self, df: pd.DataFrame) -> None:
        """
        Загрузка данных для анализа.
    
        Args:
            df: DataFrame с данными
        """
        try:
            # Проверяем наличие необходимых столбцов
            required_columns = ['date', 'temperature_day']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("В данных отсутствуют необходимые столбцы: date, temperature_day")
        
            # Создаем копию данных
            self.df = df.copy()
        
            # Проверяем формат даты
            self.df['date'] = pd.to_datetime(self.df['date'])
        
            # Активируем кнопку подготовки данных
            self.prepare_btn.setEnabled(True)
        
            # Обновляем информацию
            self.info_label.setText(
                "Данные загружены успешно\n"
                "Выполните подготовку данных для обучения модели"
            )
        
            self.logger.info(
                f"Загружены данные: {len(self.df)} строк, "
                f"период: с {self.df['date'].min().date()} по {self.df['date'].max().date()}"
            )
        
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке данных: {str(e)}")


    def prepare_data(self) -> None:
        """Подготовка данных для обучения."""
        try:
            if self.df is None:
                raise ValueError("Данные не загружены")
            
            # Получаем размер тестовой выборки
            test_size = self.test_size_spin.value() / 100
            
            # Разделяем данные
            split_idx = int(len(self.df) * (1 - test_size))
            
            self.train_data = {
                'temperature_day': self.df['temperature_day'][:split_idx]
            }
            
            self.test_data = {
                'temperature_day': self.df['temperature_day'][split_idx:]
            }
            
            # Обновляем интерфейс
            self.train_btn.setEnabled(True)
            self.analyze_btn.setEnabled(True)
            self.tune_btn.setEnabled(True)
            self.info_label.setText(
                f"Данные подготовлены:\n"
                f"Размер обучающей выборки: {len(self.train_data['temperature_day'])}\n"
                f"Размер тестовой выборки: {len(self.test_data['temperature_day'])}"
            )
            
            self.logger.info("Данные успешно подготовлены для обучения")
            
        except Exception as e:
            self.logger.error(f"Ошибка при подготовке данных: {str(e)}")
            QMessageBox.critical(self, "Ошибка", str(e))

    def analyze_time_series(self) -> None:
        """Анализ временного ряда перед обучением."""
        try:
            if self.train_data is None:
                raise ValueError("Данные не подготовлены")
            
            self.progress_bar.setVisible(True)
            self.info_label.setText("Выполняется анализ временного ряда...")

            # Вывод отладочной информации
            print("Информация о train_data:")
            print(f"Тип train_data: {type(self.train_data)}")
            print(f"Содержимое train_data: {self.train_data}")
            print(f"Ключи train_data: {self.train_data.keys() if isinstance(self.train_data, dict) else 'не словарь'}")
        
            # Получаем данные температуры
            temperature_data = self.train_data['temperature_day']
            print(f"\nИнформация о температурных данных:")
            print(f"Тип данных: {type(temperature_data)}")
            print(f"Количество значений: {len(temperature_data)}")
            print(f"Количество NaN: {pd.isna(temperature_data).sum()}")
            print(f"Первые несколько значений: {temperature_data[:5]}")
        
            # Преобразуем в Series с индексом
            temperature_series = pd.Series(temperature_data)
        
            # Проверяем, есть ли данные
            if len(temperature_series.dropna()) < 2:
                raise ValueError(f"Недостаточно данных для анализа. Найдено значений: {len(temperature_series.dropna())}")
            
            # Анализируем временной ряд
            analysis_results = self.model.analyze_time_series(temperature_series)
        
            if analysis_results and 'error' not in analysis_results:
                msg = "Результаты анализа временного ряда:\n\n"
            
                # Добавляем базовую статистику
                if 'statistics' in analysis_results:
                    stats = analysis_results['statistics']
                    msg += "Общая статистика:\n"
                    msg += f"Количество наблюдений: {stats['n_observations']}\n"
                    msg += f"Среднее значение: {stats['mean']:.2f}\n"
                    msg += f"Стандартное отклонение: {stats['std']:.2f}\n"
                    msg += f"Минимум: {stats['min']:.2f}\n"
                    msg += f"Максимум: {stats['max']:.2f}\n\n"
            
                if 'stationarity' in analysis_results:
                    stationarity = analysis_results['stationarity']
                    msg += "Стационарность:\n"
                    if 'error' not in stationarity:
                        msg += f"- Тест-статистика: {stationarity['test_statistic']:.4f}\n"
                        msg += f"- p-значение: {stationarity['p_value']:.4f}\n"
                        msg += f"- Ряд {'стационарен' if stationarity['is_stationary'] else 'не стационарен'}\n"
                    else:
                        msg += f"- {stationarity['error']}\n"
                    msg += "\n"
            
                if 'autocorrelation' in analysis_results:
                    autocorr = analysis_results['autocorrelation']
                    msg += "Автокорреляция:\n"
                    if 'error' not in autocorr:
                        msg += f"- Найдено {len(autocorr['significant_lags'])} значимых лагов\n"
                    else:
                        msg += f"- {autocorr['error']}\n"
            
                if 'report_file' in analysis_results:
                    msg += f"\nПодробный отчет сохранен в:\n{analysis_results['report_file']}"
            
                # Показываем сообщение
                QMessageBox.information(self, "Анализ временного ряда", msg)
            
                # Обновляем информацию на интерфейсе
                self.info_label.setText(
                    "Анализ временного ряда завершен\n"
                    f"Отчет сохранен в: {analysis_results.get('report_file', 'unknown')}"
                )
            
            else:
                error_msg = analysis_results.get('error', 'Неизвестная ошибка при анализе')
                raise ValueError(error_msg)
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе временного ряда: {str(e)}")
            QMessageBox.critical(self, "Ошибка", str(e))
        
        finally:
            self.progress_bar.setVisible(False)

    def train_model(self) -> None:
        """Обучение модели и анализ результатов."""
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(5)
        
            # Анализируем временной ряд перед обучением
            self.info_label.setText("Анализ временного ряда...")
            analysis_results = self.model.analyze_time_series(
                self.train_data['temperature_day']
            )
        
            self.progress_bar.setValue(20)
        
            if analysis_results:
                # Используем стационарные данные если они есть
                if 'stationary_data' in analysis_results:
                    train_data = analysis_results['stationary_data']
                else:
                    train_data = self.train_data['temperature_day']
            
                # Получаем параметры
                order = (self.p_spin.value(), self.d_spin.value(), self.q_spin.value())
                seasonal_order = (
                    self.P_spin.value(),
                    self.D_spin.value(),
                    self.Q_spin.value(),
                    self.s_spin.value()
                )
            
                # Обучаем модель
                result = self.model.train(train_data, order, seasonal_order)
            
                self.progress_bar.setValue(40)
            
                if result['status'] == 'success':
                    # Делаем прогноз
                    predictions = self.model.predict(len(self.test_data['temperature_day']))
                
                    if predictions is not None:
                        self.progress_bar.setValue(60)
                    
                        # Проводим подробный анализ прогноза
                        analysis_results = self.model.analyze_forecast(
                            self.test_data['temperature_day'].values,
                            predictions,
                            self.test_data['temperature_day'].index
                        )
                    
                        # Сохраняем отчет
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        report_file = f'results/forecast_report_{timestamp}.txt'
                        os.makedirs('results', exist_ok=True)
                        self.model.save_forecast_report(analysis_results, report_file)
                    
                        self.progress_bar.setValue(80)
                    
                        # Обновляем таблицу результатов
                        self.update_results_table(analysis_results['metrics'])
                    
                        # Строим график
                        self.plot_forecast_analysis(
                            self.test_data['temperature_day'],
                            predictions,
                            analysis_results
                        )
                    
                        # Обновляем информацию
                        self.info_label.setText(
                            f"Модель обучена успешно\n"
                            f"MSE: {analysis_results['metrics']['mse']:.4f}\n"
                            f"RMSE: {analysis_results['metrics']['rmse']:.4f}\n"
                            f"MAPE: {analysis_results['metrics']['mape']:.2f}%\n"
                            f"Отчет сохранен в: {report_file}"
                        )
                    
                        # Активируем кнопку сохранения
                        self.save_btn.setEnabled(True)
                    
                    self.progress_bar.setValue(100)
                else:
                    raise ValueError(result['message'])
            
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {str(e)}")
            QMessageBox.critical(self, "Ошибка", str(e))
        finally:
            self.progress_bar.setVisible(False)

    def plot_forecast_analysis(self, actual: pd.Series, predicted: np.ndarray, analysis_results: Dict) -> None:
        """
        Построение расширенного графика анализа прогноза.
    
        Args:
            actual: Фактические значения
            predicted: Прогнозные значения
            analysis_results: Результаты анализа
        """
        try:
            # Очищаем предыдущий график
            for i in reversed(range(self.plot_frame.layout().count())): 
                self.plot_frame.layout().itemAt(i).widget().deleteLater()
        
            # Создаем фигуру с двумя графиками
            fig = Figure(figsize=(10, 8))
        
            # График прогноза
            ax1 = fig.add_subplot(211)
            ax1.plot(actual.index, actual.values, 
                    label='Фактические значения', color='blue')
            ax1.plot(actual.index, predicted, 
                    label='Прогноз', color='red', linestyle='--')
            ax1.set_title('Сравнение прогноза с фактическими значениями')
            ax1.set_xlabel('Дата')
            ax1.set_ylabel('Температура (°C)')
            ax1.legend()
            ax1.grid(True)
        
            # График ошибок
            ax2 = fig.add_subplot(212)
            errors = actual.values - predicted
            ax2.plot(actual.index, errors, color='green', marker='o')
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_title('Ошибки прогноза')
            ax2.set_xlabel('Дата')
            ax2.set_ylabel('Ошибка (°C)')
            ax2.grid(True)
        
            # Добавляем аннотации с метриками
            metrics_text = (
                f"MSE: {analysis_results['metrics']['mse']:.4f}\n"
                f"RMSE: {analysis_results['metrics']['rmse']:.4f}\n"
                f"MAPE: {analysis_results['metrics']['mape']:.2f}%"
            )
            fig.text(0.02, 0.02, metrics_text, fontsize=8)
        
            # Поворачиваем подписи дат
            fig.autofmt_xdate()
        
            # Добавляем график на форму
            canvas = FigureCanvas(fig)
            self.plot_frame.layout().addWidget(canvas)
        
        except Exception as e:
            self.logger.error(f"Ошибка при построении графика: {str(e)}")
            QMessageBox.critical(self, "Ошибка", str(e))

    def update_results_table(self, metrics: Dict[str, float]) -> None:
        """Обновление таблицы результатов."""
        self.results_table.setRowCount(0)
        for name, value in metrics.items():
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(name))
        
            # Проверяем значение метрики
            if np.isnan(value):
                value_str = "Не удалось рассчитать"
            elif name == 'mape':
                value_str = f"{value:.2f}%" if not np.isinf(value) else "Не удалось рассчитать"
            else:
                value_str = f"{value:.4f}"
            
            self.results_table.setItem(row, 1, QTableWidgetItem(value_str))

 
    def tune_parameters(self) -> None:
        """Подбор оптимальных параметров модели."""
        try:
            self.progress_bar.setVisible(True)
            self.info_label.setText("Выполняется подбор параметров...")
        
            parameter_combinations = [
                # (p, d, q, P, D, Q, s)
                (0, 1, 1, 0, 1, 1, 7),  # Комбинация 1
                (1, 1, 1, 1, 1, 1, 7),  # Комбинация 2
                (2, 1, 2, 0, 1, 1, 7),  # Комбинация 3
                (1, 1, 2, 1, 1, 1, 12), # Комбинация 4
                (2, 1, 2, 1, 1, 1, 12)  # Комбинация 5
            ]
        
            results = []
            best_rmse = float('inf')
            best_params = None
        
            total_combinations = len(parameter_combinations)
        
            for i, params in enumerate(parameter_combinations):
                try:
                    # Распаковываем параметры
                    p, d, q, P, D, Q, s = params
                    order = (p, d, q)
                    seasonal_order = (P, D, Q, s)
                
                    # Обучаем модель
                    result = self.model.train(
                        self.train_data['temperature_day'],
                        order,
                        seasonal_order
                    )
                
                    if result['status'] == 'success':
                        # Делаем прогноз
                        predictions = self.model.predict(len(self.test_data['temperature_day']))
                    
                        if predictions is not None:
                            # Оцениваем качество
                            metrics = self.model.evaluate(
                                self.test_data['temperature_day'].values,
                                predictions
                            )
                        
                            results.append({
                                'parameters': {
                                    'order': order,
                                    'seasonal_order': seasonal_order
                                },
                                'metrics': metrics
                            })
                        
                            # Проверяем, лучше ли текущий результат
                            if metrics['rmse'] < best_rmse:
                                best_rmse = metrics['rmse']
                                best_params = params
                
                    # Обновляем прогресс
                    progress = int((i + 1) / total_combinations * 100)
                    self.progress_bar.setValue(progress)
                
                except Exception as e:
                    self.logger.warning(f"Ошибка для комбинации {params}: {str(e)}")
                    continue
        
            # Сохраняем результаты
            self.save_tuning_results(results, best_params)
        
            # Устанавливаем лучшие параметры
            if best_params:
                p, d, q, P, D, Q, s = best_params
                self.p_spin.setValue(p)
                self.d_spin.setValue(d)
                self.q_spin.setValue(q)
                self.P_spin.setValue(P)
                self.D_spin.setValue(D)
                self.Q_spin.setValue(Q)
                self.s_spin.setValue(s)
            
                QMessageBox.information(
                    self,
                    "Подбор параметров",
                    f"Найдены оптимальные параметры:\n"
                    f"order=({p},{d},{q})\n"
                    f"seasonal_order=({P},{D},{Q},{s})\n"
                    f"RMSE={best_rmse:.4f}"
                )
        
            self.info_label.setText("Подбор параметров завершен")
        
        except Exception as e:
            self.logger.error(f"Ошибка при подборе параметров: {str(e)}")
            QMessageBox.critical(self, "Ошибка", str(e))
        
        finally:
            self.progress_bar.setVisible(False)

    def save_tuning_results(self, results: List[Dict], best_params: Tuple) -> None:
        """
        Сохранение результатов подбора параметров.
    
        Args:
            results: Список результатов для всех комбинаций
            best_params: Лучшие найденные параметры
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'results/parameter_tuning_{timestamp}.txt'
        
            if not os.path.exists('results'):
                os.makedirs('results')
        
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Результаты подбора параметров SARIMA\n")
                f.write("=" * 50 + "\n\n")
            
                # Лучшие параметры
                f.write("Лучшие параметры:\n")
                f.write("-" * 20 + "\n")
                p, d, q, P, D, Q, s = best_params
                f.write(f"order (p,d,q): ({p},{d},{q})\n")
                f.write(f"seasonal_order (P,D,Q,s): ({P},{D},{Q},{s})\n\n")
            
                # Результаты всех комбинаций
                f.write("Все проверенные комбинации:\n")
                f.write("-" * 20 + "\n")
            
                for result in results:
                    params = result['parameters']
                    metrics = result['metrics']
                
                    f.write("\nПараметры:\n")
                    f.write(f"order: {params['order']}\n")
                    f.write(f"seasonal_order: {params['seasonal_order']}\n")
                    f.write("Метрики:\n")
                    for metric_name, metric_value in metrics.items():
                        f.write(f"{metric_name}: {metric_value:.4f}\n")
                    f.write("-" * 20 + "\n")
            
                f.write(f"\nДата создания отчета: {timestamp}")
        
            self.logger.info(f"Результаты подбора параметров сохранены в {filename}")
        
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении результатов: {str(e)}")

    def update_info_label(self, analysis_results: Dict) -> None:
        """Обновление информационной метки."""
        metrics = analysis_results['metrics']
        info_text = "Модель обучена успешно\n"
    
        if not np.isnan(metrics['mse']):
            info_text += f"MSE: {metrics['mse']:.4f}\n"
        if not np.isnan(metrics['rmse']):
            info_text += f"RMSE: {metrics['rmse']:.4f}\n"
        if not np.isnan(metrics['mape']):
            info_text += f"MAPE: {metrics['mape']:.2f}%\n"
        else:
            info_text += "MAPE: Не удалось рассчитать\n"
        
        self.info_label.setText(info_text)


    def plot_results(self, actual: pd.Series, predicted: np.ndarray) -> None:
        """Построение графика результатов."""
        try:
            # Очищаем предыдущий график
            for i in reversed(range(self.plot_frame.layout().count())): 
                self.plot_frame.layout().itemAt(i).widget().deleteLater()
            
            # Создаем новый график
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            
            # Строим графики
            ax.plot(actual.index, actual.values, 
                   label='Фактические значения', color='blue')
            ax.plot(actual.index, predicted, 
                   label='Прогноз', color='red', linestyle='--')
            
            ax.set_title('Сравнение прогноза с фактическими значениями')
            ax.set_xlabel('Дата')
            ax.set_ylabel('Температура')
            ax.legend()
            ax.grid(True)
            
            # Поворачиваем подписи дат
            fig.autofmt_xdate()
            
            # Добавляем график на форму
            canvas = FigureCanvas(fig)
            self.plot_frame.layout().addWidget(canvas)
            
        except Exception as e:
            self.logger.error(f"Ошибка при построении графика: {str(e)}")
            QMessageBox.critical(self, "Ошибка", str(e))

    def save_model(self) -> None:
       """Сохранение обученной модели."""
       try:
           # Создаем директорию если её нет
           if not os.path.exists('models'):
               os.makedirs('models')
           
           # Генерируем имя файла
           timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
           filename = os.path.join('models', f'sarima_model_{timestamp}.txt')
           
           # Собираем информацию о модели
           model_info = {
               'parameters': {
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
                   )
               },
               'training_data': {
                   'train_size': len(self.train_data['temperature_day']),
                   'test_size': len(self.test_data['temperature_day']),
               }
           }
           
           # Получаем прогноз и метрики для сохранения
           predictions = self.model.predict(len(self.test_data['temperature_day']))
           metrics = self.model.evaluate(
               self.test_data['temperature_day'].values,
               predictions
           )
           
           # Сохраняем всю информацию в файл
           with open(filename, 'w', encoding='utf-8') as f:
               f.write("Отчет о модели SARIMA\n")
               f.write("=" * 50 + "\n\n")
               
               # Параметры модели
               f.write("Параметры модели:\n")
               f.write("-" * 20 + "\n")
               f.write(f"order (p,d,q): {model_info['parameters']['order']}\n")
               f.write(f"seasonal_order (P,D,Q,s): {model_info['parameters']['seasonal_order']}\n\n")
               
               # Информация о данных
               f.write("Информация о данных:\n")
               f.write("-" * 20 + "\n")
               f.write(f"Размер обучающей выборки: {model_info['training_data']['train_size']}\n")
               f.write(f"Размер тестовой выборки: {model_info['training_data']['test_size']}\n\n")
               
               # Метрики качества
               f.write("Метрики качества:\n")
               f.write("-" * 20 + "\n")
               for metric, value in metrics.items():
                   f.write(f"{metric}: {value:.4f}\n")
               
               f.write(f"\nМодель сохранена: {timestamp}")
           
           QMessageBox.information(
               self,
               "Сохранение модели",
               f"Модель успешно сохранена в файл:\n{filename}"
           )
           
           self.logger.info(f"Модель сохранена в {filename}")
           
       except Exception as e:
           self.logger.error(f"Ошибка при сохранении модели: {str(e)}")
           QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении модели: {str(e)}")