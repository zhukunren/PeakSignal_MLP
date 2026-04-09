# gui.py
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QLineEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox,
    QDateEdit, QSpinBox, QGroupBox, QComboBox
)
from PyQt5.QtWebEngineWidgets import QWebEngineView

from models import set_seed
from preprocess import read_day_fromtdx, select_time, preprocess_data
from train import train_model
from predict import predict_new_data
from plotly.subplots import make_subplots
import plotly.graph_objs as go

import itertools
# 设置随机种子
set_seed(42)




class PlotWorker(QThread):
    plot_ready = pyqtSignal(str)

    def __init__(self, data, stock_code, start_date, end_date, title="K线图", peaks=None, troughs=None, prediction=False, selected_classifiers=None, bt_result=None):
        super().__init__()
        self.data = data
        self.stock_code = stock_code
        self.start_date = start_date
        self.end_date = end_date
        self.title = title
        self.peaks = peaks
        self.troughs = troughs
        self.prediction = prediction
        self.selected_classifiers = selected_classifiers
        self.bt_result = bt_result  # 回测结果字典

    def run(self):
        try:
            html_chart = self.plot_candlestick_plotly(
                self.data, self.stock_code, self.start_date, self.end_date,
                self.peaks, self.troughs, self.prediction
            )
            self.plot_ready.emit(html_chart)
        except Exception as e:
            self.plot_ready.emit(f"Error: {e}")

    
    def plot_candlestick_plotly(self, data, stock_code, start_date, end_date, peaks=None, troughs=None, prediction=False):
            if prediction and self.selected_classifiers:
                classifiers_str = ", ".join(self.selected_classifiers)
                title = f"{stock_code} {start_date} 至 {end_date} 基础模型: {classifiers_str}"
            else:
                title = f"{stock_code} {start_date} 至 {end_date}"

            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    data.index = pd.to_datetime(data.index)
                except Exception as e:
                    raise ValueError(f"data.index 无法转换为日期格式: {e}")
            data.index = data.index.strftime('%Y-%m-%d')

            if peaks is not None and not peaks.empty:
                if not isinstance(peaks.index, pd.DatetimeIndex):
                    try:
                        peaks.index = pd.to_datetime(peaks.index)
                    except Exception as e:
                        raise ValueError(f"peaks.index 无法转换为日期格式: {e}")
                peaks.index = peaks.index.strftime('%Y-%m-%d')

            if troughs is not None and not troughs.empty:
                if not isinstance(troughs.index, pd.DatetimeIndex):
                    try:
                        troughs.index = pd.to_datetime(troughs.index)
                    except Exception as e:
                        raise ValueError(f"troughs.index 无法转换为日期格式: {e}")
                troughs.index = troughs.index.strftime('%Y-%m-%d')

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.7, 0.3],
                specs=[[{"type": "candlestick"}],[{"type": "bar"}]]
            )

            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=stock_code,
                increasing=dict(line=dict(color='red')),
                decreasing=dict(line=dict(color='green')),
                hoverinfo='x+y+text',
            ), row=1, col=1)

            if 'Volume' in data.columns:
                volume_colors = ['red' if row['Close'] > row['Open'] else 'green' for _, row in data.iterrows()]
                fig.add_trace(go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    marker_color=volume_colors,
                    name='成交量',
                    hoverinfo='x+y'
                ), row=2, col=1)

            if peaks is not None and not peaks.empty:
                marker_y_peaks = peaks['High'] * 1.02
                marker_x_peaks = peaks.index
                color_peak = 'green'
                label_peak = '局部高点' if not prediction else '预测高点'
                fig.add_trace(go.Scatter(
                    x=marker_x_peaks,
                    y=marker_y_peaks,
                    mode='text',
                    text='W',
                    textfont=dict(color=color_peak, size=20),
                    name=label_peak
                ), row=1, col=1)

            if troughs is not None and not troughs.empty:
                marker_y_troughs = troughs['Low'] * 0.98
                marker_x_troughs = troughs.index
                color_trough = 'red'
                label_trough = '局部低点' if not prediction else '预测低点'
                fig.add_trace(go.Scatter(
                    x=marker_x_troughs,
                    y=marker_y_troughs,
                    mode='text',
                    text='D',
                    textfont=dict(color=color_trough, size=20),
                    name=label_trough
                ), row=1, col=1)

            # 显示每个回测结果指标在图表右侧
            if self.bt_result is not None:
                y_position = 1  # 起始Y位置
                for key, value in self.bt_result.items():
                    if isinstance(value, float):  # 确保只显示数字类型
                        if key in {"同期标的涨跌幅", '"波段盈"累计收益率', "超额收益率", 
                                "单笔交易最大收益", "单笔交易最低收益", "单笔平均收益率", "胜率"}:
                            value_display = f"{value*100:.2f}%"  # 百分比显示
                        else:
                            value_display = f"{value:.2f}"

                        # 添加每个回测结果到图表的右侧
                        fig.add_annotation(
                            text=f"{key}: {value_display}",
                            xref="paper", yref="paper",
                            x=1.12,  # 让文本左对齐，适当调整
                            y=0.8 - y_position * 0.06,  # 控制Y位置，使其分段排列
                            showarrow=False,
                            align="left",  # 设为左对齐
                            bordercolor="black",
                            borderwidth=1,
                            bgcolor="white",
                            opacity=0.8
                        )
                        y_position += 1  # 向下偏移下一行

            fig.update_layout(
                title=title,
                xaxis=dict(
                    title="日期",
                    type="category",
                    tickangle=45,
                    tickmode="auto",
                    nticks=10
                ),
                xaxis2=dict(
                    title="日期",
                    type="category",
                    tickangle=45,
                    tickmode="auto",
                    nticks=10
                ),
                yaxis_title="价格",
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                template='plotly_white',
                showlegend=True,
                height=800,
                font=dict(
                    family="Microsoft YaHei, SimHei",
                    size=14,
                    color="black"
                )
            )

            html = fig.to_html(include_plotlyjs='cdn')
            html = html.replace('</head>', '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head>')
            html = html.replace(
                '<body>',
                '<body><script src="https://cdn.plot.ly/locale/zh-cn.js"></script><script>Plotly.setPlotConfig({locale: "zh-CN"});</script>'
            )
            return html
class TrainWorker(QThread):
    initial_plot_ready = pyqtSignal(str)
    training_finished = pyqtSignal(
        str,
        object, object, object, list, list, float, dict, float,
        object, object, object, list, list, float, dict, float
    )
    error = pyqtSignal(str)
    def __init__(self, file_path, stock_code, start_date, end_date, N, classifier_name, mixture_depth, n_features_selected, oversample_method):
        super().__init__()
        self.file_path = file_path
        self.stock_code = stock_code
        self.start_date = start_date
        self.end_date = end_date
        self.N = N
        self.classifier_name = classifier_name
        self.mixture_depth = mixture_depth
        self.n_features_selected = n_features_selected
        self.oversample_method = oversample_method
    def run(self):
        try:
            stock_code_tdx = self.stock_code[-2:].lower() + self.stock_code[:6]
            data = read_day_fromtdx(self.file_path, stock_code_tdx)
            if data.empty:
                raise ValueError("读取的数据为空，请检查文件路径和股票代码。")
            print("读取的数据预览：")
            print(data.head())
            df = select_time(data, self.start_date, self.end_date)
            if df.empty:
                raise ValueError("训练集为空，请检查日期范围和数据文件")
            df_preprocessed, all_features = preprocess_data(
                df, self.N, self.mixture_depth, mark_labels=True
            )
            peaks = df_preprocessed[df_preprocessed['Peak'] == 1] if 'Peak' in df_preprocessed.columns else pd.DataFrame()
            troughs = df_preprocessed[df_preprocessed['Trough'] == 1] if 'Trough' in df_preprocessed.columns else pd.DataFrame()
            plot_worker = PlotWorker(
                df_preprocessed, self.stock_code, self.start_date, self.end_date,
                title="训练集K线图", peaks=peaks, troughs=troughs, prediction=False, selected_classifiers=None
            )
            fig_html_initial = plot_worker.plot_candlestick_plotly(
                df_preprocessed, self.stock_code, self.start_date, self.end_date, 
                peaks=peaks, troughs=troughs, prediction=False
            )
            self.initial_plot_ready.emit(fig_html_initial)
            (peak_model, peak_scaler, peak_selector, peak_selected_features, all_features_peak, peak_best_score,
             peak_metrics, peak_threshold,
             trough_model, trough_scaler, trough_selector, trough_selected_features, all_features_trough,
             trough_best_score, trough_metrics, trough_threshold) = train_model(
                 df_preprocessed, self.N, all_features, self.classifier_name, self.mixture_depth, 
                 self.n_features_selected, self.oversample_method
            )
            self.training_finished.emit(
                fig_html_initial,
                peak_model, peak_scaler, peak_selector, peak_selected_features, all_features_peak, peak_best_score, peak_metrics, peak_threshold,
                trough_model, trough_scaler, trough_selector, trough_selected_features, all_features_trough, trough_best_score, trough_metrics, trough_threshold
            )
        except Exception as e:
            self.error.emit(str(e))

from itertools import product

class PredictWorker(QThread):
    prediction_finished = pyqtSignal(str, pd.DataFrame)
    error = pyqtSignal(str)
    
    def __init__(self, file_path, stock_code, train_start_date, train_end_date, start_new_date, end_new_date, N,
                 classifier_name, mixture_depth, n_features_selected, oversample_method):
        super().__init__()
        self.file_path = file_path
        self.stock_code = stock_code
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.start_new_date = start_new_date
        self.end_new_date = end_new_date
        self.N = N
        self.classifier_name = classifier_name
        self.mixture_depth = mixture_depth
        self.n_features_selected = n_features_selected
        self.oversample_method = oversample_method
        
    def run(self):
        try:
            stock_code_tdx = self.stock_code[-2:].lower() + self.stock_code[:6]
            best_excess_return = -np.inf
            best_models = None
            peak_models = []
            trough_models = []

            # 重新训练10次，分别保存高点和低点模型
            for i in range(10):
                print(f"重新训练第 {i+1} 次...")

                # 读取训练数据
                train_data = read_day_fromtdx(self.file_path, stock_code_tdx)
                training_df = select_time(train_data, self.train_start_date, self.train_end_date)
                if training_df.empty:
                    raise ValueError("训练集为空，请检查训练日期范围和数据文件")

                # 数据预处理
                df_preprocessed_train, all_features = preprocess_data(training_df, self.N, mixture_depth=self.mixture_depth, mark_labels=True)
                
                # 训练模型
                result_tuple = train_model(df_preprocessed_train, self.N, all_features, self.classifier_name,
                                           self.mixture_depth, self.n_features_selected, self.oversample_method, window_size=10)
                (peak_model, peak_scaler, peak_selector, peak_selected_features, all_features_peak, peak_best_score,
                 peak_metrics, peak_threshold,
                 trough_model, trough_scaler, trough_selector, trough_selected_features, all_features_trough,
                 trough_best_score, trough_metrics, trough_threshold) = result_tuple

                # 将高点和低点模型分别保存
                peak_models.append((peak_model, peak_scaler, peak_selector, peak_selected_features, all_features_peak, peak_threshold))
                trough_models.append((trough_model, trough_scaler, trough_selector, trough_selected_features, all_features_trough, trough_threshold))

            print("开始生成模型笛卡尔积...")

            # 生成笛卡尔积，得到 100 种模型组合
            model_combinations = list(product(peak_models, trough_models))

            best_excess_return = -np.inf
            best_combination = None

            # 回测每个组合并选择超额收益率最高的
            for peak_model, trough_model in model_combinations:
                peak_model_data = peak_model
                trough_model_data = trough_model

                # 使用当前模型组合进行回测
                pred_data = read_day_fromtdx(self.file_path, stock_code_tdx)
                new_df = select_time(pred_data, self.start_new_date, self.end_new_date)
                if new_df.empty:
                    raise ValueError("预测集为空，请检查预测日期范围和数据文件")

                # 执行回测
                result_eval, eval_bt = predict_new_data(new_df, peak_model_data[0], peak_model_data[1], peak_model_data[2], peak_model_data[4], peak_model_data[5],
                                                       trough_model_data[0], trough_model_data[1], trough_model_data[2], trough_model_data[4], trough_model_data[5],
                                                       self.N, self.mixture_depth, window_size=10, eval_mode=True)

                # 比较超额收益率
                excess_return = eval_bt.get('超额收益率', -np.inf)
                print(f"模型组合超额收益率: {excess_return:.4f}")

                if excess_return > best_excess_return:
                    best_excess_return = excess_return
                    best_combination = (peak_model_data, trough_model_data)

            print(f"最佳超额收益率: {best_excess_return:.4f}")

            if best_combination is None:
                raise ValueError("未能找到最佳模型组合")

            # 获取最佳模型组合
            best_peak_model_data, best_trough_model_data = best_combination

            # 使用最佳组合进行最终预测
            final_result, final_bt_result = predict_new_data(new_df, best_peak_model_data[0], best_peak_model_data[1], best_peak_model_data[2], best_peak_model_data[4], best_peak_model_data[5],
                                            best_trough_model_data[0], best_trough_model_data[1], best_trough_model_data[2], best_trough_model_data[4], best_trough_model_data[5],
                                            self.N, self.mixture_depth, window_size=10, eval_mode=False)

            plot_worker = PlotWorker(
                final_result, self.stock_code, self.start_new_date, self.end_new_date,
                title="预测结果K线图",
                peaks=final_result[final_result['Peak_Prediction'] == 1],
                troughs=final_result[final_result['Trough_Prediction'] == 1],
                prediction=True, selected_classifiers=[self.classifier_name],
                bt_result=final_bt_result   # 将回测结果传递给图表
            )
            fig_html = plot_worker.plot_candlestick_plotly(final_result, self.stock_code, self.start_new_date, self.end_new_date,
                                                           peaks=final_result[final_result['Peak_Prediction'] == 1],
                                                           troughs=final_result[final_result['Trough_Prediction'] == 1],
                                                           prediction=True)
            self.prediction_finished.emit(fig_html, final_result)
        except Exception as e:
            self.error.emit(str(e))



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("指数局部高低点预测")
        self.setGeometry(100, 100, 1800, 1000)
        self.peak_model = None
        self.peak_scaler = None
        self.peak_selector = None
        self.peak_selected_features = None
        self.trough_model = None
        self.trough_scaler = None
        self.trough_selector = None
        self.trough_selected_features = None
        self.all_features_peak = None
        self.all_features_trough = None
        self.peak_threshold = 0.5
        self.trough_threshold = 0.5
        self.all_features = None
        self.classifier_name = None
        self.initUI()
    def initUI(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f7f7f7; color: #333333; font-family: Segoe UI, Arial, sans-serif; font-size: 14px; }
            QPushButton { background-color: #0078D4; border: none; color: white; padding: 8px 16px; font-size: 14px; border-radius: 5px; }
            QPushButton:hover { background-color: #0063B1; }
            QPushButton:pressed { background-color: #00508C; }
            QGroupBox { border: 1px solid #B6B6B6; border-radius: 5px; margin-top: 10px; background-color: #FFFFFF; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; color: #4C4C4C; font-weight: bold; }
            QLabel { color: #333333; }
            QLineEdit, QSpinBox, QDateEdit, QComboBox { background-color: #FFFFFF; border: 1px solid #C8C8C8; border-radius: 4px; padding: 4px; color: #333333; }
            QScrollBar:horizontal, QScrollBar:vertical { border: 1px solid #C8C8C8; background: #F1F1F1; width: 10px; height: 10px; border-radius: 5px; }
            QScrollBar::handle:horizontal, QScrollBar::handle:vertical { background: #B6B6B6; border-radius: 5px; }
            QScrollBar::handle:horizontal:hover, QScrollBar::handle:vertical:hover { background: #A8A8A8; }
            QScrollBar::add-line:horizontal, QScrollBar::add-line:vertical { background: none; }
            QScrollBar::sub-line:horizontal, QScrollBar::sub-line:vertical { background: none; }
        """)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 1)
        control_panel = QVBoxLayout()
        control_panel.setSpacing(15)
        left_layout.addLayout(control_panel)
        file_group = QGroupBox("数据选择")
        file_layout = QHBoxLayout()
        file_group.setLayout(file_layout)
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("请选择通达信安装目录，例如：C:/TDX")
        btn_select_folder = QPushButton("选择")
        btn_select_folder.setToolTip("点击选择通达信安装目录")
        btn_select_folder.setIcon(QtGui.QIcon.fromTheme("folder"))
        btn_select_folder.clicked.connect(self.select_folder)
        file_layout.addWidget(btn_select_folder)
        file_layout.addWidget(self.file_path_edit)
        control_panel.addWidget(file_group)
        stock_group = QGroupBox("股票信息")
        stock_layout = QGridLayout()
        stock_group.setLayout(stock_layout)
        lbl_stock_code = QLabel("股票代码：")
        self.stock_code_edit = QLineEdit()
        self.stock_code_edit.setPlaceholderText("例如：000001.SH")
        lbl_N = QLabel("窗口长度 N：")
        self.N_spin = QSpinBox()
        self.N_spin.setRange(1, 1000000)
        self.N_spin.setValue(30)
        stock_layout.addWidget(lbl_stock_code, 0, 0)
        stock_layout.addWidget(self.stock_code_edit, 0, 1)
        stock_layout.addWidget(lbl_N, 1, 0)
        stock_layout.addWidget(self.N_spin, 1, 1)
        control_panel.addWidget(stock_group)
        classifier_group = QGroupBox("选择基础模型")
        classifier_layout = QVBoxLayout()
        classifier_group.setLayout(classifier_layout)
        self.classifier_combo = QComboBox()
        available_classifiers = ['Transformer', 'MLP']
        self.classifier_combo.addItems(available_classifiers)
        classifier_layout.addWidget(self.classifier_combo)
        control_panel.addWidget(classifier_group)
        oversample_group = QGroupBox("选择处理类别不均衡的方法")
        oversample_layout = QHBoxLayout()
        oversample_group.setLayout(oversample_layout)
        self.oversample_combo = QComboBox()
        oversample_methods = ['None', 'SMOTE', 'ADASYN', 'Borderline-SMOTE', 'SMOTEENN', 'SMOTETomek', 'Class Weights']
        self.oversample_combo.addItems(oversample_methods)
        self.oversample_combo.setToolTip("选择用于处理类别不均衡的方法：过采样或类别权重")
        oversample_layout.addWidget(self.oversample_combo)
        control_panel.addWidget(oversample_group)
        mixture_group = QGroupBox("因子混合深度")
        mixture_layout = QHBoxLayout()
        mixture_group.setLayout(mixture_layout)
        lbl_mixture_depth = QLabel("选择混合深度 (1-3)：")
        self.mixture_depth_spin = QSpinBox()
        self.mixture_depth_spin.setRange(1, 5)
        self.mixture_depth_spin.setValue(1)
        mixture_layout.addWidget(lbl_mixture_depth)
        mixture_layout.addWidget(self.mixture_depth_spin)
        self.factor_count_label = QLabel("生成因子数量：N/A")
        mixture_layout.addWidget(self.factor_count_label)
        control_panel.addWidget(mixture_group)
        feature_selection_group = QGroupBox("重要特征选择")
        feature_selection_layout = QHBoxLayout()
        feature_selection_group.setLayout(feature_selection_layout)
        self.auto_feature_checkbox = QtWidgets.QCheckBox("自动选择特征数量")
        self.auto_feature_checkbox.setChecked(True)
        self.auto_feature_checkbox.stateChanged.connect(self.toggle_feature_selection)
        lbl_n_features = QLabel("选择特征数量：")
        self.n_features_spin = QSpinBox()
        self.n_features_spin.setRange(1, 1000)
        self.n_features_spin.setValue(20)
        self.n_features_spin.setEnabled(False)
        feature_selection_layout.addWidget(self.auto_feature_checkbox)
        feature_selection_layout.addWidget(lbl_n_features)
        feature_selection_layout.addWidget(self.n_features_spin)
        control_panel.addWidget(feature_selection_group)
        train_group = QGroupBox("训练模型")
        train_layout = QGridLayout()
        train_group.setLayout(train_layout)
        lbl_start_date = QLabel("开始日期：")
        self.start_date_edit = QDateEdit(calendarPopup=True)
        self.start_date_edit.setDisplayFormat('yyyyMMdd')
        self.start_date_edit.setDate(datetime.strptime("20000101", "%Y%m%d"))
        lbl_end_date = QLabel("结束日期：")
        self.end_date_edit = QDateEdit(calendarPopup=True)
        self.end_date_edit.setDisplayFormat('yyyyMMdd')
        self.end_date_edit.setDate(datetime.strptime("20201231", "%Y%m%d"))
        btn_train = QPushButton("训练模型")
        btn_train.setIcon(QtGui.QIcon.fromTheme("media-playback-start"))
        btn_train.clicked.connect(self.start_training)
        self.train_status_label = QLabel("状态：未训练")
        self.best_score_label = QLabel("最佳得分：N/A")
        self.metrics_labels = {}
        metrics = ['ROC AUC', 'PR AUC', 'Precision', 'Recall', 'MCC']
        for i, metric in enumerate(metrics, start=5):
            lbl_metric = QLabel(f"{metric}：")
            lbl_value = QLabel("N/A")
            self.metrics_labels[metric] = lbl_value
            train_layout.addWidget(lbl_metric, i, 0)
            train_layout.addWidget(lbl_value, i, 1)
        train_layout.addWidget(lbl_start_date, 0, 0)
        train_layout.addWidget(self.start_date_edit, 0, 1)
        train_layout.addWidget(lbl_end_date, 1, 0)
        train_layout.addWidget(self.end_date_edit, 1, 1)
        train_layout.addWidget(btn_train, 2, 0, 1, 2)
        train_layout.addWidget(self.train_status_label, 3, 0, 1, 2)
        train_layout.addWidget(self.best_score_label, 4, 0, 1, 2)
        control_panel.addWidget(train_group)
        predict_group = QGroupBox("调用模型进行预测")
        predict_layout = QGridLayout()
        predict_group.setLayout(predict_layout)
        lbl_start_new_date = QLabel("开始日期：")
        self.start_new_date_edit = QDateEdit(calendarPopup=True)
        self.start_new_date_edit.setDisplayFormat('yyyyMMdd')
        lbl_end_new_date = QLabel("结束日期：")
        self.end_new_date_edit = QDateEdit(calendarPopup=True)
        self.end_new_date_edit.setDisplayFormat('yyyyMMdd')
        btn_predict = QPushButton("开始预测")
        btn_predict.setIcon(QtGui.QIcon.fromTheme("media-playback-start"))
        btn_predict.clicked.connect(self.start_prediction)
        predict_layout.addWidget(lbl_start_new_date, 0, 0)
        predict_layout.addWidget(self.start_new_date_edit, 0, 1)
        predict_layout.addWidget(lbl_end_new_date, 1, 0)
        predict_layout.addWidget(self.end_new_date_edit, 1, 1)
        predict_layout.addWidget(btn_predict, 2, 0, 1, 2)
        control_panel.addWidget(predict_group)
        result_group = QGroupBox("预测结果（显示预测的高低点）")
        result_layout = QVBoxLayout()
        result_group.setLayout(result_layout)
        self.result_table = QtWidgets.QTableWidget()
        self.result_table.setColumnCount(5)
        self.result_table.setHorizontalHeaderLabels(["交易日期", "高点", "高点概率", "低点", "低点概率"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.setStyleSheet("""
            QTableWidget { gridline-color: #cccccc; }
            QHeaderView::section { background-color: #e9ecef; color: #333333; padding: 4px; border: 1px solid #cccccc; }
        """)
        self.result_table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.result_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        result_layout.addWidget(self.result_table)
        left_layout.addWidget(result_group)
        chart_group = QGroupBox("K线图与成交量")
        chart_layout = QVBoxLayout()
        chart_group.setLayout(chart_layout)
        self.web_view = QWebEngineView()
        chart_layout.addWidget(self.web_view)
        main_layout.addWidget(chart_group, 3)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 3)
    def toggle_feature_selection(self):
        if self.auto_feature_checkbox.isChecked():
            self.n_features_spin.setEnabled(False)
        else:
            self.n_features_spin.setEnabled(True)
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择数据文件夹")
        if folder:
            self.file_path_edit.setText(folder)
            print(f"选择的数据文件夹路径: {folder}")
    def start_training(self):
        file_path = self.file_path_edit.text()
        stock_code = self.stock_code_edit.text().strip()
        start_date = self.start_date_edit.date().toString("yyyyMMdd")
        end_date = self.end_date_edit.date().toString("yyyyMMdd")
        N = self.N_spin.value()
        mixture_depth = self.mixture_depth_spin.value()
        classifier_name = self.classifier_combo.currentText()
        self.classifier_name = classifier_name
        if self.auto_feature_checkbox.isChecked():
            n_features_selected = 'auto'
        else:
            n_features_selected = self.n_features_spin.value()
        oversample_method = self.oversample_combo.currentText()
        if not all([file_path, stock_code, start_date, end_date, N]):
            QMessageBox.critical(self, "错误", "请填写所有训练参数")
            return
        try:
            datetime.strptime(start_date, '%Y%m%d')
            datetime.strptime(end_date, '%Y%m%d')
        except ValueError:
            QMessageBox.critical(self, "错误", "日期格式应为YYYYMMDD")
            return
        if start_date > end_date:
            QMessageBox.critical(self, "错误", "开始日期不能晚于结束日期")
            return
        self.thread = TrainWorker(
            file_path, stock_code, start_date, end_date, N, classifier_name, mixture_depth, n_features_selected,
            oversample_method
        )
        self.thread.initial_plot_ready.connect(self.on_initial_plot_ready)
        self.thread.training_finished.connect(self.on_training_finished)
        self.thread.error.connect(self.on_training_error)
        self.thread.start()
        self.train_status_label.setText("状态：正在训练，请稍候...")
        self.best_score_label.setText("最佳得分：N/A")
        for metric in self.metrics_labels:
            self.metrics_labels[metric].setText("N/A")
        self.result_table.setRowCount(0)
    def on_initial_plot_ready(self, fig_html):
        self.web_view.setHtml(fig_html)
        print("初始K线图已显示。")
    def on_training_finished(self, fig_html_initial,
                         peak_model, peak_scaler, peak_selector, peak_selected_features, all_features_peak, peak_best_score,
                         peak_metrics, peak_threshold,
                         trough_model, trough_scaler, trough_selector, trough_selected_features, all_features_trough,
                         trough_best_score, trough_metrics, trough_threshold):
        self.train_status_label.setText("状态：训练完成")
        self.best_score_label.setText(f"Peak最佳得分：{peak_best_score:.4f} | Trough最佳得分：{trough_best_score:.4f}")
        for metric, value in peak_metrics.items():
            if metric in self.metrics_labels:
                self.metrics_labels[metric].setText(f"{metric}: {value:.4f}")
        for metric, value in trough_metrics.items():
            if metric in self.metrics_labels:
                self.metrics_labels[metric].setText(f"{metric}: {value:.4f}")
        self.peak_model = peak_model
        self.peak_scaler = peak_scaler
        self.peak_selector = peak_selector
        self.peak_selected_features = peak_selected_features
        self.all_features_peak = all_features_peak
        self.peak_threshold = peak_threshold
        self.trough_model = trough_model
        self.trough_scaler = trough_scaler
        self.trough_selector = trough_selector
        self.trough_selected_features = trough_selected_features
        self.all_features_trough = all_features_trough
        self.trough_threshold = trough_threshold
        factor_count = len(all_features_peak)
        self.factor_count_label.setText(f"生成因子数量：{factor_count}")
        print("训练模型完成。")
    def on_training_error(self, error_msg):
        self.train_status_label.setText("状态：训练失败")
        QMessageBox.critical(self, "错误", f"训练模型失败：{error_msg}")
        print(f"训练模型失败：{error_msg}")
    def start_prediction(self):
        file_path = self.file_path_edit.text()
        stock_code = self.stock_code_edit.text().strip()
        start_new_date = self.start_new_date_edit.date().toString("yyyyMMdd")
        end_new_date = self.end_new_date_edit.date().toString("yyyyMMdd")
        N = self.N_spin.value()

        if not all([file_path, stock_code, start_new_date, end_new_date, N]):
            QMessageBox.critical(self, "错误", "请填写所有预测参数")
            print("预测参数不完整。")
            return

        try:
            datetime.strptime(start_new_date, '%Y%m%d')
            datetime.strptime(end_new_date, '%Y%m%d')
        except ValueError:
            QMessageBox.critical(self, "错误", "日期格式应为YYYYMMDD")
            print("日期格式错误。")
            return

        if start_new_date > end_new_date:
            QMessageBox.critical(self, "错误", "开始日期不能晚于结束日期")
            print("开始日期晚于结束日期。")
            return

        # 采用训练界面中设置的训练日期范围
        train_start_date = self.start_date_edit.date().toString("yyyyMMdd")
        train_end_date = self.end_date_edit.date().toString("yyyyMMdd")
        if not all([train_start_date, train_end_date]):
            QMessageBox.critical(self, "错误", "训练日期未设置")
            return

        if self.auto_feature_checkbox.isChecked():
            n_features_selected = 'auto'
        else:
            n_features_selected = self.n_features_spin.value()

        oversample_method = self.oversample_combo.currentText()

        self.train_status_label.setText("状态：正在预测并重新训练，请稍候...")
        self.thread = PredictWorker(file_path, stock_code, train_start_date, train_end_date,
                                    start_new_date, end_new_date, N,
                                    self.classifier_name, self.mixture_depth_spin.value(),
                                    n_features_selected, oversample_method)
        self.thread.prediction_finished.connect(self.on_prediction_finished)
        self.thread.error.connect(self.on_prediction_error)
        self.thread.start()
    def on_prediction_finished(self, fig_html, result_df):
        self.train_status_label.setText("状态：预测完成")
        self.web_view.setHtml(fig_html)
        self.display_result(result_df)
        print("预测完成。")
    def on_prediction_error(self, error_msg):
        self.train_status_label.setText("状态：预测失败")
        QMessageBox.critical(self, "错误", f"预测失败：{error_msg}")
        print(f"预测失败：{error_msg}")
    def display_result(self, result):
        filtered_result = result[(result['Peak_Prediction'] == 1) | (result['Trough_Prediction'] == 1)]
        if filtered_result.empty:
            QMessageBox.information(self, "信息", "没有预测到高点或低点。")
            self.result_table.setRowCount(0)
            return
        result_table = filtered_result[['TradeDate', 'Peak_Prediction', 'Peak_Probability',
                                        'Trough_Prediction', 'Trough_Probability']].copy()
        self.result_table.setRowCount(0)
        self.result_table.setRowCount(len(result_table))
        for row_idx, (_, row) in enumerate(result_table.iterrows()):
            date_item = QtWidgets.QTableWidgetItem(row['TradeDate'])
            peak_pred_item = QtWidgets.QTableWidgetItem(str(int(row['Peak_Prediction'])))
            peak_prob_item = QtWidgets.QTableWidgetItem(f"{row['Peak_Probability']:.4f}")
            trough_pred_item = QtWidgets.QTableWidgetItem(str(int(row['Trough_Prediction'])))
            trough_prob_item = QtWidgets.QTableWidgetItem(f"{row['Trough_Probability']:.4f}")
            date_item.setBackground(QtGui.QColor("#d1e7dd"))
            if row['Peak_Prediction'] == 1:
                peak_pred_item.setBackground(QtGui.QColor("#fff3cd"))
                peak_prob_item.setBackground(QtGui.QColor("#fff3cd"))
            if row['Trough_Prediction'] == 1:
                trough_pred_item.setBackground(QtGui.QColor("#f8d7da"))
                trough_prob_item.setBackground(QtGui.QColor("#f8d7da"))
            self.result_table.setItem(row_idx, 0, date_item)
            self.result_table.setItem(row_idx, 1, peak_pred_item)
            self.result_table.setItem(row_idx, 2, peak_prob_item)
            self.result_table.setItem(row_idx, 3, trough_pred_item)
            self.result_table.setItem(row_idx, 4, trough_prob_item)
        print("预测结果已插入表格。")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
