# -*- coding: utf-8 -*-
"""
=============================================
FileName: main.py
Author: 查诚
StudentID: 20211210004
School: 淮北师范大学
College: 计算机科学与技术学院
Major: 智能科学与技术专业
Supervisor: 李晓
Description:
    手势识别系统主程序，功能包括：
    1. 图像/视频文件手势检测
    2. 单目标/多目标手势检测
    3. 手势识别与结果显示
    4. 检测记录存储与管理
=============================================
"""

import sys
import pandas as pd
from datetime import datetime
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QFileDialog, QLabel
from ui.new_MainWindow import Ui_MainWindow
from utils.database import DatabaseManager
from utils.image import *
from utils.log import LogWriter
from model.yolov5_model import YOLOV5_ONNX
from utils.tableview import PandasModelX


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.file=None
        self.video=None
        self.video_rotation=0
        self.file_type=None
        self.current_frame_num = 0
        self.is_running = False
        self.timer=QTimer()
        self.timer.setInterval(16)

        self._setup_data_config()
        self._setup_ui_attributes()
        self._bind_slot()


    def _setup_ui_attributes(self):
        # 设置标题
        self.setWindowTitle('手势识别系统')
        # 表格视图设置
        self.table_model = PandasModelX(self.display_df)
        self.tableView_Result.setModel(self.table_model)
        # 表头策略
        self.tableView_Result.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.tableView_Result.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tableView_Result.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.tableView_Result.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        # 表格显示优化 确保表格填满布局
        self.tableView_Result.verticalHeader().setVisible(False)
        self.tableView_Result.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        # 日志窗口设置
        self.textBrowser_Log.setReadOnly(True)
        self.log_writer = LogWriter(self.textBrowser_Log)


    def _setup_data_config(self):
        # 导入模型
        self.model = YOLOV5_ONNX('model/weights/yolov5n.onnx', ['bent', 'fist', 'pinky', 'pinky_index', 'palm', 'thumb', 'thumb_index'])
        # 初始化数据结构
        self.df_columns = ['frame_num', 'class_name', 'confidence', 'bbox']
        self.full_df = pd.DataFrame(columns=self.df_columns)
        self.display_df = pd.DataFrame(columns=self.df_columns)


    def clear_cache(self):
        """清除上一次的文件、视图等缓存，并安全释放视频捕获对象。"""
        # 清除视图
        self.label_InputView.clear()
        self.label_DetectView.clear()
        # 释放视频捕获对象（如果存在且已打开）
        if self.video is not None:
            self.video.release()
            self.video = None
        # 停止定时器,重置状态
        self.stop_detect()
        self.video_rotation = 0
        self.file = None
        self.file_type = None


    def display_label_view(self, pixmap: QPixmap, label: QLabel):
        """
        将传入的 QPixmap 进行缩放后显示到传入的 QLabel 上
        :param pixmap: 未处理的 QPixmap
        :param label: self.label_DetectView 或 self.label_InputView
        :return: None
        """
        label_size = label.size() # 获取 QLabel 的尺寸
        target_size = label_size.scaled(
            label_size.width() - 10,
            label_size.height() - 10,
            Qt.KeepAspectRatio)# 计算目标尺寸，保持宽高比
        pixmap_view = pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation)# 缩放 pixmap
        label.setPixmap(pixmap_view)# 将缩放后的 pixmap 设置到 QLabel 上


    def load_file(self):
        """加载文件，根据选择的文件类型修改属性，文件路径绑定，源文件显示在预览窗口"""
        self.clear_cache()
        mode=self.comboBox_ModeSelect.currentIndex()
        if mode==0:
            file_path=QFileDialog.getOpenFileName(self,filter='*.jpg;*.jpeg;*.png;*.bmp')
            self.file_type='image'
        else:
            file_path=QFileDialog.getOpenFileName(self, filter='*.mp4')
            self.file_type='video'

        if file_path[0]:
            self.file=file_path[0]
            if self.file_type=='image':
                self.display_label_view(image2QPixmap(self.file), self.label_InputView)
            else:
                self.video_rotation,frame= get_cover_rotation(self.file)
                self.display_label_view(image2QPixmap(rotate_frame(frame,self.video_rotation)), self.label_InputView)
        else:
            self.clear_cache()


    def run_detect(self):
        """点击按钮后开始检测，根据文件类型执行不同的函数"""
        if self.file_type=='image':
            self.log_writer.write([f'当前为图片检测...\n文件路径:{self.file}'],'info')
            self.image_detect()
        elif self.file_type=='video':
            self.pushButton_Run_Stop.setText("结束检测")
            # 设置running属性为true来触发QSS样式变化
            self.pushButton_Run_Stop.setProperty("running", "true")
            self.pushButton_Run_Stop.style().polish(self.pushButton_Run_Stop)  # 刷新样式
            self.is_running = True
            self.log_writer.write([f'当前为视频检测...\n文件路径:{self.file}'],'info')
            self.video = cv2.VideoCapture(self.file)
            self.timer.start()
        else:
            self.log_writer.write(['未选择文件!!!'],'warning')


    def stop_detect(self):
        self.pushButton_Run_Stop.setText("开始检测")
        # 设置running属性为false来恢复默认样式
        self.pushButton_Run_Stop.setProperty("running", "false")
        self.pushButton_Run_Stop.style().polish(self.pushButton_Run_Stop)  # 刷新样式
        self.is_running=False
        self.timer.stop()


    def image_detect(self):
        self.current_frame_num=0
        single_ob = (self.comboBox_DetectOption.currentIndex() == 0)
        result,frame_result= self.model.predict(self.file, single_ob)
        self.display_label_view(ndarray2QPixmap(result), self.label_DetectView)
        self.refresh_table(self.current_frame_num, frame_result)


    def video_detect(self):
        single_ob = (self.comboBox_DetectOption.currentIndex() == 0)
        ret, frame = self.video.read()
        if ret:
            self.current_frame_num += 1  # 更新帧号
            rotated_frame = rotate_frame(frame, self.video_rotation)
            # 显示原始帧
            self.display_label_view(ndarray2QPixmap(rotated_frame), self.label_InputView)
            # 进行模型推理
            result,frame_result = self.model.predict(rotated_frame, single_ob)
            # 显示推理结果
            self.display_label_view(ndarray2QPixmap(result), self.label_DetectView)
            self.refresh_table(self.current_frame_num, frame_result)
        else:
            self.stop_detect()


    def rs_button_clicked(self):
        """开始、停止按钮绑定函数，程序未运行时点击按钮就开始运行，反之程序停止"""
        if self.is_running:
            self.stop_detect()
        else:
            self.run_detect()


    def refresh_table(self,frame_num,frame_result):
        # 处理检测结果
        if not frame_result or frame_result[0]['class_name'] is None:
            new_row = {
                'frame_num': frame_num,
                'class_name': "无结果",  # 替换None为友好提示
                'confidence': 0.0,  # 替换None为默认值
                'bbox': "[]"  # 替换None为字符串
            }
            new_data = pd.DataFrame([new_row])
        else:
            new_rows = []
            for detection in frame_result:
                new_rows.append({
                    'frame_num': frame_num,
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'bbox': detection['bbox']
                })
            new_data = pd.DataFrame(new_rows)
        # 更新数据
        self.update_data(new_data)


    def update_data(self, new_data):
        last_result=new_data.iloc[-1]
        self.full_df = pd.concat([
            self.full_df.dropna(how='all', axis=1),
            new_data.dropna(how='all', axis=1)
        ], ignore_index=True)
        # 更新显示数据（最近100行）
        recent_rows = min(100, len(self.full_df))
        self.display_df = self.full_df.iloc[-recent_rows:].copy()
        # 更新视图
        self.display_current_label(last_result)
        self.table_model.update_data(self.display_df)
        self.tableView_Result.scrollToBottom()


    def clear_record(self):
        # 清除历史记录，清除tableView_Result中的内容
        self.full_df = pd.DataFrame(columns=self.df_columns)
        self.display_df = pd.DataFrame(columns=self.df_columns)
        self.table_model.update_data(self.display_df)
        self.current_frame_num=0

        self.label_OBClass.setText('待检测')
        self.label_OBConfidence.setText('待检测')
        self.label_OBBox.setText('待检测')
        self.clear_cache()

        self.log_writer.write(["已清除所有检测记录"],'success')


    def display_current_label(self,current_data):
        if current_data['confidence']==0.0:
            self.label_OBClass.setText('未检测到目标')
            self.label_OBConfidence.setText('未检测到目标')
            self.label_OBBox.setText('未检测到目标')
        else:
            self.label_OBClass.setText(str(current_data['class_name']))
            self.label_OBConfidence.setText(str(current_data['confidence']))
            self.label_OBBox.setText(str(current_data['bbox']))


    def save_to_db(self):
        if self.full_df.empty:
            self.log_writer.write(['没有数据可保存'],'warning')
            return
        table_name = "Record_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        sql_df = self.full_df.copy()
        db_manager = DatabaseManager("DetectionResults.db")
        success, result = db_manager.save(sql_df, table_name)
        if success:
            self.log_writer.write([f"成功保存 {len(self.full_df.index)} 条数据到 {result}"],'success')
        else:
            self.log_writer.write([result], 'error')


    def export_to_csv(self):
        if self.full_df.empty:
            self.log_writer.write(['没有数据可保存'],'warning')
            return
        # 创建副本避免修改原DataFrame
        csv_df = self.full_df.copy()

        # 处理confidence列（强制保留2位小数）
        # if 'confidence' in csv_df.columns:
        #     csv_df['confidence'] = csv_df['confidence'].apply(
        #         lambda x: f"{float(x):.2f}" if pd.notnull(x) else ""
        #     )
        # 使用round+astype（性能更好但需要确保全是数值）
        if 'confidence' in csv_df.columns:
            csv_df['confidence'] = csv_df['confidence'].round(2).astype(str)
            # 补充处理 .0 的情况 → 0.00
            csv_df['confidence'] = csv_df['confidence'].str.replace(r'\.0$', '.00', regex=True)

        # 处理bbox列
        if 'bbox' in csv_df.columns:
            csv_df['bbox'] = csv_df['bbox'].astype(str)  # 将列表转为字符串

        # 导出为CSV
        csv_file = "Record_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
        csv_df.to_csv(csv_file, index=False, encoding='utf_8_sig')

        self.log_writer.write([f"数据已导出为CSV文件: {csv_file}"],'success')


    def _bind_slot(self):
        self.pushButton_LoadFile.clicked.connect(self.load_file)
        self.pushButton_Run_Stop.clicked.connect(self.rs_button_clicked)
        self.pushButton_ClearRecord.clicked.connect(self.clear_record)
        self.pushButton_SaveToDB.clicked.connect(self.save_to_db)
        self.pushButton_ExportAsCSV.clicked.connect(self.export_to_csv)
        self.timer.timeout.connect(self.video_detect)




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())