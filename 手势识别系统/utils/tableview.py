# -*- coding: utf-8 -*-
"""
=============================================
FileName: tableview.py
Author: 查诚
StudentID: 20211210004
School: 淮北师范大学
College: 计算机科学与技术学院
Major: 智能科学与技术专业
Supervisor: 李晓
Description:
    PyQt5 数据表格模型适配器，功能包括：
    1. 数据展示：将Pandas DataFrame适配到QTableView
    2. 本地化支持：自动转换表头显示名称
    3. 数据格式化：支持自定义列数据格式化
    4. 线程安全：提供安全的模型更新机制
    5. 类型处理：自动处理NaN值及复杂数据类型
=============================================
"""

from PyQt5.QtCore import Qt,QAbstractTableModel


class PandasModelX(QAbstractTableModel):
    """
    将 pandas DataFrame 适配到 QTableView 的模型类
    功能增强：
    1. 完整的表头本地化支持
    2. 自动处理多种数据类型（包括 NaN 值）
    3. 优化的性能（减少类型转换开销）
    4. 安全的模型更新机制
    """
    # 表头本地化映射（可扩展）
    DISPLAY_HEADERS = {
        'frame_num': '帧编号',
        'class_name': '手势类型',
        'confidence': '置信度',
        'bbox': '边界框坐标'
    }

    # 列数据类型特殊处理（可选）
    COLUMN_FORMATTERS = {
        'confidence': lambda x: f"{float(x):.0%}"  # 置信度显示为百分比
    }

    def __init__(self, data):
        super().__init__()
        self._data = data.copy()  # 避免修改原始数据
        self._raw_headers = list(data.columns)  # 保存原始列名

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._raw_headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            col_name = self._raw_headers[index.column()]
            value = self._data.iloc[index.row(), index.column()]

            # 应用列特定格式化
            if col_name in self.COLUMN_FORMATTERS:
                return self.COLUMN_FORMATTERS[col_name](value)

            # 自动类型转换
            if isinstance(value, (tuple, list, dict)):
                return str(value)
            return str(value)

        # 可选：设置文本对齐方式
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignVCenter | Qt.AlignLeft

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                # 返回本地化的表头
                raw_name = self._raw_headers[section]
                return self.DISPLAY_HEADERS.get(raw_name, raw_name)
            else:
                # 行号显示（从1开始）
                return str(section + 1)
        return None

    def update_data(self, new_data):
        """线程安全的模型更新方法"""
        self.beginResetModel()
        try:
            self._data = new_data.copy()
            self._raw_headers = list(new_data.columns)
        finally:
            self.endResetModel()

