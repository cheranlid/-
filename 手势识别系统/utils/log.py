# -*- coding: utf-8 -*-
"""
=============================================
Description:
    PyQt5 日志输出工具，功能包括：
    1. 多级日志显示：支持 info/success/warning/error/debug 分级
    2. 彩色格式化输出：不同级别自动匹配颜色样式
    3. 自动时间戳：每条日志添加 [hh:mm:ss] 前缀
    4. 批量写入：支持多行日志同时输出
    5. 自动滚动：始终显示最新日志内容
=============================================
"""

from PyQt5.QtWidgets import QTextBrowser
from PyQt5.QtGui import QTextCharFormat, QColor, QTextCursor
from PyQt5.QtCore import QDateTime


class LogWriter:
    def __init__(self, text_browser: QTextBrowser):
        self.text_browser = text_browser
        self._setup_formats()

    def _setup_formats(self):
        """预定义不同日志级别的文本格式"""
        self.formats = {
            'info': self._create_format(QColor(0, 0, 0)),      # 黑色
            'success': self._create_format(QColor(0, 128, 0)), # 绿色
            'warning': self._create_format(QColor(200, 100, 0)),  # 橙色
            'error': self._create_format(QColor(255, 0, 0)),   # 红色
            'debug': self._create_format(QColor(100, 100, 100)),  # 灰色
            'timestamp': self._create_format(QColor(100, 100, 150))  # 时间戳颜色
        }

    def _create_format(self, color):
        """创建文本格式"""
        fmt = QTextCharFormat()
        fmt.setForeground(color)
        return fmt

    def _get_current_time(self):
        """获取当前时间（仅 hh:mm:ss 格式）"""
        return QDateTime.currentDateTime().toString("[hh:mm:ss]")

    def write(self, logs: list, level='info'):
        """写入日志（支持多行）"""
        cursor = self.text_browser.textCursor()
        cursor.movePosition(QTextCursor.End)

        # 写入时间戳
        timestamp = self._get_current_time()
        cursor.setCharFormat(self.formats['timestamp'])
        cursor.insertText(timestamp + " ")

        # 应用日志级别格式
        format_to_apply = self.formats.get(level, self.formats['info'])
        cursor.setCharFormat(format_to_apply)

        # 插入日志内容
        full_text = '\n'.join(logs) + '\n'
        cursor.insertText(full_text)

        # 自动滚动到底部
        self.text_ensure_visible()

    def text_ensure_visible(self):
        """确保文本可见"""
        self.text_browser.ensureCursorVisible()
        scrollbar = self.text_browser.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

