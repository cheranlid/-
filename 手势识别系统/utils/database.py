# -*- coding: utf-8 -*-
"""
=============================================
FileName: database.py
Author: 查诚
StudentID: 20211210004
School: 淮北师范大学
College: 计算机科学与技术学院
Major: 智能科学与技术专业
Supervisor: 李晓
Description:
    手势识别系统的数据库管理工具，功能包括：
    1. 数据预处理：处理空值、标准化边界框(bbox)格式
    2. 数据存储：将检测结果保存至SQLite数据库
    3. 类型安全：强制字段类型约束
    4. 错误处理：捕获数据库操作异常
=============================================
"""

import sqlite3
import pandas as pd


class DatabaseManager:
    """数据库管理类，整合数据预处理和存储功能"""

    def __init__(self, db_path):
        self.db_path = db_path
        self.column_types = {
            "frame_num": 'INTEGER',
            'class_name': 'TEXT',
            'confidence': 'TEXT',
            'bbox_x1': 'REAL',
            'bbox_y1': 'REAL',
            'bbox_x2': 'REAL',
            'bbox_y2': 'REAL'
        }

    def _preprocess_data(self, sql_df):

        # 处理默认值
        sql_df['class_name'] = sql_df['class_name'].fillna("无结果")
        sql_df['confidence'] = sql_df['confidence'].fillna(0.0)
        sql_df['bbox'] = sql_df['bbox'].fillna("[]")
        # 检查并处理bbox列
        if 'bbox' in sql_df.columns:
            # 先过滤掉字符串类型的bbox值（如"[]"）
            bbox_mask = sql_df['bbox'].apply(lambda x: isinstance(x, list) or isinstance(x, tuple))

            # 只对有效的bbox数据进行拆分
            bbox_expanded = sql_df.loc[bbox_mask, 'bbox'].apply(
                lambda x: pd.Series({
                    'bbox_x1': x[0][0],
                    'bbox_y1': x[0][1],
                    'bbox_x2': x[1][0],
                    'bbox_y2': x[1][1]
                })
            )

            # 合并结果，保留原始字符串类型的bbox数据
            if not bbox_expanded.empty:
                sql_df = pd.concat([
                    sql_df.drop(columns=['bbox']),
                    bbox_expanded
                ], axis=1)
            else:
                # 如果没有有效的bbox数据，添加空的坐标列
                for col in ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']:
                    sql_df[col] = None

        # 处理confidence列
        if 'confidence' in sql_df.columns:
            sql_df['confidence'] = sql_df['confidence'].apply(
                lambda x: str(round(x, 2)) if isinstance(x, (int, float)) else str(x)
            )

        return sql_df

    def save(self, sql_df,table_name):
        try:
            processed_df = self._preprocess_data(sql_df)
            with sqlite3.connect(self.db_path) as conn:
                processed_df.to_sql(
                    table_name,
                    conn,
                    if_exists='replace',
                    index=False,
                    dtype=self.column_types
                )
            return True, table_name
        except sqlite3.Error as e:
            return False, f"数据库操作失败: {str(e)}"
        except Exception as e:
            return False, f"保存过程中发生错误: {str(e)}"
