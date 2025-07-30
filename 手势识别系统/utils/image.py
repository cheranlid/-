# -*- coding: utf-8 -*-
"""
=============================================
FileName: image.py
Author: 查诚
StudentID: 20211210004
School: 淮北师范大学
College: 计算机科学与技术学院
Major: 智能科学与技术专业
Supervisor: 李晓
Description:
    图像处理工具，功能包括：
    1. 图像格式转换：支持 NumPy 数组/QPixmap 互转
    2. 视频处理：提取封面帧、获取旋转信息、自动旋转校正
    3. 色彩空间转换：自动处理 BGR/RGB 格式
    4. 异常处理：完善的错误检测机制
=============================================
"""

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap


def ndarray2QPixmap(arr, is_bgr=True):
    """
    将 NumPy 数组转换为 QPixmap。
    参数:
        arr: NumPy 数组，形状为 (height, width, channels)。
            支持的通道数：3 (RGB/BGR) 或 4 (RGBA/BGRA)。
        is_bgr: 如果为 True，表示输入是 BGR 格式；如果为 False，表示输入是 RGB 格式。
    返回:
        QPixmap 对象。
    """
    # 检查输入是否为 NumPy 数组
    if not isinstance(arr, np.ndarray):
        raise ValueError("输入必须是 NumPy 数组")
    # 检查数组的维度
    if arr.ndim != 3:
        raise ValueError("输入数组必须是 3 维 (height, width, channels)")
    # 检查通道数
    h, w, c = arr.shape
    if c not in [3, 4]:
        raise ValueError("输入数组必须是 3 通道 (RGB/BGR) 或 4 通道 (RGBA/BGRA)")
    # 确保数据是连续的
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    # 将 BGR 转换为 RGB（如果需要）
    if is_bgr and c == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif is_bgr and c == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    # 将 NumPy 数组转换为 QImage
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if c == 3:
        qimage = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    else:
        qimage = QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888)
    # 将 QImage 转换为 QPixmap
    return QPixmap.fromImage(qimage)

def image2QPixmap(image_input):
    """
    将图像输入转换为 QPixmap。
    支持输入类型：
    - 图像文件路径（如 "example.jpg"）
    - NumPy 数组（OpenCV 图像）
    - OpenCV 图像对象
    参数:
        image_input: 图像文件路径、NumPy 数组或 OpenCV 图像对象。
    返回:
        QPixmap 对象。
    """
    # 如果输入是文件路径，读取图像
    if isinstance(image_input, str):
        arr = cv2.imread(image_input)
        if arr is None:
            raise ValueError(f"无法读取图像文件: {image_input}")
    # 如果输入是 NumPy 数组或 OpenCV 图像对象
    elif isinstance(image_input, np.ndarray):
        arr = image_input
    else:
        raise ValueError("输入必须是文件路径、NumPy 数组或 OpenCV 图像对象")
    # 确保图像是 3 通道（RGB）或 4 通道（RGBA）
    if len(arr.shape) != 3 or arr.shape[2] not in [3, 4]:
        raise ValueError("图像必须是 3 通道 (RGB) 或 4 通道 (RGBA)")
    # 将 BGR 转换为 RGB（如果是 OpenCV 图像）
    if arr.shape[2] == 3:  # 3 通道图像
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif arr.shape[2] == 4:  # 4 通道图像
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    # 将 NumPy 数组转换为 QImage
    h, w, c = arr.shape
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if c == 3:
        qimage = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    else:
        qimage = QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888)
    # 将 QImage 转换为 QPixmap
    return QPixmap.fromImage(qimage)



def get_cover_rotation(file):
    """
    获取视频的第一帧作为封面，并返回旋转标志和帧数据。
    Args:
        file (str): 视频文件路径。
    Returns:
        tuple: (rotation_flag, frame)
            - rotation_flag (int): 旋转标志（0、90、180 或 270）。
            - frame (np.ndarray): 视频的第一帧（BGR 格式）。
    Raises:
        RuntimeError: 如果无法打开视频文件或读取第一帧。
    """
    # 打开视频文件
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {file}")
    try:
        # 获取旋转标志
        rotation_flag = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        # 设置视频帧位置为第一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # 读取第一帧
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"无法读取视频文件的第一帧: {file}")
        return rotation_flag, frame
    finally:
        # 释放视频捕获对象
        cap.release()


def get_video_rotation(video_path):
    """
    获取视频的旋转标志。
    Args:
        video_path (str): 视频文件路径。
    Returns:
        int: 旋转标志（0、90、180 或 270）。
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
    # 获取旋转标志
    rotation_flag = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    # 释放视频捕获对象
    cap.release()
    return rotation_flag


def rotate_frame(frame, rotation_flag):
    """
    根据旋转标志旋转视频帧。
    Args:
        frame (np.ndarray): 视频帧（BGR 格式）。
        rotation_flag (int): 旋转标志（0、90、180 或 270）
    Returns:
        np.ndarray: 旋转后的视频帧。
    """
    if rotation_flag == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_flag == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_flag == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return frame
