# -*- coding: utf-8 -*-
"""
=============================================
Description:
    目标检测边界框处理工具集，功能包括：
    1. 坐标格式转换：xywh<->xyxy格式互转
    2. 边界框裁剪：确保坐标不超出图像范围
    3. 非极大值抑制：优化版NMS实现
    4. 结果过滤：置信度阈值和IoU阈值过滤
    5. 可视化绘制：支持单目标/多目标模式
    6. 图像缩放：保持长宽比的letterbox处理
    7. 坐标映射：缩放图像坐标到原始图像空间
=============================================
"""

import cv2
import numpy as np


def clip_boxes(boxes, shape):
    """
    裁剪边界框坐标，确保边界框的坐标不超过图像的尺寸范围
    Args:
        boxes:边界框坐标，格式为 (N, 4)，其中 N 是边界框的数量，每个边界框用 [x1, y1, x2, y2]
        shape:图像的尺寸，格式为 (height, width)
    Returns:
    """
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  #裁剪 x 坐标
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  #裁剪 y 坐标


def xywh2xyxy(boxes):
    """
    将边界框的表示格式从 [中心点坐标 + 宽高]（即 [x, y, w, h]）转换为 [左上角坐标 + 右下角坐标]（即 [x1, y1, x2, y2]）
    Args:
        boxes:一个形状为 (N, 4) 的 NumPy 数组，其中 N 是边界框的数量，每个边界框用 [x, y, w, h],[中心点坐标 + 宽高]
    Returns:一个形状为 (N, 4) 的 NumPy 数组，每个边界框用 [x1, y1, x2, y2],[左上角坐标 + 右下角坐标]
    """
    converted_boxes = np.copy(boxes)
    converted_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted_boxes


def nms(boxes, iou_thres):
    """
    优化后的NMS实现 (单类别)，使用向量化计算IoU
    Args:
        boxes: [N,5], 格式为 [x1,y1,x2,y2,score]
        iou_thres: IoU阈值
    Returns:
        list: 保留的框索引
    """
    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # 按分数降序

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算IoU (向量化)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留IoU低于阈值的框
        order = order[np.where(iou <= iou_thres)[0] + 1]
    return keep


def filter_box(org_box, conf_thres=0.25, iou_thres=0.45):
    """
    优化后的边界框过滤函数，支持批量处理和高效率NMS。
    Args:
        org_box: 输入边界框，形状为 (batch_size, N, 6) 或 (N, 6)，格式为 [x, y, w, h, score, class_probs]
        conf_thres: 置信度阈值 (默认: 0.25)
        iou_thres: IoU阈值 (默认: 0.45)
    Returns:
        np.ndarray: 过滤后的边界框，形状为 (M, 6)，格式为 [x1, y1, x2, y2, score, class]
    """
    # 1. 输入维度处理 (支持单张/批量输入)
    org_box = np.squeeze(org_box)
    if org_box.ndim == 1:
        org_box = np.expand_dims(org_box, axis=0)
    assert org_box.ndim == 2, f"输入维度错误，应为 (N,6)，实际是 {org_box.shape}"

    # 2. 置信度过滤 (向量化操作)
    conf_mask = org_box[:, 4] >= conf_thres
    box = org_box[conf_mask]
    if len(box) == 0:
        return np.empty((0, 6))

    # 3. 类别处理 (向量化替代循环)
    cls_probs = box[:, 5:]
    cls_ids = np.argmax(cls_probs, axis=1)
    box[:, 5] = cls_ids  # 用类别ID替换概率

    # 4. 坐标转换 (xywhxy)
    box_xyxy = np.empty_like(box)
    box_xyxy[:, :2] = box[:, :2] - box[:, 2:4] / 2  # x1, y1
    box_xyxy[:, 2:4] = box[:, :2] + box[:, 2:4] / 2  # x2, y2
    box_xyxy[:, 4:] = box[:, 4:]  # 保留 score 和 class

    # 5. 按类别分组 + NMS (优化实现)
    output = []
    for cls_id in np.unique(cls_ids):
        cls_mask = cls_ids == cls_id
        curr_boxes = box_xyxy[cls_mask]
        if len(curr_boxes) == 0:
            continue

        # 优化后的NMS (单类别)
        keep = nms(curr_boxes[:, :5], iou_thres)  # 输入格式: [x1,y1,x2,y2,score]
        output.extend(curr_boxes[keep])

    return np.array(output) if output else np.empty((0, 6))


# def draw(image, box_data,label_classes,single_ob=False):
#     """
#     在图像上绘制检测到的边界框和类别标签
#     Args:
#         single_ob:
#         image:原始图像（通常是 NumPy 数组或 OpenCV 图像对象）
#         box_data:检测到的边界框数据，通常是一个形状为 (N, 6) 的 NumPy 数组，其中 N 是边界框的数量，每个边界框用 [x1, y1, x2, y2, score, class] 表示
#         label_classes:类别标签列表，用于将目标检测模型输出的类别索引
#     Returns:
#         image: 绘制了边界框和类别标签的图像。
#         result_logs: 包含检测结果的日志列表。
#     """
#     frame_result=[]
#     if box_data.size == 0:
#         frame_result.append({"class_name": None,"confidence": None,"bbox":None})
#         return image,  frame_result
#     #将边界框坐标转换成整数类型
#     boxes = box_data[..., :4].astype(np.int32)
#     #提取边界框的置信度和类别索引
#     scores = box_data[..., 4]
#     classes = box_data[..., 5].astype(np.int32)
#
#     if not single_ob:
#         # 如果是不是单目标检测，就绘制置所有的类别的框，并返回所有目标结果
#         #遍历每个边界框
#         for box, score, cl in zip(boxes, scores, classes):
#             top, left, right, bottom = box  #提取边界框坐标[x1, y1, x2, y2] 左上角，右下角
#             frame_result.append({"class_name": label_classes[cl],"confidence": round(score,2),"bbox":[(left, top), (right, bottom)]})
#
#             # 在图像上绘制边界框
#             cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 255), 2)    #图片，坐标，检测框颜色，矩形框的线宽（以像素为单位）
#             cv2.putText(image, f'{label_classes[cl]} {score:.2f}',(top, left-7),cv2.FONT_HERSHEY_COMPLEX,0.5, (0, 0, 255), 1,lineType=cv2.LINE_AA)
#         return image, frame_result
#     else:
#         # 如果是单目标检测，就绘制置信度最大的类别的框，并返回置信度最大的目标结果
#         # 单目标检测模式 - 只处理置信度最高的目标
#         max_idx = np.argmax(scores)  # 找到置信度最高的索引
#         box = boxes[max_idx]
#         score = scores[max_idx]
#         cl = classes[max_idx]
#         x1, y1, x2, y2 = box
#
#         # 记录检测结果
#         frame_result.append({
#             "class_name": label_classes[cl],
#             "confidence": round(score, 2),
#             "bbox": [(x1, y1), (x2, y2)]
#         })
#
#         # 绘制边界框和标签（使用不同颜色区分单目标模式）
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 绿色边框
#         cv2.putText(
#             image,
#             f'{label_classes[cl]} {score:.2f} (TOP)',
#             (x1, y1 - 10),
#             cv2.FONT_HERSHEY_COMPLEX,
#             0.6,
#             (0, 255, 0),  # 绿色文字
#             2,
#             lineType=cv2.LINE_AA
#         )
#
#     return image,  frame_result

def draw(image, box_data, label_classes, single_ob=False):
    frame_result = []
    if box_data.size == 0:
        frame_result.append({"class_name": None, "confidence": None, "bbox": None})
        return image, frame_result

    # 检查图像通道并设置颜色
    is_gray = len(image.shape) == 2
    color = 255 if is_gray else ((0, 255, 0) if  single_ob else (255, 0, 255))  # BGR格式

    boxes = box_data[..., :4].astype(np.int32)
    scores = box_data[..., 4]
    classes = box_data[..., 5].astype(np.int32)

    if single_ob:
        max_idx = np.argmax(scores)
        boxes = [boxes[max_idx]]
        scores = [scores[max_idx]]
        classes = [classes[max_idx]]

    for (x1, y1, x2, y2), score, cl in zip(boxes, scores, classes):
        # 边界框裁剪
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[:2][1] - 1, x2), min(image.shape[:2][0] - 1, y2)
        class_name = str(cl) if not label_classes else label_classes[cl]
        frame_result.append({
            "class_name": class_name,
            "confidence": round(score, 2),
            "bbox": [(x1, y1), (x2, y2)]
        })
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f'{class_name} {score:.2f}' + (' (TOP)' if single_ob else '')
        cv2.putText(image, text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return image, frame_result


# def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scale_fill=False, scaleup=True):
#     """
#     实现 图像缩放和填充，用于将输入图像调整为模型所需的固定尺寸new_shape=(640, 640)，同时保持图像的宽高比不变
#     Args:
#         img:输入的原始图像（通常是 NumPy 数组或 OpenCV 图像对象)
#         new_shape:目标图像的尺寸即模型的输入尺寸，默认是 (640, 640)
#         color:填充区域的颜色，默认是 (114, 114, 114)（灰色）
#         auto:是否自动将填充量调整为 32 的倍数，默认是 True，即填充量会被调整为最接近的 32 的倍数
#         scale_fill:是否拉伸图像以完全填充目标尺寸，默认是 False。如果为 True，图像会被拉伸，忽略宽高比
#         scaleup:是否允许放大图像，默认是 True。如果为 False，图像只会缩小，不会放大
#     Returns:
#     """
#     #确保形状是元组
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)
#     shape = img.shape[:2]  #获取原始图像的高度和宽度
#
#     # 计算缩放比例r，确保图像不会超过目标尺寸，如果 scaleup 为 False，则限制缩放比例 r 最大为 1.0（即不允许放大）
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:
#         r = min(r, 1.0)
#
#     # 等比缩放图像 。 round() 将计算出的浮点数尺寸四舍五入为最接近的整数，确保尺寸是整数像素值。
#     new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
#     # 计算定义的模型输入尺寸与图片等比例缩放后的图片尺寸之间的差值，—（即计算图像在宽度和高度方向上需要的填充量（即多余的部分），dw 表示在宽度上的填充，dh 表示在高度上的填充）目的是为了后面把不是正方形的等比例缩放图片变成正方形图片
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
#     if auto:  # 如果自动调整填充量
#         dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # 将填充量调整为 32 的倍数
#     elif scale_fill:  # 如果拉伸图像
#         dw, dh = 0, 0
#         new_unpad = new_shape
#         r = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 计算拉伸比例
#
#     # 将填充量平分到两侧
#     dw /= 2
#     dh /= 2
#
#     # 判断原始图片尺寸是否等于等比例缩放尺寸
#     if shape[::-1] != new_unpad:  # resize
#         interpolation = cv2.INTER_AREA if r < 1 else cv2.INTER_CUBIC
#         img = cv2.resize(img, new_unpad, interpolation=interpolation)
#     # 计算填充区域
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加填充
#     # 返回填充后的图像、缩放比例r以及填充量(dw, dh)
#     return img, r, (dw, dh)

def letterbox(
        img,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=True,
        scale_fill=False,
        scaleup=True
):
    """
    图像缩放填充函数 (保持元组返回值)

    Args:
        img: 输入图像 (H, W, C)
        new_shape: 目标尺寸 (h, w)
        color: 填充颜色 (B, G, R)
        auto: 是否自动对齐到32的倍数
        scale_fill: 是否强制拉伸填充
        scaleup: 是否允许放大图像

    Returns:
        tuple: (处理后的图像, 缩放比例, (水平填充量, 垂直填充量))
    """
    # 输入验证（可选，生产环境推荐添加）
    if not isinstance(img, np.ndarray):
        raise ValueError("输入必须是NumPy数组")
    if len(img.shape) != 3:
        raise ValueError("图像必须是3通道 (H, W, C)")

    # 统一new_shape格式
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = img.shape[:2]
    target_h, target_w = new_shape

    # 强制拉伸模式
    if scale_fill:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return img, (target_w / w, target_h / h), (0, 0)

    # 计算缩放比例
    r = min(target_h / h, target_w / w)
    if not scaleup:
        r = min(r, 1.0) if max(target_h, target_w) > max(h, w) else r

    # 等比缩放
    new_unpad = (max(1, int(round(w * r))), max(1, int(round(h * r))))
    dw, dh = target_w - new_unpad[0], target_h - new_unpad[1]

    # 自动对齐填充
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)

    # 计算四边填充量（保持与原始函数兼容）
    top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
    left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))

    # 缩放图像
    if (w, h) != new_unpad:
        interpolation = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, new_unpad, interpolation=interpolation)

    # 添加填充
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return img, r, (dw, dh)  # 严格保持原始返回格式


def scale_boxes(pad_img_shape, boxes, ori_img_shape, ratio_pad=None):
    """
    将缩放并填充后的图像上的边界框坐标 映射回原始图像的坐标空间
    Args:
        pad_img_shape:缩放并填充后的图像的尺寸，格式为 (height, width)
        boxes:检测到的边界框坐标，格式为 (N, 4)，其中 N 是边界框的数量，每个边界框用 [x1, y1, x2, y2] 表示
        ori_img_shape:原始图像的尺寸，格式为 (height, width)
        ratio_pad:可选参数，包含缩放比例和填充量的元组，格式为 (ratio, pad)，如果为 None，则根据 pad_img_shape 和 ori_img_shape 计算缩放比例和填充量
    Returns:返回调整后的边界框
    """
    # 计算缩放比例和填充量
    if ratio_pad is None:  # 如果没有提供 ratio_pad
        gain = min(pad_img_shape[0] / ori_img_shape[0], pad_img_shape[1] / ori_img_shape[1])
        pad = (pad_img_shape[1] - ori_img_shape[1] * gain) / 2, (pad_img_shape[0] - ori_img_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]  # 使用提供的缩放比例
        pad = ratio_pad[1]  # 使用提供的填充量

    #调整边界框坐标
    boxes[:, [0, 2]] -= pad[0]  # 调整 x 坐标（减去水平填充量）
    boxes[:, [1, 3]] -= pad[1]  # 调整 y 坐标（减去垂直填充量）
    boxes[:, :4] /= gain    # 将坐标缩放到原始图像的尺寸
    clip_boxes(boxes, ori_img_shape)    #裁剪边界框，确保边界框坐标不超过原始图像的尺寸范围
    return boxes
