# -*- coding: utf-8 -*-
"""
=============================================
Description:
    基于ONNX Runtime的YOLOv5手势识别推理类，功能包括：
    1. 加载ONNX格式的YOLOv5模型，支持CUDA/CPU加速
    2. 执行目标检测并返回手势类别（7种静态手势）
    3. 提供结果可视化接口（原始尺寸或缩放显示）
    4. 支持单目标检测模式（single_ob）
=============================================
"""

import onnxruntime
from utils.model import *


CLASS_LABELS = ['bent', 'fist', 'pinky', 'pinky_index', 'palm', 'thumb', 'thumb_index']


class YOLOV5_ONNX:
    def __init__(self, onnx_path,class_labels):
        self.class_labels = class_labels
        # 创建一个 SessionOptions 对象，用于配置 ONNX Runtime 推理会话的选项
        self.session_options = onnxruntime.SessionOptions()
        #初始化 ONNX Runtime 推理会话，并指定使用的执行提供程序(按顺序 CUDA > CPU)
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, self.session_options,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # 获取ONNX模型的输入和输出节点名称
        self.input_name = self._get_input_name()
        self.output_name = self._get_output_name()
        # 获取输入形状（用于预热）
        self.input_shape = self._get_input_shape()  # 确保输入形状是有效的整数元组
        # 预热模型
        self._warmup()


    # 获取模型输入的名称，并创建字典。 在ONNX Runtime 中进行模型推理时，不能直接将原始图片数据作为输入传递给模型。因为模型期望的输入数据通常具有特定的格式、维度和数据类型，这些要求是由模型的训练和转换过程决定的。get_input_feed方法的作用正是为了准备符合模型要求的输入数据。
    def _get_input_name(self):
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    # 得到onnx 模型输出节点
    def _get_output_name(self):
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def _build_input_dict(self, img_tensor):
        """
        构建模型推理所需的输入数据字典。在 ONNX Runtime 中，推理时需要将输入数据传递给模型的输入节点。
        该方法通过遍历模型的输入节点名称，将输入数据（img_tensor）与每个输入节点名称关联起来，构建一个输入数据字典
        Args:
            img_tensor:经过预处理的图像数据，通常是一个 4 维张量，形状为 (batch_size, channels, height, width)
        Returns:返回值是一个 输入数据字典，其中键是模型的输入节点名称，值是对应的输入数据（即 img_tensor）。这个字典用于在模型推理时指定输入数据
        """
        input_dict = {}
        for name in self.input_name:
            input_dict[name] = img_tensor
        return input_dict

    def _get_input_shape(self):
        """
        获取模型的输入形状，并确保它是一个有效的整数元组。
        如果输入形状是动态的（例如 None 或字符串），则返回一个默认形状（例如 (1, 3, 640, 640)）。
        """
        input_shape = self.onnx_session.get_inputs()[0].shape
        if any(dim is None or isinstance(dim, str) for dim in input_shape):
            # 如果输入形状是动态的，返回一个默认形状
            return 1, 3, 640, 640  # 默认形状，可以根据模型调整
        return tuple(map(int, input_shape))  # 确保所有维度是整数

    def _warmup(self, warmup_iterations=5):
        """
        预热模型。
        :param warmup_iterations: 预热迭代次数，默认为 10 次。
        """
        # 生成虚拟输入数据（与模型输入形状一致）
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        # 运行预热推理
        for i in range(warmup_iterations):
            self.onnx_session.run(self.output_name, {self.input_name[0]: dummy_input})

    def inference(self, img_input):
        """
        对输入图像进行预处理并执行模型推理，最终返回推理结果以及原始图像和预处理后的图像。
        支持输入类型：
        - 图像文件路径（str）
        - NumPy 数组（ndarray），形状为 (height, width, channels)，通道顺序为 BGR（OpenCV 默认格式）
        Args:
            img_input: 图像文件路径或 NumPy 数组。

        Returns:
            pred: 模型的推理结果。
            org_img: 原始图像（NumPy 数组）。
            pad_img: 预处理后的图像（NumPy 数组）。
        """
        # 检查输入类型
        if isinstance(img_input, str):  # 如果输入是文件路径
            org_img = cv2.imread(img_input)  # 读取图像
            if org_img is None:
                raise ValueError(f"无法读取图像文件: {img_input}")
        elif isinstance(img_input, np.ndarray):  # 如果输入是 NumPy 数组
            org_img = img_input
        else:
            raise ValueError("输入必须是文件路径或 NumPy 数组")
        pad_img, r, (dw, dh) = letterbox(org_img, (640, 640))
        img = pad_img[:, :, ::-1].transpose(2, 0, 1)  # 通道转换和转置,BGR2RGB 和 HWC2CHW
        # 归一化
        img = img.astype(dtype=np.float32)
        img /= 255.0
        # 添加批次维度
        img = np.expand_dims(img, axis=0)
        input_feed = self._build_input_dict(img)  #将输入数据与模型的输入节点名称关联，构建输入数据字典
        pred = self.onnx_session.run(None, input_feed)[0]
        return pred, org_img, pad_img

    def predict(self, img_path,single_ob=False):
        output, or_img, pad_img = self.inference(img_path)
        outbox = filter_box(output, conf_thres=0.5, iou_thres=0.2)
        arr,frame_result=draw(pad_img, outbox, CLASS_LABELS,single_ob)
        return arr,frame_result

    def show_detection(self,img_path,single_ob=False,original_scale=False):
        output, or_img, pad_img = self.inference(img_path)
        outbox = filter_box(output, conf_thres=0.5, iou_thres=0.5)
        if original_scale:
            outbox[:, :4] = scale_boxes(pad_img.shape, outbox[:, :4], or_img.shape).round()
            display_img, _ = draw(or_img, outbox,CLASS_LABELS,single_ob)
        else:
            display_img, _ = draw(pad_img, outbox, CLASS_LABELS, single_ob)
        cv2.imshow('Prediction', display_img)
        cv2.waitKey(0)

if __name__ == '__main__':
    onnx_path = 'weights/yolov5n.onnx'
    model = YOLOV5_ONNX(onnx_path,CLASS_LABELS)

    model.show_detection(r"C:\Users\Cheng\Pictures\pic\IMG_20250323_165127.jpg")

