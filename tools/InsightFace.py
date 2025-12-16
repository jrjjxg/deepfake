import os

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# 模型目录与路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# insightface 会在 root 下自动创建 models/ 子目录，因此把 root 指向工具目录本身
FACE_ROOT = BASE_DIR

# 检查可用的执行器并设置上下文 - 强制使用CPU以避免CUDA问题
providers = ['CPUExecutionProvider']
use_cuda = False  # 强制禁用CUDA
ctx_id = -1  # 使用CPU
print("强制使用 CPU 模式以避免 CUDA 配置问题")

# 初始化FaceAnalysis应用，指向本地模型缓存目录，强制使用CPU
try:
    app = FaceAnalysis(name='buffalo_l', root=FACE_ROOT, providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    print("InsightFace 初始化成功 (CPU 模式)")
except Exception as e:
    print(f"InsightFace 初始化失败: {e}")
    print("请确保已正确安装 insightface 和相关依赖")
    app = None

# 加载换脸模型，并确保文件存在
swapper_path = os.path.join(MODELS_DIR, 'inswapper_128.onnx')
swapper = None

try:
    if not os.path.exists(swapper_path):
        raise FileNotFoundError(f"未找到换脸模型: {swapper_path}")
    
    swapper = get_model(swapper_path, providers=['CPUExecutionProvider'])
    print("换脸模型加载成功")
except Exception as e:
    print(f"换脸模型加载失败: {e}")
    print("请确保模型文件存在且格式正确")


def swap_face(source_img_path, target_img_path, output_path):
    """
    执行AI换脸

    参数:
        source_img_path: 源图片路径（提供人脸）
        target_img_path: 目标图片路径（接收人脸）
        output_path: 输出图片路径
    """
    # 检查模型是否已加载
    if app is None:
        raise RuntimeError("FaceAnalysis 模型未初始化，请检查 insightface 安装")
    if swapper is None:
        raise RuntimeError("换脸模型未加载，请检查模型文件")
    
    # 读取图片
    source_img = cv2.imread(source_img_path)
    target_img = cv2.imread(target_img_path)

    if source_img is None or target_img is None:
        raise ValueError("无法读取图片，请检查路径是否正确")

    # 检测人脸
    try:
        source_faces = app.get(source_img)
        target_faces = app.get(target_img)
    except Exception as e:
        raise RuntimeError(f"人脸检测失败: {e}")

    if len(source_faces) == 0:
        raise ValueError("源图片中未检测到人脸")
    if len(target_faces) == 0:
        raise ValueError("目标图片中未检测到人脸")

    # 选择第一张人脸进行替换
    source_face = source_faces[0]
    target_face = target_faces[0]

    # 执行换脸（关键修改：使用swapper.get方法）
    try:
        res = swapper.get(target_img, target_face, source_face, paste_back=True)
    except Exception as e:
        raise RuntimeError(f"换脸处理失败: {e}")

    # 保存结果
    try:
        cv2.imwrite(output_path, res)
        print(f"换脸完成，结果已保存到: {output_path}")
    except Exception as e:
        raise RuntimeError(f"保存结果失败: {e}")

    return res


# 使用示例
if __name__ == "__main__":
    # 请替换为你自己的图片路径
    source_path = "./1.JPG"  # 提供人脸的图片
    target_path = "./2.jpg"  # 接收人脸的图片
    output_path = "result.jpg"  # 输出结果的路径

    try:
        result = swap_face(source_path, target_path, output_path)

        # 显示结果
        cv2.imshow('Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"发生错误: {e}")