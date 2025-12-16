import argparse
import base64
import os
import sys
import traceback
from io import BytesIO

# Force Torch/TIMM cache to project drive to avoid C: downloads
os.environ.setdefault("TORCH_HOME", r"E:\DeepfakeBench\.cache\torch")

import cv2
import dlib
import numpy as np
import yaml
import torch
import joblib
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import requests
from skimage import transform as trans
from skimage import feature as skfeature
from skimage import filters
import uvicorn
from fastapi.staticfiles import StaticFiles

# InsightFace imports
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# SimSwap imports (延迟导入以避免路径问题)
simswap_available = False
try:
    import torch.nn.functional as F
    simswap_available = True
except ImportError:
    print("警告: SimSwap 依赖未安装，SimSwap 换脸功能将不可用")

# 添加training目录到Python路径，以便导入检测器
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training'))

# 添加simswap目录到Python路径
SIMSWAP_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'simswap')
sys.path.append(SIMSWAP_ROOT)

# 添加sber-swap目录到Python路径
GHOST_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sber-swap')
sys.path.append(GHOST_ROOT)
# 远程 GHOST 服务地址，可通过环境变量覆盖
GHOST_REMOTE_URL = os.getenv("GHOST_REMOTE_URL", "http://127.0.0.1:9000")

# 尝试导入DETECTOR，如果失败则提供基础配置
try:
    from detectors import DETECTOR
except ImportError:
    DETECTOR = {}
    print("警告: 无法导入DETECTOR，检测功能可能不可用")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECTOR_CONFIG_DIR = os.path.join(PROJECT_ROOT, "training", "config", "detector")
TEST_CONFIG_PATH = os.path.join(PROJECT_ROOT, "training", "config", "test_config.yaml")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "training", "weights")

# 按类别组织的检测器清单（仅保留已下载权重的模型）
CATEGORY_ORDER = ["Naive", "Spatial", "Frequency", "MachineLearning"]
CATEGORY_LABELS = {
    "Naive": "Naive（基础）",
    "Spatial": "Spatial（空间特征）",
    "Frequency": "Frequency（频域特征）",
    "MachineLearning": "Machine Learning（传统机器学习）",
}

RAW_DETECTORS = [
    # Naive
    {"id": "xception", "label": "Xception", "category": "Naive", "config": "xception.yaml", "weight": "xception_best.pth"},
    {"id": "meso4", "label": "MesoNet (Meso4)", "category": "Naive", "config": "meso4.yaml", "weight": "meso4_best.pth"},
    {"id": "meso4Inception", "label": "MesoInception", "category": "Naive", "config": "meso4Inception.yaml", "weight": "meso4Incep_best.pth"},
    {"id": "resnet34", "label": "CNN-Aug (ResNet34)", "category": "Naive", "config": "resnet34.yaml", "weight": "cnnaug_best.pth"},
    {"id": "efficientnetb4", "label": "EfficientNet-B4", "category": "Naive", "config": "efficientnetb4.yaml", "weight": "effnb4_best.pth"},
    # Spatial
    {"id": "capsule_net", "label": "Capsule", "category": "Spatial", "config": "capsule_net.yaml", "weight": "capsule_best.pth"},
    {"id": "ffd", "label": "FFD", "category": "Spatial", "config": "ffd.yaml", "weight": "ffd_best.pth"},
    {"id": "core", "label": "CORE", "category": "Spatial", "config": "core.yaml", "weight": "core_best.pth"},
    {"id": "recce", "label": "RECCE", "category": "Spatial", "config": "recce.yaml", "weight": "recce_best.pth"},
    {"id": "ucf", "label": "UCF (Xception)", "category": "Spatial", "config": "ucf.yaml", "weight": "ucf_best.pth"},
    # Frequency
    {"id": "f3net", "label": "F3Net", "category": "Frequency", "config": "f3net.yaml", "weight": "f3net_best.pth"},
    {"id": "spsl", "label": "SPSL", "category": "Frequency", "config": "spsl.yaml", "weight": "spsl_best.pth"},
    {"id": "srm", "label": "SRM", "category": "Frequency", "config": "srm.yaml", "weight": "srm_best.pth"},
    # Machine Learning
    {"id": "svm_features", "label": "SVM (Image Features)", "category": "MachineLearning", "config": None, "weight": "../image_feature_svm.joblib"},
]

DETECTOR_CATALOG = []
for item in RAW_DETECTORS:
    # Special handling for SVM model
    if item["id"] == "svm_features":
        weight_path = os.path.join(PROJECT_ROOT, item["weight"])
        if not os.path.exists(weight_path):
            print(f"警告: 找不到SVM模型文件，已跳过 {item['label']} ({weight_path})")
            continue
        DETECTOR_CATALOG.append(
            {
                "id": item["id"],
                "label": item["label"],
                "category": item["category"],
                "detector_cfg": None,
                "test_cfg": None,
                "weights_path": weight_path,
            }
        )
        continue
    
    detector_cfg = os.path.join(DETECTOR_CONFIG_DIR, item["config"])
    weight_path = os.path.join(WEIGHTS_DIR, item["weight"])
    if not os.path.exists(detector_cfg):
        print(f"警告: 找不到配置文件，已跳过 {item['label']} ({detector_cfg})")
        continue
    if not os.path.exists(weight_path):
        print(f"警告: 找不到权重文件，已跳过 {item['label']} ({weight_path})")
        continue
    DETECTOR_CATALOG.append(
        {
            "id": item["id"],
            "label": item["label"],
            "category": item["category"],
            "detector_cfg": detector_cfg,
            "test_cfg": TEST_CONFIG_PATH,
            "weights_path": weight_path,
        }
    )

AVAILABLE_MODELS = {item["id"]: item for item in DETECTOR_CATALOG}


def build_model_options():
    """按类别生成下拉框选项 HTML。"""
    grouped = {}
    for det in DETECTOR_CATALOG:
        grouped.setdefault(det["category"], []).append(det)
    options = []
    for category in CATEGORY_ORDER:
        if category not in grouped:
            continue
        options.append(f'<optgroup label="{CATEGORY_LABELS.get(category, category)}">')
        for det in grouped[category]:
            options.append(f'<option value="{det["id"]}">{det["label"]}</option>')
        options.append("</optgroup>")
    return "\n".join(options)


MODEL_OPTIONS_HTML = build_model_options()


def load_config(detector_cfg_path, test_cfg_path):
    with open(detector_cfg_path, "r") as f:
        det_cfg = yaml.safe_load(f)
    with open(test_cfg_path, "r") as f:
        test_cfg = yaml.safe_load(f)
    det_cfg.update(test_cfg)
    
    # 修复配置文件中的相对路径
    # 将相对于项目根目录的路径转换为绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if 'pretrained' in det_cfg and det_cfg['pretrained']:
        pretrained_path = det_cfg['pretrained']
        if pretrained_path.startswith('./'):
            # 相对于项目根目录
            det_cfg['pretrained'] = os.path.join(project_root, pretrained_path[2:])
        elif not os.path.isabs(pretrained_path):
            # 其他相对路径
            det_cfg['pretrained'] = os.path.join(project_root, pretrained_path)
    
    return det_cfg


def build_model(cfg, weight_path, device):
    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)
    
    # 使用传入的 weight_path 而不是配置文件中的路径
    # 确保路径是绝对路径
    if not os.path.isabs(weight_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weight_path = os.path.join(project_root, weight_path)
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")
    
    print(f"加载权重文件: {weight_path}")
    ckpt = torch.load(weight_path, map_location=device)
    
    # 使用 strict=False 以便更宽容地处理权重不匹配
    # 某些模型（如 F3Net, SPSL）会在 build_backbone 中修改 conv1 层
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
    
    if missing_keys:
        print(f"  警告: 缺少的权重键 (通常是正常的): {missing_keys[:3]}...")
    if unexpected_keys:
        print(f"  警告: 未预期的权重键: {unexpected_keys[:3]}...")
    
    model.eval()
    return model


def preprocess_image(image: Image.Image, resolution, mean, std):
    preproc = transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    tensor = preproc(image.convert("RGB")).unsqueeze(0)
    return tensor


def encode_image_to_base64(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("结果编码失败，无法生成输出图像")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


class SimSwapService:
    """SimSwap 换脸服务类"""
    
    def __init__(self):
        self.model = None
        self.crop_app = None
        self.spnorm = None
        self.transformer = None
        self.is_initialized = False
        self.crop_size = 224
        self.ctx_id = -1
        self.initialize()
    
    def initialize(self):
        """初始化 SimSwap 模型"""
        try:
            if not simswap_available:
                print("SimSwap 依赖不可用")
                return
            
            # 检查 CUDA 是否可用
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                print("检测到 CUDA 可用，SimSwap 使用 GPU 模式")
                self.ctx_id = 0
            else:
                print("CUDA 不可用，SimSwap 使用 CPU 模式")
                self.ctx_id = -1
            
            # 导入 SimSwap 模块
            from models.models import create_model
            from options.test_options import TestOptions
            from insightface_func.face_detect_crop_multi import Face_detect_crop
            from util.norm import SpecificNorm
            
            # 构建选项
            arcface_path = os.path.join(SIMSWAP_ROOT, 'arcface_model', 'arcface_checkpoint.tar')
            if not os.path.exists(arcface_path):
                raise FileNotFoundError(f"未找到 ArcFace 权重: {arcface_path}")

            try:
                from torch.serialization import add_safe_globals
                from models.arcface_models import ResNet, IRBlock, SEBlock
                add_safe_globals([ResNet, IRBlock, SEBlock])
            except Exception as safe_exc:
                print(f"警告: 注册 ArcFace 反序列化类型失败: {safe_exc}")

            # 设置 checkpoints 绝对路径
            checkpoints_dir = os.path.join(SIMSWAP_ROOT, 'checkpoints')
            
            opt = TestOptions().parse([
                '--name', 'people',
                '--Arc_path', arcface_path,
                '--pic_a_path', 'temp.jpg',  # 占位符
                '--pic_b_path', 'temp.jpg',  # 占位符
                '--isTrain', 'False',
                '--no_simswaplogo',
                '--checkpoints_dir', checkpoints_dir
            ])
            opt.Arc_path = arcface_path
            opt.checkpoints_dir = checkpoints_dir
            
            # 设置 GPU
            if use_cuda:
                torch.cuda.set_device(0)
            
            # 创建模型
            torch.nn.Module.dump_patches = True
            self.model = create_model(opt)
            self.model.eval()
            
            # 初始化人脸检测
            self.crop_app = Face_detect_crop(
                name='antelope',
                root=os.path.join(SIMSWAP_ROOT, 'insightface_func', 'models')
            )
            self.crop_app.prepare(ctx_id=self.ctx_id, det_thresh=0.6, det_size=(640, 640), mode='None')
            
            # 初始化归一化
            self.spnorm = SpecificNorm()
            
            # 初始化图像转换器
            self.transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            self.crop_size = opt.crop_size
            self.is_initialized = True
            
            mode_str = "GPU 模式" if use_cuda else "CPU 模式"
            print(f"SimSwap 模型初始化成功 ({mode_str})")
            
        except Exception as e:
            print(f"SimSwap 初始化失败: {e}")
            print("SimSwap 换脸功能将不可用")
            traceback.print_exc()
            self.is_initialized = False
    
    def _totensor(self, array):
        """将 numpy 数组转换为 tensor"""
        tensor = torch.from_numpy(array)
        img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255)
    
    def swap(self, source_bytes: bytes, target_bytes: bytes) -> np.ndarray:
        """
        执行换脸操作
        
        参数:
            source_bytes: 源图片的字节数据（提供人脸）
            target_bytes: 目标图片的字节数据（接收人脸）
            
        返回:
            换脸结果图像 (numpy.ndarray)
        """
        if not self.is_initialized:
            raise RuntimeError("SimSwap 服务未初始化")
        
        from util.reverse2original import reverse2wholeimage
        
        # 将字节数据转换为 OpenCV 图像
        source_nparr = np.frombuffer(source_bytes, np.uint8)
        target_nparr = np.frombuffer(target_bytes, np.uint8)
        
        source_img = cv2.imdecode(source_nparr, cv2.IMREAD_COLOR)
        target_img = cv2.imdecode(target_nparr, cv2.IMREAD_COLOR)
        
        if source_img is None or target_img is None:
            raise ValueError("无法解码图片数据")
        
        # 检测并对齐源人脸
        source_align_crop, _ = self.crop_app.get(source_img, self.crop_size)
        if len(source_align_crop) == 0:
            raise ValueError("源图片中未检测到人脸")
        
        source_align_crop_pil = Image.fromarray(cv2.cvtColor(source_align_crop[0], cv2.COLOR_BGR2RGB))
        source_tensor = self.transformer(source_align_crop_pil)
        source_tensor = source_tensor.view(-1, source_tensor.shape[0], source_tensor.shape[1], source_tensor.shape[2])
        
        if self.ctx_id >= 0:
            source_tensor = source_tensor.cuda()
        
        # 创建 latent id
        source_downsample = F.interpolate(source_tensor, size=(112, 112))
        latent_id = self.model.netArc(source_downsample)
        latent_id = F.normalize(latent_id, p=2, dim=1)
        
        # 检测目标人脸
        target_align_crop_list, target_mat_list = self.crop_app.get(target_img, self.crop_size)
        if len(target_align_crop_list) == 0:
            raise ValueError("目标图片中未检测到人脸")
        
        # 对每个目标人脸执行换脸
        swap_result_list = []
        target_tensor_list = []
        
        for target_crop in target_align_crop_list:
            target_tensor = self._totensor(cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB))[None, ...]
            if self.ctx_id >= 0:
                target_tensor = target_tensor.cuda()
            
            swap_result = self.model(None, target_tensor, latent_id, None, True)[0]
            swap_result_list.append(swap_result)
            target_tensor_list.append(target_tensor)
        
        # 将换脸结果贴回原图
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            reverse2wholeimage(
                target_tensor_list,
                swap_result_list,
                target_mat_list,
                self.crop_size,
                target_img,
                None,  # logoclass
                output_path,
                no_simswaplogo=True,
                pasring_model=None,
                use_mask=False,
                norm=self.spnorm
            )
            
            result = cv2.imread(output_path)
            os.remove(output_path)
            
            if result is None:
                raise ValueError("换脸结果生成失败")
            
            return result
            
        except Exception as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            raise e


class GhostService:
    """GHOST (sber-swap) 换脸服务类"""
    
    def __init__(self):
        self.app = None
        self.G = None
        self.netArc = None
        self.handler = None
        self.ctx_id = -1
        self.is_initialized = False
        self.crop_size = 224
        self.initialize()
    
    def initialize(self):
        """初始化 GHOST 模型"""
        try:
            # 检查 CUDA 是否可用
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                print("检测到 CUDA 可用，GHOST 使用 GPU 模式")
                self.ctx_id = 0
            else:
                print("CUDA 不可用，GHOST 使用 CPU 模式")
                self.ctx_id = -1
                print("警告: GHOST 在 CPU 上运行会非常慢")
            
            # 切换工作目录到 sber-swap
            original_cwd = os.getcwd()
            os.chdir(GHOST_ROOT)
            
            try:
                # 导入 GHOST 模块
                from network.AEI_Net import AEI_Net
                from arcface_model.iresnet import iresnet100
                from insightface_func.face_detect_crop_multi import Face_detect_crop
                from coordinate_reg.image_infer import Handler
                
                # 初始化人脸检测
                self.app = Face_detect_crop(name='antelope', root='./insightface_func/models')
                self.app.prepare(ctx_id=self.ctx_id, det_thresh=0.6, det_size=(640, 640))
                
                # 加载主生成器 (使用 2 blocks 版本)
                G_path = os.path.join(GHOST_ROOT, 'weights', 'G_unet_2blocks.pth')
                if not os.path.exists(G_path):
                    raise FileNotFoundError(f"未找到 GHOST 生成器权重: {G_path}")
                
                self.G = AEI_Net('unet', num_blocks=2, c_id=512)
                self.G.eval()
                self.G.load_state_dict(torch.load(G_path, map_location=torch.device('cpu'), weights_only=False))
                
                if use_cuda:
                    self.G = self.G.cuda()
                    self.G = self.G.half()
                
                # 加载 ArcFace 模型
                arcface_path = os.path.join(GHOST_ROOT, 'arcface_model', 'backbone.pth')
                if not os.path.exists(arcface_path):
                    raise FileNotFoundError(f"未找到 ArcFace 权重: {arcface_path}")
                
                self.netArc = iresnet100(fp16=False)
                self.netArc.load_state_dict(torch.load(arcface_path, map_location=torch.device('cpu'), weights_only=False))
                
                if use_cuda:
                    self.netArc = self.netArc.cuda()
                self.netArc.eval()
                
                # 加载关键点检测模型
                self.handler = Handler('./coordinate_reg/model/2d106det', 0, ctx_id=self.ctx_id, det_size=640)
                
                self.is_initialized = True
                mode_str = "GPU 模式" if use_cuda else "CPU 模式"
                print(f"GHOST 模型初始化成功 ({mode_str})")
                
            finally:
                # 恢复工作目录
                os.chdir(original_cwd)
            
        except Exception as e:
            print(f"GHOST 初始化失败: {e}")
            print("GHOST 换脸功能将不可用")
            traceback.print_exc()
            self.is_initialized = False
            # 确保恢复工作目录
            try:
                os.chdir(original_cwd)
            except:
                pass
    
    def swap(self, source_bytes: bytes, target_bytes: bytes) -> np.ndarray:
        """
        执行换脸操作
        
        参数:
            source_bytes: 源图片的字节数据（提供人脸）
            target_bytes: 目标图片的字节数据（接收人脸）
            
        返回:
            换脸结果图像 (numpy.ndarray)
        """
        if not self.is_initialized:
            raise RuntimeError("GHOST 服务未初始化")
        
        # 切换工作目录
        original_cwd = os.getcwd()
        os.chdir(GHOST_ROOT)
        
        try:
            from utils.inference.image_processing import normalize_and_torch, get_final_image
            from insightface.utils import face_align
            import torch.nn.functional as F
            
            # 解码图像
            source_nparr = np.frombuffer(source_bytes, np.uint8)
            target_nparr = np.frombuffer(target_bytes, np.uint8)
            
            source_img = cv2.imdecode(source_nparr, cv2.IMREAD_COLOR)
            target_img = cv2.imdecode(target_nparr, cv2.IMREAD_COLOR)
            
            if source_img is None or target_img is None:
                raise ValueError("无法解码图片数据")
            
            # 检测源人脸
            source_kps_list = self.app.get(source_img, self.crop_size)
            if source_kps_list is None or len(source_kps_list) == 0:
                raise ValueError("源图片中未检测到人脸")
            
            # 对齐并裁剪源人脸
            M_src, _ = face_align.estimate_norm(source_kps_list[0], self.crop_size, mode='None')
            source_crop = cv2.warpAffine(source_img, M_src, (self.crop_size, self.crop_size), borderValue=0.0)
            source_crop_rgb = source_crop[:, :, ::-1]  # BGR to RGB
            
            # 检测目标人脸
            target_kps_list = self.app.get(target_img, self.crop_size)
            if target_kps_list is None or len(target_kps_list) == 0:
                raise ValueError("目标图片中未检测到人脸")
            
            # 获取源人脸特征
            source_norm = normalize_and_torch(source_crop_rgb)
            source_embed = self.netArc(F.interpolate(source_norm, scale_factor=0.5, mode='bilinear', align_corners=True))
            
            if self.ctx_id >= 0:
                source_embed = source_embed.half()
            
            # 对每个目标人脸执行换脸
            final_frames_list = []
            crop_frames_list = []
            tfm_arrays_list = []
            
            for target_kps in target_kps_list:
                # 对齐并裁剪目标人脸
                M_tgt, _ = face_align.estimate_norm(target_kps, self.crop_size, mode='None')
                target_crop = cv2.warpAffine(target_img, M_tgt, (self.crop_size, self.crop_size), borderValue=0.0)
                target_crop_rgb = target_crop[:, :, ::-1]  # BGR to RGB
                
                # 准备目标图像
                target_norm = normalize_and_torch(target_crop_rgb)
                
                if self.ctx_id >= 0:
                    target_norm = target_norm.half()
                
                # 生成换脸结果
                with torch.no_grad():
                    Y_st, _ = self.G(source_embed, target_norm)
                
                # 转换回 numpy
                Y_st = (Y_st.squeeze().permute(1, 2, 0).cpu().clamp(-1, 1).numpy() + 1) / 2
                Y_st = (Y_st * 255).astype(np.uint8)
                Y_st = Y_st[:, :, ::-1]  # RGB to BGR
                
                final_frames_list.append([Y_st])
                crop_frames_list.append([target_crop])
                tfm_arrays_list.append([M_tgt])
            
            # 将换脸结果贴回原图
            result = get_final_image(final_frames_list, crop_frames_list, target_img, tfm_arrays_list, self.handler)
            
            return result
            
        finally:
            # 恢复工作目录
            os.chdir(original_cwd)


class InsightFaceService:
    """InsightFace 换脸服务类"""
    
    def __init__(self):
        self.app = None
        self.swapper = None
        self.ctx_id = -1  # 默认为CPU
        self.is_initialized = False
        self.initialize()
    
    def initialize(self):
        """初始化 InsightFace 模型"""
        try:
            # 检查 CUDA 是否可用
            providers = ort.get_available_providers()
            use_cuda = 'CUDAExecutionProvider' in providers
            
            if use_cuda:
                print("检测到 CUDA 可用，使用 GPU 模式")
                self.ctx_id = 0
                provider_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                print("CUDA 不可用，使用 CPU 模式")
                self.ctx_id = -1
                provider_list = ['CPUExecutionProvider']
            
            # 模型目录与路径
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            MODELS_DIR = os.path.join(BASE_DIR, 'models')
            FACE_ROOT = BASE_DIR

            # 初始化 FaceAnalysis 应用
            self.app = FaceAnalysis(name='buffalo_l', root=FACE_ROOT, providers=provider_list)
            self.app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))

            # 加载换脸模型
            swapper_path = os.path.join(MODELS_DIR, 'inswapper_128.onnx')
            if not os.path.exists(swapper_path):
                raise FileNotFoundError(f"未找到换脸模型: {swapper_path}")

            self.swapper = get_model(swapper_path, providers=provider_list)
            self.is_initialized = True
            
            mode_str = "GPU 模式" if use_cuda else "CPU 模式"
            print(f"InsightFace 模型初始化成功 ({mode_str})")
            
        except Exception as e:
            print(f"InsightFace 初始化失败: {e}")
            print("换脸功能将不可用")
            traceback.print_exc()
            self.is_initialized = False
    
    def swap(self, source_bytes: bytes, target_bytes: bytes) -> np.ndarray:
        """
        执行换脸操作
        
        参数:
            source_bytes: 源图片的字节数据（提供人脸）
            target_bytes: 目标图片的字节数据（接收人脸）
            
        返回:
            换脸结果图像 (numpy.ndarray)
        """
        if not self.is_initialized:
            raise RuntimeError("InsightFace 服务未初始化")
        
        # 将字节数据转换为 OpenCV 图像
        source_nparr = np.frombuffer(source_bytes, np.uint8)
        target_nparr = np.frombuffer(target_bytes, np.uint8)
        
        source_img = cv2.imdecode(source_nparr, cv2.IMREAD_COLOR)
        target_img = cv2.imdecode(target_nparr, cv2.IMREAD_COLOR)
        
        if source_img is None or target_img is None:
            raise ValueError("无法解码图片数据")
        
        # 检测人脸
        source_faces = self.app.get(source_img)
        target_faces = self.app.get(target_img)
        
        if len(source_faces) == 0:
            raise ValueError("源图片中未检测到人脸")
        if len(target_faces) == 0:
            raise ValueError("目标图片中未检测到人脸")
        
        # 选择第一张人脸进行替换
        source_face = source_faces[0]
        target_face = target_faces[0]
        
        # 执行换脸
        result = self.swapper.get(target_img, target_face, source_face, paste_back=True)
        
        return result


class ModelManager:
    def __init__(self, device):
        self.device = device
        self.cache = {}

    def get_model(self, model_name):
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"未知检测模型: {model_name}")
        if model_name in self.cache:
            return self.cache[model_name]

        cfg_paths = AVAILABLE_MODELS[model_name]
        cfg = load_config(cfg_paths["detector_cfg"], cfg_paths["test_cfg"])
        model = build_model(cfg, cfg_paths["weights_path"], self.device)
        bundle = {"cfg": cfg, "model": model}
        self.cache[model_name] = bundle
        return bundle


class FaceAligner:
    def __init__(self, predictor_path, scale=1.3):
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Predictor file not found: {predictor_path}")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.scale = scale

    def align(self, image: Image.Image, resolution: int):
        rgb_image = np.array(image.convert("RGB"))
        faces = self.detector(rgb_image, 1)
        if len(faces) == 0:
            return None
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        landmarks = self._extract_five_points(rgb_image, face)
        if landmarks is None:
            return None
        aligned = self._warp_face(rgb_image, landmarks, resolution)
        if aligned is None:
            return None
        return Image.fromarray(aligned)

    def _extract_five_points(self, image, face):
        shape = self.predictor(image, face)
        points = np.array(
            [
                (shape.part(36).x, shape.part(36).y),
                (shape.part(45).x, shape.part(45).y),
                (shape.part(30).x, shape.part(30).y),
                (shape.part(48).x, shape.part(48).y),
                (shape.part(54).x, shape.part(54).y),
            ],
            dtype=np.float32,
        )
        return points

    def _warp_face(self, image, landmark, resolution):
        target_size = [112, 112]
        dst = np.array(
            [
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        if target_size[1] == 112:
            dst[:, 0] += 8.0

        outsize = (resolution, resolution)
        dst[:, 0] *= (outsize[0] / target_size[0])
        dst[:, 1] *= (outsize[1] / target_size[1])

        tform = trans.SimilarityTransform()
        tform.estimate(landmark, dst)
        M = tform.params[0:2, :]
        warped = cv2.warpAffine(image, M, outsize, borderValue=0.0)
        return warped


def extract_svm_features(image: np.ndarray) -> np.ndarray:
    """
    从图像中提取 SVM 用的特征（与训练时完全一致）
    特征包括：SRM + Spatial features
    
    参数:
        image: BGR格式的图像 (numpy.ndarray)
    
    返回:
        特征向量 (numpy.ndarray) - 约7337维
    """
    from scipy.stats import skew, kurtosis
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
    
    # 确保是BGR格式并resize到256x256
    if len(image.shape) == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image.copy()
    
    bgr = cv2.resize(bgr, (256, 256))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    all_features = []
    
    # ========== SRM特征提取 ==========
    # 定义3个SRM核
    SRM_KERNELS = np.stack([
        [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
        [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
        [[0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, -2, 4, -2, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0]],
    ], axis=0).astype(np.float32)
    SRM_KERNELS[0] /= 4.0
    SRM_KERNELS[1] /= 12.0
    SRM_KERNELS[2] /= 4.0
    
    bins = 32
    clip = 3.0
    
    for k in SRM_KERNELS:
        res = cv2.filter2D(gray, -1, k, borderType=cv2.BORDER_REFLECT)
        res = np.clip(res, -clip, clip)
        res_flat = res.reshape(-1)
        
        # 统计特征
        all_features.append(res_flat.mean())
        all_features.append(res_flat.std() + 1e-6)
        all_features.append(skew(res_flat))
        all_features.append(kurtosis(res_flat))
        
        # 直方图特征
        hist, _ = np.histogram(res_flat, bins=bins, range=(-clip, clip), density=True)
        all_features.extend(hist.tolist())
    
    # ========== 空域特征提取 ==========
    
    # 1. LBP histogram
    try:
        lbp = local_binary_pattern(gray.astype(np.uint8), P=8, R=2, method="uniform")
        hist, _ = np.histogram(lbp, bins=np.arange(0, 8 + 3), range=(0, 8 + 2), density=True)
        all_features.extend(hist.tolist())
    except Exception as e:
        print(f"LBP特征提取失败: {e}")
        all_features.extend([0.0] * 10)
    
    # 2. GLCM stats
    try:
        quant = (gray / 4).astype(np.uint8)
        glcm = graycomatrix(quant, distances=[1], angles=[0], levels=64, symmetric=True, normed=True)
        all_features.append(graycoprops(glcm, "contrast")[0, 0])
        all_features.append(graycoprops(glcm, "homogeneity")[0, 0])
        all_features.append(graycoprops(glcm, "energy")[0, 0])
        all_features.append(graycoprops(glcm, "correlation")[0, 0])
    except Exception as e:
        print(f"GLCM特征提取失败: {e}")
        all_features.extend([0.0] * 4)
    
    # 3. Gradient stats (Sobel)
    try:
        gx = cv2.Sobel(gray.astype(np.uint8), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray.astype(np.uint8), cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        all_features.append(mag.mean())
        all_features.append(mag.std())
    except Exception as e:
        print(f"梯度特征提取失败: {e}")
        all_features.extend([0.0] * 2)
    
    # 4. Laplacian stats
    try:
        lap = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_32F, ksize=3)
        lap_flat = lap.reshape(-1)
        all_features.append(lap.mean())
        all_features.append(lap.std())
        all_features.append(skew(lap_flat))
        all_features.append(kurtosis(lap_flat))
    except Exception as e:
        print(f"Laplacian特征提取失败: {e}")
        all_features.extend([0.0] * 4)
    
    # 5. HOG features
    try:
        hog_feat = hog(
            gray.astype(np.uint8), 
            pixels_per_cell=(16, 16), 
            cells_per_block=(2, 2), 
            orientations=8, 
            feature_vector=True
        )
        all_features.extend(hog_feat.tolist())
    except Exception as e:
        print(f"HOG特征提取失败: {e}")
        # HOG通常生成很多特征，这里用默认填充
        # 对于256x256图像，16x16 cell，应该是15x15个cell，with (2,2)块，大约生成7200维
        all_features.extend([0.0] * 7200)
    
    # 6. Color consistency
    try:
        means = bgr.reshape(-1, 3).mean(axis=0)
        stds = bgr.reshape(-1, 3).std(axis=0)
        all_features.extend(means.tolist())
        all_features.extend(stds.tolist())
        
        # Center vs border difference
        h, w, _ = bgr.shape
        margin = max(1, h // 8)
        center = bgr[margin:-margin, margin:-margin]
        border = bgr.copy()
        border[margin:-margin, margin:-margin] = 0
        
        if np.any(border):
            b_mask = (np.sum(border, axis=2) > 0)
            b_vals = border[b_mask]
            b_mean = b_vals.reshape(-1, 3).mean(axis=0)
            c_mean = center.reshape(-1, 3).mean(axis=0)
            all_features.extend((c_mean - b_mean).tolist())
        else:
            all_features.extend([0.0, 0.0, 0.0])
    except Exception as e:
        print(f"颜色特征提取失败: {e}")
        all_features.extend([0.0] * 9)
    
    return np.array(all_features, dtype=np.float32)



def main():
    parser = argparse.ArgumentParser(description="DeepfakeBench WebUI")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--device", default="cpu", help="Device to use")
    args = parser.parse_args()

    # 初始化服务
    device = torch.device(args.device)
    model_manager = ModelManager(device)
    
    # 初始化人脸对齐器（可选）
    predictor_path = "../preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"
    face_aligner = None
    if os.path.exists(predictor_path):
        try:
            face_aligner = FaceAligner(predictor_path)
            print("人脸对齐器初始化成功")
        except Exception as e:
            print(f"人脸对齐器初始化失败: {e}")
    else:
        print("未找到人脸对齐模型，将跳过人脸对齐")
    
    # 延迟初始化换脸服务，避免页面加载阻塞
    insightface_service = None
    simswap_service = None



    app = FastAPI(title="DeepfakeBench WebUI", version="1.0.0")

    # 挂载静态文件
    # ensure 'tools/static' exists relative to execution path or use absolute
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index():
        # 直接读取模板文件返回，或者使用 FileResponse
        # 简单起见，读取内容返回 HTMLResponse
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                content = f.read()
            return HTMLResponse(content)
        return HTMLResponse("<h1>Error: Template not found</h1>")

    @app.get("/api/models")
    async def get_models():
        """返回按类别分组的模型列表"""
        grouped = {}
        for det in DETECTOR_CATALOG:
            cat = CATEGORY_LABELS.get(det["category"], det["category"])
            grouped.setdefault(cat, []).append({
                "id": det["id"],
                "label": det["label"]
            })
        return JSONResponse(grouped)

    @app.post("/api/predict")
    async def api_predict(model_name: str = Form(...), image: UploadFile = File(...)):
        try:
            # Special handling for SVM model
            if model_name == "svm_features":
                if model_name not in AVAILABLE_MODELS:
                    return JSONResponse({"error": "SVM模型未找到"}, status_code=500)
                
                # Load SVM model
                svm_path = AVAILABLE_MODELS[model_name]["weights_path"]
                try:
                    svm_model = joblib.load(svm_path)
                except Exception as load_exc:
                    return JSONResponse({"error": f"SVM模型加载失败: {load_exc}"}, status_code=500)
                
                # Read and preprocess image
                contents = await image.read()
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    return JSONResponse({"error": "无法解码图像"}, status_code=400)
                
                # Extract features
                try:
                    features = extract_svm_features(img)
                    features = features.reshape(1, -1)  # Shape: (1, n_features)
                except Exception as feat_exc:
                    return JSONResponse({"error": f"特征提取失败: {feat_exc}"}, status_code=500)
                
                # Predict
                try:
                    # SVM预测：0=Real, 1=Fake
                    prediction = svm_model.predict(features)[0]
                    
                    # 尝试获取概率
                    if hasattr(svm_model, 'predict_proba'):
                        proba = svm_model.predict_proba(features)[0]
                        prob_real = float(proba[0])
                        prob_fake = float(proba[1])
                    elif hasattr(svm_model, 'decision_function'):
                        # 使用决策函数转换为概率
                        decision = svm_model.decision_function(features)[0]
                        # Sigmoid 转换
                        prob_fake = float(1 / (1 + np.exp(-decision)))
                        prob_real = 1.0 - prob_fake
                    else:
                        # 只有硬分类结果
                        prob_fake = 1.0 if prediction == 1 else 0.0
                        prob_real = 1.0 - prob_fake
                    
                    return JSONResponse({
                        "fake": prob_fake, 
                        "real": prob_real, 
                        "note": "使用传统机器学习模型 (SVM + 图像特征) 进行检测"
                    })
                    
                except Exception as pred_exc:
                    return JSONResponse({"error": f"SVM预测失败: {pred_exc}"}, status_code=500)
            
            # Normal deep learning model handling
            if not DETECTOR:
                return JSONResponse({"error": "检测模型未加载，请检查模型配置"}, status_code=500)
                
            bundle = model_manager.get_model(model_name)
            cfg, model = bundle["cfg"], bundle["model"]

            contents = await image.read()
            pil_image = Image.open(BytesIO(contents)).convert("RGB")
            processed_image = pil_image
            preprocess_note = None
            
            if face_aligner:
                aligned_image = face_aligner.align(pil_image, cfg["resolution"])
                if aligned_image is not None:
                    processed_image = aligned_image
                else:
                    preprocess_note = "未检测到清晰人脸，将直接对整张图像推理，结果可能不可靠。"
            else:
                preprocess_note = "未找到对齐模型，将直接对整张图像推理，结果可能不可靠。"

            image_tensor = preprocess_image(
                processed_image,
                resolution=cfg["resolution"],
                mean=cfg["mean"],
                std=cfg["std"],
            ).to(device)

            # UCF 模型在推理阶段依然会访问 label 用于内部统计
            label_tensor = torch.zeros(1, dtype=torch.long, device=device)
            data_dict = {"image": image_tensor, "label": label_tensor}
            
            with torch.no_grad():
                preds = model(data_dict, inference=True)

            if "prob" in preds:
                prob_fake = preds["prob"].item()
            else:
                logits = preds["cls"]
                if logits.shape[-1] == 1:
                    prob_fake = torch.sigmoid(logits).item()
                else:
                    prob_fake = torch.softmax(logits, dim=-1)[0, 1].item()
            
            prob_real = 1.0 - prob_fake
            return JSONResponse({"fake": prob_fake, "real": prob_real, "note": preprocess_note})
            
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            traceback.print_exc()
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/swap")
    async def api_swap(
        source_image: UploadFile = File(..., description="提供人脸的图片"),
        target_image: UploadFile = File(..., description="接收人脸的图片"),
        backend: str = Form("insightface", description="换脸引擎: insightface, simswap, ghost 或 ghost_remote"),
    ):
        try:
            source_bytes = await source_image.read()
            target_bytes = await target_image.read()
            
            note = None
            
            nonlocal simswap_service, insightface_service

            if backend == "simswap":
                if simswap_service is None:
                    simswap_service = SimSwapService()
                if not simswap_service.is_initialized:
                    return JSONResponse({"error": "SimSwap 服务未初始化，请检查模型配置"}, status_code=500)
                
                result_img = simswap_service.swap(source_bytes, target_bytes)
                
                if simswap_service.ctx_id == -1:
                    note = "当前使用 CPU 推理 (SimSwap)，速度可能较慢。"
                else:
                    note = "使用 SimSwap 引擎完成换脸"
                    
            elif backend == "ghost":
                return JSONResponse({"error": "本地 GHOST 已禁用，请使用 GHOST Remote 后端"}, status_code=400)
            
            elif backend == "ghost_remote":
                try:
                    resp = requests.post(
                        f"{GHOST_REMOTE_URL}/swap",
                        files={
                            "source_image": ("source.jpg", source_bytes, "image/jpeg"),
                            "target_image": ("target.jpg", target_bytes, "image/jpeg"),
                        },
                        timeout=120,
                    )
                except requests.RequestException as exc:
                    return JSONResponse({"error": f"调用远程 GHOST 失败: {exc}"}, status_code=500)
                
                if resp.status_code >= 400:
                    return JSONResponse({"error": f"远程 GHOST 返回错误: {resp.text}"}, status_code=resp.status_code)
                
                data = resp.json()
                if "error" in data:
                    return JSONResponse({"error": data.get("error")}, status_code=400)
                encoded = data.get("image")
                if not encoded:
                    return JSONResponse({"error": "远程 GHOST 未返回图像"}, status_code=500)
                note = data.get("note", f"使用远程 GHOST: {GHOST_REMOTE_URL}")
                return JSONResponse({"image": encoded, "note": note})

            else:  # insightface
                if insightface_service is None:
                    insightface_service = InsightFaceService()
                if not insightface_service.is_initialized:
                    return JSONResponse({"error": "InsightFace 服务未初始化，换脸功能不可用"}, status_code=500)
                
                result_img = insightface_service.swap(source_bytes, target_bytes)
                
                if insightface_service.ctx_id == -1:
                    note = "当前使用 CPU 推理 (InsightFace)，速度可能较慢。"
                else:
                    note = "使用 InsightFace 引擎完成换脸"
            
            encoded = encode_image_to_base64(result_img)
            return JSONResponse({"image": encoded, "note": note})
            
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            traceback.print_exc()
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/visualize")
    async def api_visualize(model_name: str = Form(...), image: UploadFile = File(...)):
        """
        生成检测可视化结果
        
        根据模型类型返回不同的可视化：
        - Naive模型：Grad-CAM热力图
        - Frequency模型：DCT频谱、高频成分、SRM滤波响应
        - Spatial模型：注意力图、边缘检测、纹理特征
        - ML模型：SRM特征、梯度特征、LBP纹理
        """
        try:
            # 导入可视化模块
            from visualization import (
                generate_visualization, 
                get_detector_type,
                FrequencyVisualizer,
                SpatialVisualizer,
                MLFeatureVisualizer
            )
            
            # 读取图像
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return JSONResponse({"error": "无法解码图像"}, status_code=400)
            
            input_tensor = None
            model = None
            
            # 对于深度学习模型，获取模型和输入张量用于Grad-CAM
            if model_name != "svm_features" and model_name in AVAILABLE_MODELS:
                try:
                    bundle = model_manager.get_model(model_name)
                    cfg, model = bundle["cfg"], bundle["model"]
                    
                    # 预处理图像
                    pil_image = Image.open(BytesIO(contents)).convert("RGB")
                    input_tensor = preprocess_image(
                        pil_image,
                        resolution=cfg["resolution"],
                        mean=cfg["mean"],
                        std=cfg["std"],
                    ).to(device)
                except Exception as e:
                    print(f"模型加载失败，将使用基础可视化: {e}")
            
            # 生成可视化
            visualizations = generate_visualization(model, model_name, img, input_tensor)
            
            # 将可视化结果编码为base64
            result = {}
            for name, vis in visualizations.items():
                if len(vis.shape) == 2:
                    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                _, buffer = cv2.imencode('.jpg', vis)
                result[name] = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            # 获取检测器类型描述
            detector_type = get_detector_type(model_name)
            type_descriptions = {
                'naive': '基础CNN模型 - Grad-CAM显示网络关注的可疑区域',
                'frequency': '频域检测模型 - 分析DCT频谱和高频伪造痕迹',
                'spatial': '空间特征模型 - 注意力图和纹理异常分析',
                'ml': '传统机器学习 - SRM噪声残差和纹理特征分析',
                'unknown': '通用分析 - 边缘和梯度特征'
            }
            
            return JSONResponse({
                "visualizations": result,
                "detector_type": detector_type,
                "description": type_descriptions.get(detector_type, "未知模型类型"),
                "count": len(result)
            })
            
        except ImportError as ie:
            return JSONResponse({
                "error": f"可视化模块导入失败: {ie}",
                "note": "请确保 visualization.py 文件存在"
            }, status_code=500)
        except Exception as exc:
            traceback.print_exc()
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/visualization_types")
    async def get_visualization_types():
        """返回各类型模型的可视化说明"""
        return JSONResponse({
            "types": {
                "naive": {
                    "name": "Naive（基础CNN）",
                    "description": "使用Grad-CAM技术，展示神经网络关注的图像区域。红色区域表示模型认为最可疑的位置。",
                    "visualizations": ["grad_cam", "grad_cam_overlay", "edge_detection"],
                    "models": ["xception", "meso4", "meso4Inception", "resnet34", "efficientnetb4"]
                },
                "frequency": {
                    "name": "Frequency（频域特征）",
                    "description": "分析图像的频率成分。伪造图像通常在高频区域有异常模式，这是GAN生成或图像处理留下的痕迹。",
                    "visualizations": ["dct_spectrum", "high_frequency", "srm_noise_residual"],
                    "models": ["f3net", "spsl", "srm"]
                },
                "spatial": {
                    "name": "Spatial（空间特征）",
                    "description": "检测空间域的异常，如边缘不连续、纹理不一致等。注意力图显示模型关注的边界和纹理区域。",
                    "visualizations": ["attention_map", "attention_overlay", "spatial_edges", "texture_pattern"],
                    "models": ["capsule_net", "ffd", "core", "recce", "ucf"]
                },
                "ml": {
                    "name": "Machine Learning（传统ML）",
                    "description": "使用手工设计的特征，包括SRM噪声残差（检测隐写和伪造痕迹）、梯度特征（边缘信息）和LBP纹理（局部模式）。",
                    "visualizations": ["srm_features", "gradient_features", "lbp_texture"],
                    "models": ["svm_features"]
                }
            }
        })

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
