"""
可视化模块：展示不同类型检测器的判断依据
支持的可视化类型：
1. Naive模型：Grad-CAM热力图
2. Frequency模型：频域特征可视化（DCT、滤波器响应）
3. Spatial模型：空间注意力图
4. ML模型：传统特征可视化（SRM、LBP、边缘等）
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非GUI后端


class GradCAM:
    """Grad-CAM实现：用于CNN模型的注意力可视化"""
    
    def __init__(self, model, target_layer_name: str = None):
        self.model = model
        self.gradients = None
        self.activations = None
        self.target_layer = None
        self.hooks = []
        
        # 自动找到合适的目标层
        if target_layer_name:
            self._find_layer(target_layer_name)
        else:
            self._auto_find_layer()
    
    def _find_layer(self, name: str):
        """根据名称查找层"""
        for n, m in self.model.named_modules():
            if name in n:
                self.target_layer = m
                break
    
    def _auto_find_layer(self):
        """自动找到最后一个卷积层"""
        for module in reversed(list(self.model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                self.target_layer = module
                break
    
    def _save_gradient(self, grad):
        self.gradients = grad.detach()
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def register_hooks(self):
        if self.target_layer is None:
            return False
        
        # 注册前向和反向钩子
        handle_forward = self.target_layer.register_forward_hook(self._save_activation)
        handle_backward = self.target_layer.register_full_backward_hook(
            lambda m, gi, go: self._save_gradient(go[0])
        )
        self.hooks = [handle_forward, handle_backward]
        return True
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        生成Grad-CAM热力图
        
        Args:
            input_tensor: 输入图像张量 [1, C, H, W]
            target_class: 目标类别（默认为预测类别）
        
        Returns:
            cam: 热力图 [H, W]
        """
        if not self.register_hooks():
            return None
        
        self.model.eval()
        
        # 前向传播
        output = self.model({'image': input_tensor, 'label': torch.zeros(1).long()}, inference=True)
        
        if 'cls' in output:
            logits = output['cls']
        elif 'prob' in output:
            logits = output['prob'].unsqueeze(1)
        else:
            self.remove_hooks()
            return None
        
        if target_class is None:
            if logits.dim() > 1 and logits.size(-1) > 1:
                target_class = logits.argmax(dim=-1).item()
            else:
                target_class = 1 if logits.sigmoid().item() > 0.5 else 0
        
        # 反向传播
        self.model.zero_grad()
        if logits.dim() > 1 and logits.size(-1) > 1:
            score = logits[0, target_class]
        else:
            score = logits[0, 0] if target_class == 1 else -logits[0, 0]
        score.backward(retain_graph=True)
        
        self.remove_hooks()
        
        if self.gradients is None or self.activations is None:
            return None
        
        # 计算权重（全局平均池化梯度）
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # 加权求和
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # ReLU并归一化
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # 归一化到0-1
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam


class FrequencyVisualizer:
    """频域特征可视化器"""
    
    @staticmethod
    def visualize_dct(image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        可视化DCT频谱
        
        Returns:
            Dict包含：
            - dct_spectrum: DCT频谱图
            - low_freq: 低频成分
            - mid_freq: 中频成分  
            - high_freq: 高频成分
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (256, 256))
        
        # 计算DCT
        dct = cv2.dct(np.float32(gray) / 255.0)
        
        # 对数变换以便可视化
        dct_log = np.log(np.abs(dct) + 1e-10)
        dct_log = (dct_log - dct_log.min()) / (dct_log.max() - dct_log.min() + 1e-10)
        
        # 频带分割
        h, w = dct.shape
        center = (h // 2, w // 2)
        
        # 创建频带掩码
        y, x = np.ogrid[:h, :w]
        # 对于DCT，低频在左上角
        dist = np.sqrt(x**2 + y**2)
        
        low_mask = dist < h // 4
        mid_mask = (dist >= h // 4) & (dist < h // 2)
        high_mask = dist >= h // 2
        
        # IDCT重建各频带
        low_freq = cv2.idct(dct * low_mask.astype(np.float32))
        mid_freq = cv2.idct(dct * mid_mask.astype(np.float32))
        high_freq = cv2.idct(dct * high_mask.astype(np.float32))
        
        # 归一化到0-255
        def normalize(img):
            img = img - img.min()
            if img.max() > 0:
                img = img / img.max()
            return (img * 255).astype(np.uint8)
        
        return {
            'dct_spectrum': (dct_log * 255).astype(np.uint8),
            'low_freq': normalize(low_freq),
            'mid_freq': normalize(mid_freq),
            'high_freq': normalize(high_freq),
        }
    
    @staticmethod
    def visualize_srm_filters(image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        可视化SRM滤波器响应
        
        Returns:
            Dict包含各个SRM滤波器的响应
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (256, 256)).astype(np.float32)
        
        # 定义SRM滤波器
        # 滤波器1: KB (基本边缘)
        filter1 = np.array([[0, 0, 0, 0, 0],
                           [0, -1, 2, -1, 0],
                           [0, 2, -4, 2, 0],
                           [0, -1, 2, -1, 0],
                           [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0
        
        # 滤波器2: KV (高频噪声)
        filter2 = np.array([[-1, 2, -2, 2, -1],
                           [2, -6, 8, -6, 2],
                           [-2, 8, -12, 8, -2],
                           [2, -6, 8, -6, 2],
                           [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0
        
        # 滤波器3: 水平二阶
        filter3 = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 1, -2, 1, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]], dtype=np.float32) / 2.0
        
        # 应用滤波器
        response1 = cv2.filter2D(gray, -1, filter1)
        response2 = cv2.filter2D(gray, -1, filter2)
        response3 = cv2.filter2D(gray, -1, filter3)
        
        # 综合响应
        combined = np.abs(response1) + np.abs(response2) + np.abs(response3)
        
        def normalize_response(r):
            r = np.abs(r)
            r = r - r.min()
            if r.max() > 0:
                r = r / r.max()
            return (r * 255).astype(np.uint8)
        
        return {
            'srm_edge': normalize_response(response1),
            'srm_noise': normalize_response(response2),
            'srm_horizontal': normalize_response(response3),
            'srm_combined': normalize_response(combined),
        }


class SpatialVisualizer:
    """空间特征可视化器"""
    
    @staticmethod
    def visualize_edges(image: np.ndarray) -> Dict[str, np.ndarray]:
        """可视化边缘检测结果"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (256, 256))
        
        # Sobel边缘
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        
        # Canny边缘
        canny = cv2.Canny(gray, 50, 150)
        
        # Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        def normalize(img):
            img = np.abs(img)
            img = img - img.min()
            if img.max() > 0:
                img = img / img.max()
            return (img * 255).astype(np.uint8)
        
        return {
            'sobel_magnitude': normalize(sobel_mag),
            'canny_edges': canny,
            'laplacian': normalize(laplacian),
        }
    
    @staticmethod
    def visualize_texture(image: np.ndarray) -> Dict[str, np.ndarray]:
        """可视化纹理特征"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (256, 256))
        
        try:
            from skimage.feature import local_binary_pattern
            
            # LBP纹理
            lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
            lbp_normalized = ((lbp / lbp.max()) * 255).astype(np.uint8)
            
            return {
                'lbp_texture': lbp_normalized,
            }
        except ImportError:
            return {}


class MLFeatureVisualizer:
    """机器学习特征可视化器"""
    
    @staticmethod
    def visualize_all_features(image: np.ndarray) -> Dict[str, np.ndarray]:
        """可视化所有ML使用的特征"""
        results = {}
        
        # SRM特征
        srm_vis = FrequencyVisualizer.visualize_srm_filters(image)
        results.update(srm_vis)
        
        # 边缘特征
        edge_vis = SpatialVisualizer.visualize_edges(image)
        results.update(edge_vis)
        
        # 纹理特征
        texture_vis = SpatialVisualizer.visualize_texture(image)
        results.update(texture_vis)
        
        return results


def create_heatmap_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    将热力图叠加到原始图像上
    
    Args:
        image: 原始图像 (H, W, 3) BGR格式
        heatmap: 热力图 (H, W) 0-1范围
        alpha: 透明度
    
    Returns:
        叠加后的图像
    """
    # 确保热力图尺寸匹配
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # 将热力图转换为颜色映射
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # 叠加
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def create_visualization_grid(visualizations: Dict[str, np.ndarray], 
                             original_image: np.ndarray,
                             title: str = "Detection Visualization") -> np.ndarray:
    """
    创建可视化网格图
    
    Args:
        visualizations: 可视化结果字典
        original_image: 原始图像
        title: 标题
    
    Returns:
        网格图像
    """
    n_vis = len(visualizations) + 1  # +1 for original
    cols = min(4, n_vis)
    rows = (n_vis + cols - 1) // cols
    
    # 创建matplotlib图形
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle(title, fontsize=14)
    
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]
    
    # 显示原始图像
    if len(original_image.shape) == 3:
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # 显示可视化结果
    for i, (name, vis) in enumerate(visualizations.items(), 1):
        if i < len(axes):
            if len(vis.shape) == 3:
                axes[i].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            else:
                axes[i].imshow(vis, cmap='jet' if 'cam' in name.lower() else 'gray')
            axes[i].set_title(name.replace('_', ' ').title())
            axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(n_vis, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # 转换为numpy数组
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img_array


def get_detector_type(model_id: str) -> str:
    """根据模型ID获取检测器类型"""
    naive_models = ['xception', 'meso4', 'meso4Inception', 'resnet34', 'efficientnetb4']
    frequency_models = ['f3net', 'spsl', 'srm']
    spatial_models = ['capsule_net', 'ffd', 'core', 'recce', 'ucf']
    ml_models = ['svm_features']
    
    if model_id in naive_models:
        return 'naive'
    elif model_id in frequency_models:
        return 'frequency'
    elif model_id in spatial_models:
        return 'spatial'
    elif model_id in ml_models:
        return 'ml'
    else:
        return 'unknown'


def generate_visualization(model, model_id: str, image: np.ndarray, 
                          input_tensor: torch.Tensor = None) -> Dict[str, np.ndarray]:
    """
    根据模型类型生成相应的可视化
    
    Args:
        model: 检测模型
        model_id: 模型ID
        image: 原始图像 (BGR格式)
        input_tensor: 预处理后的输入张量
    
    Returns:
        可视化结果字典
    """
    detector_type = get_detector_type(model_id)
    visualizations = {}
    
    # 调整图像尺寸
    image_resized = cv2.resize(image, (256, 256))
    
    if detector_type == 'naive':
        # Grad-CAM可视化
        if model is not None and input_tensor is not None:
            try:
                grad_cam = GradCAM(model)
                cam = grad_cam.generate(input_tensor)
                if cam is not None:
                    # 调整CAM尺寸
                    cam_resized = cv2.resize(cam, (image_resized.shape[1], image_resized.shape[0]))
                    visualizations['grad_cam'] = (cam_resized * 255).astype(np.uint8)
                    visualizations['grad_cam_overlay'] = create_heatmap_overlay(image_resized, cam_resized)
            except Exception as e:
                print(f"Grad-CAM生成失败: {e}")
        
        # 基础边缘检测作为补充
        edge_vis = SpatialVisualizer.visualize_edges(image_resized)
        visualizations['edge_detection'] = edge_vis.get('sobel_magnitude', image_resized)
    
    elif detector_type == 'frequency':
        # 频域可视化
        dct_vis = FrequencyVisualizer.visualize_dct(image_resized)
        visualizations['dct_spectrum'] = dct_vis['dct_spectrum']
        visualizations['high_frequency'] = dct_vis['high_freq']
        
        # SRM滤波器响应
        srm_vis = FrequencyVisualizer.visualize_srm_filters(image_resized)
        visualizations['srm_noise_residual'] = srm_vis['srm_combined']
    
    elif detector_type == 'spatial':
        # Grad-CAM（如果可用）
        if model is not None and input_tensor is not None:
            try:
                grad_cam = GradCAM(model)
                cam = grad_cam.generate(input_tensor)
                if cam is not None:
                    cam_resized = cv2.resize(cam, (image_resized.shape[1], image_resized.shape[0]))
                    visualizations['attention_map'] = (cam_resized * 255).astype(np.uint8)
                    visualizations['attention_overlay'] = create_heatmap_overlay(image_resized, cam_resized)
            except Exception as e:
                print(f"注意力图生成失败: {e}")
        
        # 边缘和纹理
        edge_vis = SpatialVisualizer.visualize_edges(image_resized)
        visualizations['spatial_edges'] = edge_vis.get('canny_edges', image_resized)
        
        texture_vis = SpatialVisualizer.visualize_texture(image_resized)
        if 'lbp_texture' in texture_vis:
            visualizations['texture_pattern'] = texture_vis['lbp_texture']
    
    elif detector_type == 'ml':
        # 机器学习特征可视化
        ml_vis = MLFeatureVisualizer.visualize_all_features(image_resized)
        
        # 选择最重要的几个
        if 'srm_combined' in ml_vis:
            visualizations['srm_features'] = ml_vis['srm_combined']
        if 'sobel_magnitude' in ml_vis:
            visualizations['gradient_features'] = ml_vis['sobel_magnitude']
        if 'lbp_texture' in ml_vis:
            visualizations['lbp_texture'] = ml_vis['lbp_texture']
    
    else:
        # 未知类型，显示基础可视化
        edge_vis = SpatialVisualizer.visualize_edges(image_resized)
        visualizations['edge_analysis'] = edge_vis.get('sobel_magnitude', image_resized)
    
    return visualizations


# 用于WebUI的简化接口
def get_visualization_for_webui(model, model_id: str, image_bytes: bytes,
                                input_tensor: torch.Tensor = None) -> Tuple[Dict[str, str], str]:
    """
    生成用于WebUI显示的可视化结果
    
    Args:
        model: 检测模型
        model_id: 模型ID
        image_bytes: 图像字节数据
        input_tensor: 预处理后的张量
    
    Returns:
        Tuple[Dict[str, str], str]: (base64编码的可视化图像字典, 可视化类型描述)
    """
    import base64
    
    # 解码图像
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {}, "无法解码图像"
    
    # 生成可视化
    visualizations = generate_visualization(model, model_id, image, input_tensor)
    
    # 编码为base64
    result = {}
    for name, vis in visualizations.items():
        if len(vis.shape) == 2:
            # 灰度图转BGR
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        _, buffer = cv2.imencode('.jpg', vis)
        result[name] = base64.b64encode(buffer.tobytes()).decode('utf-8')
    
    # 生成类型描述
    detector_type = get_detector_type(model_id)
    type_descriptions = {
        'naive': '基础CNN模型 - 使用Grad-CAM显示网络关注区域',
        'frequency': '频域检测模型 - 展示DCT频谱和高频噪声残差',
        'spatial': '空间特征模型 - 显示注意力图和纹理特征',
        'ml': '传统机器学习模型 - 展示SRM、梯度和LBP纹理特征',
        'unknown': '通用分析 - 显示边缘检测结果'
    }
    
    return result, type_descriptions.get(detector_type, '未知模型类型')
