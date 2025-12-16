"""
图像特征检测器 - 推理脚本

使用训练好的 SVM 模型对新数据集进行推理和评估

使用方法：
1. 在 Jupyter Notebook 中运行
2. 或命令行: python image_feature_inference.py

数据集格式支持：
- UADFV: {dataset_root}/real/frames/{video_id}/{frame}.png
- celeb-df-v2: {dataset_root}/{category}/frames/{video_id}/{frame}.png
"""

import os
import sys
from typing import Optional, List, Tuple, Dict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.stats import kurtosis, skew
from tqdm import tqdm
import joblib

# GLCM graceful fallback
GLCM_AVAILABLE = True
try:
    from skimage.feature import local_binary_pattern, greycomatrix, greycoprops, hog
except ImportError:
    try:
        from skimage.feature import local_binary_pattern, hog
        from skimage.feature.texture import greycomatrix, greycoprops
    except Exception:
        from skimage.feature import local_binary_pattern, hog
        GLCM_AVAILABLE = False

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report


# ========== 配置 ==========
class InferenceConfig:
    # 模型路径
    MODEL_PATH = "E:/DeepfakeBench/image_feature_svm.joblib"
    
    # 数据集路径
    DATASET_ROOT = "E:/DeepfakeBench/datasets/rgb/UADFV"
    
    # 数据集类型: 'uadfv' 或 'celebdf'
    DATASET_TYPE = "uadfv"
    
    # 推理参数
    RESIZE = 256
    MAX_FRAMES_PER_VIDEO = 5  # 每个视频采样的帧数
    
    # 输出选项
    SAVE_RESULTS = True
    RESULTS_FILE = "inference_results.txt"
    SAVE_PREDICTIONS = True
    PREDICTIONS_FILE = "predictions.csv"


# ========== 特征提取（与训练时相同）==========
def _load_image_bgr(image_input, resize: Optional[int] = None) -> np.ndarray:
    """加载图像为 BGR 格式"""
    if isinstance(image_input, bytes):
        arr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, Image.Image):
        img = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img = image_input
    elif isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")
    
    if img is None:
        raise ValueError("Failed to load/decode image")
    if resize:
        img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_LINEAR)
    return img


# SRM 核心
SRM_KERNELS = np.stack([
    [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
    [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
    [[0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, -2, 4, -2, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0]],
], axis=0).astype(np.float32)
SRM_KERNELS[0] /= 4.0
SRM_KERNELS[1] /= 12.0
SRM_KERNELS[2] /= 4.0


def extract_srm_features(bgr: np.ndarray, bins: int = 32, clip: float = 3.0) -> np.ndarray:
    """提取 SRM 频域特征"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    feats = []
    for k in SRM_KERNELS:
        res = cv2.filter2D(gray, -1, k, borderType=cv2.BORDER_REFLECT)
        res = np.clip(res, -clip, clip)
        res_flat = res.reshape(-1)
        feats += [res_flat.mean(), res_flat.std() + 1e-6, skew(res_flat), kurtosis(res_flat)]
        hist, _ = np.histogram(res_flat, bins=bins, range=(-clip, clip), density=True)
        feats += hist.tolist()
    return np.array(feats, dtype=np.float32)


def extract_spatial_features(bgr: np.ndarray, face_box=None, resize: int = 256) -> np.ndarray:
    """提取空域显式特征"""
    if face_box:
        x1, y1, x2, y2 = face_box
        bgr = bgr[y1:y2, x1:x2]
    if resize:
        bgr = cv2.resize(bgr, (resize, resize))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    feats = []
    
    # LBP
    lbp = local_binary_pattern(gray, P=8, R=2, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, 8 + 3), range=(0, 8 + 2), density=True)
    feats += hist.tolist()
    
    # GLCM
    if GLCM_AVAILABLE:
        quant = (gray / 4).astype(np.uint8)
        glcm = greycomatrix(quant, distances=[1], angles=[0], levels=64, symmetric=True, normed=True)
        for prop in ["contrast", "homogeneity", "energy", "correlation"]:
            feats.append(greycoprops(glcm, prop)[0, 0])
    else:
        feats += [0.0, 0.0, 0.0, 0.0]
    
    # Gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    feats += [mag.mean(), mag.std()]
    
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    feats += [lap.mean(), lap.std(), skew(lap.reshape(-1)), kurtosis(lap.reshape(-1))]
    
    # HOG
    hog_feat = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=8, feature_vector=True)
    feats += hog_feat.tolist()
    
    # Color consistency
    means = bgr.reshape(-1, 3).mean(axis=0)
    stds = bgr.reshape(-1, 3).std(axis=0)
    feats += means.tolist() + stds.tolist()
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
        feats += (c_mean - b_mean).tolist()
    else:
        feats += [0.0, 0.0, 0.0]
    
    return np.array(feats, dtype=np.float32)


def extract_combined_features(image_input, resize: int = 256) -> np.ndarray:
    """提取组合特征（SRM + 空域）"""
    bgr = _load_image_bgr(image_input, resize=resize)
    srm = extract_srm_features(bgr)
    spatial = extract_spatial_features(bgr, resize=resize)
    return np.concatenate([srm, spatial], axis=0)


# ========== 数据加载 ==========
def collect_videos_uadfv(dataset_root: str) -> List[Tuple[str, str, int]]:
    """
    收集 UADFV 格式的视频
    返回: [(frame_dir, video_id, label), ...]
    """
    video_list = []
    
    # Real videos
    real_dir = os.path.join(dataset_root, "real", "frames")
    if os.path.exists(real_dir):
        video_ids = [d for d in os.listdir(real_dir) if os.path.isdir(os.path.join(real_dir, d))]
        for vid in video_ids:
            video_list.append((real_dir, vid, 0))
    
    # Fake videos
    fake_dir = os.path.join(dataset_root, "fake", "frames")
    if os.path.exists(fake_dir):
        video_ids = [d for d in os.listdir(fake_dir) if os.path.isdir(os.path.join(fake_dir, d))]
        for vid in video_ids:
            video_list.append((fake_dir, vid, 1))
    
    return video_list


def collect_videos_celebdf(dataset_root: str, real_dirs: List[str], fake_dirs: List[str]) -> List[Tuple[str, str, int]]:
    """
    收集 celeb-df-v2 格式的视频
    """
    video_list = []
    
    for frame_dir in real_dirs:
        full_path = os.path.join(dataset_root, frame_dir)
        if os.path.exists(full_path):
            video_ids = [d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))]
            for vid in video_ids:
                video_list.append((full_path, vid, 0))
    
    for frame_dir in fake_dirs:
        full_path = os.path.join(dataset_root, frame_dir)
        if os.path.exists(full_path):
            video_ids = [d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))]
            for vid in video_ids:
                video_list.append((full_path, vid, 1))
    
    return video_list


def sample_frames_from_video(frame_dir: str, video_id: str, max_frames: int) -> List[str]:
    """从视频文件夹中均匀采样帧"""
    video_path = os.path.join(frame_dir, video_id)
    frame_files = sorted([
        f for f in os.listdir(video_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    if len(frame_files) == 0:
        return []
    
    if len(frame_files) > max_frames:
        indices = np.linspace(0, len(frame_files) - 1, max_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]
    
    return [os.path.join(video_path, f) for f in frame_files]


# ========== 推理引擎 ==========
class ImageFeatureInference:
    def __init__(self, model_path: str):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        print("✓ Model loaded successfully!")
    
    def predict_frame(self, image_path: str, resize: int = 256) -> Tuple[int, float]:
        """
        预测单帧
        返回: (predicted_label, probability)
        """
        feat = extract_combined_features(image_path, resize=resize)
        pred = self.model.predict(feat[None, :])[0]
        
        # 计算概率
        if hasattr(self.model, 'decision_function'):
            decision = self.model.decision_function(feat[None, :])[0]
            prob = 1 / (1 + np.exp(-decision))
        else:
            prob = float(pred)
        
        return int(pred), float(prob)
    
    def predict_video(self, frame_paths: List[str], resize: int = 256) -> Tuple[int, float, List[float]]:
        """
        预测视频（多帧聚合）
        返回: (predicted_label, avg_probability, frame_probabilities)
        """
        frame_probs = []
        
        for frame_path in frame_paths:
            try:
                _, prob = self.predict_frame(frame_path, resize)
                frame_probs.append(prob)
            except Exception as e:
                print(f"[Warning] Failed to process {frame_path}: {e}")
                continue
        
        if len(frame_probs) == 0:
            return 0, 0.0, []
        
        # 平均概率聚合
        avg_prob = np.mean(frame_probs)
        pred_label = 1 if avg_prob > 0.5 else 0
        
        return pred_label, avg_prob, frame_probs
    
    def evaluate_dataset(
        self,
        video_list: List[Tuple[str, str, int]],
        max_frames_per_video: int,
        resize: int,
        save_predictions: bool = False,
        predictions_file: str = "predictions.csv"
    ) -> Dict:
        """
        评估整个数据集
        """
        y_true = []
        y_pred = []
        y_prob = []
        video_results = []
        
        print(f"\nEvaluating {len(video_list)} videos...")
        
        for frame_dir, video_id, true_label in tqdm(video_list, desc="Inference"):
            frame_paths = sample_frames_from_video(frame_dir, video_id, max_frames_per_video)
            
            if len(frame_paths) == 0:
                continue
            
            pred_label, avg_prob, frame_probs = self.predict_video(frame_paths, resize)
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            y_prob.append(avg_prob)
            
            video_results.append({
                'video_id': video_id,
                'true_label': true_label,
                'pred_label': pred_label,
                'probability': avg_prob,
                'num_frames': len(frame_probs)
            })
        
        # 保存预测结果
        if save_predictions:
            self._save_predictions(video_results, predictions_file)
        
        # 计算指标
        metrics = self._compute_metrics(y_true, y_pred, y_prob)
        
        return {
            'metrics': metrics,
            'video_results': video_results,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    def _save_predictions(self, video_results: List[Dict], filename: str):
        """保存预测结果到 CSV"""
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['video_id', 'true_label', 'pred_label', 'probability', 'num_frames'])
            writer.writeheader()
            writer.writerows(video_results)
        
        print(f"✓ Predictions saved to: {filename}")
    
    def _compute_metrics(self, y_true, y_pred, y_prob):
        """计算评估指标"""
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.0
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'specificity': specificity,
            'fpr': fpr,
            'fnr': fnr,
            'confusion_matrix': cm,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }


# ========== 主函数 ==========
def main():
    print("\n" + "="*70)
    print("Image Feature Detector - Inference on New Dataset")
    print("="*70 + "\n")
    
    # 显示配置
    print("Configuration:")
    print(f"  Model Path:     {InferenceConfig.MODEL_PATH}")
    print(f"  Dataset Root:   {InferenceConfig.DATASET_ROOT}")
    print(f"  Dataset Type:   {InferenceConfig.DATASET_TYPE}")
    print(f"  Max Frames:     {InferenceConfig.MAX_FRAMES_PER_VIDEO}")
    print()
    
    # 加载模型
    inference = ImageFeatureInference(InferenceConfig.MODEL_PATH)
    
    # 收集视频
    print("Collecting videos...")
    if InferenceConfig.DATASET_TYPE.lower() == "uadfv":
        video_list = collect_videos_uadfv(InferenceConfig.DATASET_ROOT)
    elif InferenceConfig.DATASET_TYPE.lower() == "celebdf":
        # 需要手动指定 real_dirs 和 fake_dirs
        real_dirs = ["Celeb-real/frames", "YouTube-real/frames"]
        fake_dirs = ["Celeb-synthesis/frames"]
        video_list = collect_videos_celebdf(InferenceConfig.DATASET_ROOT, real_dirs, fake_dirs)
    else:
        raise ValueError(f"Unsupported dataset type: {InferenceConfig.DATASET_TYPE}")
    
    labels = [label for _, _, label in video_list]
    print(f"  Total videos: {len(video_list)}")
    print(f"    Real: {labels.count(0)}")
    print(f"    Fake: {labels.count(1)}")
    
    # 推理
    results = inference.evaluate_dataset(
        video_list,
        InferenceConfig.MAX_FRAMES_PER_VIDEO,
        InferenceConfig.RESIZE,
        save_predictions=InferenceConfig.SAVE_PREDICTIONS,
        predictions_file=InferenceConfig.PREDICTIONS_FILE
    )
    
    # 显示结果
    metrics = results['metrics']
    
    print("\n" + "="*70)
    print("Evaluation Results:")
    print("="*70)
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}  (假样本预测准确率)")
    print(f"  Recall:      {metrics['recall']:.4f}  (假样本召回率)")
    print(f"  F1-Score:    {metrics['f1']:.4f}")
    print(f"  AUC-ROC:     {metrics['auc']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}  (真样本识别率)")
    print(f"\nError Rates:")
    print(f"  FPR (假阳性率): {metrics['fpr']:.4f}  ({metrics['fp']}/{metrics['fp']+metrics['tn']} 真样本被误判)")
    print(f"  FNR (假阴性率): {metrics['fnr']:.4f}  ({metrics['fn']}/{metrics['fn']+metrics['tp']} 假样本被漏检)")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Real    Fake")
    print(f"  Actual Real   {metrics['tn']:5d}   {metrics['fp']:5d}")
    print(f"        Fake    {metrics['fn']:5d}   {metrics['tp']:5d}")
    print("="*70 + "\n")
    
    # 保存结果
    if InferenceConfig.SAVE_RESULTS:
        with open(InferenceConfig.RESULTS_FILE, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("Image Feature Detector - Inference Results\n")
            f.write("="*70 + "\n\n")
            f.write(f"Model: {InferenceConfig.MODEL_PATH}\n")
            f.write(f"Dataset: {InferenceConfig.DATASET_ROOT}\n")
            f.write(f"Dataset Type: {InferenceConfig.DATASET_TYPE}\n\n")
            f.write(f"Accuracy:    {metrics['accuracy']:.4f}\n")
            f.write(f"Precision:   {metrics['precision']:.4f}\n")
            f.write(f"Recall:      {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:    {metrics['f1']:.4f}\n")
            f.write(f"AUC-ROC:     {metrics['auc']:.4f}\n")
            f.write(f"Specificity: {metrics['specificity']:.4f}\n\n")
            f.write(f"Confusion Matrix:\n")
            f.write(f"  TN: {metrics['tn']:5d}  FP: {metrics['fp']:5d}\n")
            f.write(f"  FN: {metrics['fn']:5d}  TP: {metrics['tp']:5d}\n")
        
        print(f"✓ Results saved to: {InferenceConfig.RESULTS_FILE}")
    
    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
