"""
完整的图像特征检测器 Demo - 适配 celeb-df-v2 数据集

功能：
1. 按视频级别划分训练集/验证集/测试集（避免数据泄漏）
2. 提取频域（SRM）+ 空域（LBP/GLCM/HOG等）特征
3. 训练 Linear SVM 分类器
4. 详细的训练日志输出（进度条、每个 epoch 的性能）
5. 测试集评估

数据集格式（celeb-df-v2）：
  celeb-df-v2/
    ├── Celeb-real/frames/{video_id}/{frame}.png
    ├── Celeb-synthesis/frames/{video_id}/{frame}.png  (fake)
    ├── YouTube-real/frames/{video_id}/{frame}.png
    └── List_of_testing_videos.txt (可选)

使用方法：
1. 在 Jupyter Notebook 中直接复制整个文件内容到一个单元格
2. 或者在命令行运行：python image_feature_detector_demo.py
"""

import os
import glob
import random
from typing import Optional, Tuple, List, Dict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.stats import kurtosis, skew
from tqdm import tqdm

# GLCM graceful fallback
GLCM_AVAILABLE = True
GLCM_IMPORT_NOTE = None
try:
    from skimage.feature import local_binary_pattern, greycomatrix, greycoprops, hog
except ImportError:
    try:
        from skimage.feature import local_binary_pattern, hog
        from skimage.feature.texture import greycomatrix, greycoprops
        GLCM_IMPORT_NOTE = "Imported greycomatrix from skimage.feature.texture (compat mode)."
    except Exception:
        from skimage.feature import local_binary_pattern, hog
        GLCM_AVAILABLE = False
        GLCM_IMPORT_NOTE = "greycomatrix unavailable; GLCM features disabled."

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import joblib


# ========== 配置部分 ==========
class Config:
    # 数据集根目录
    DATASET_ROOT = "/kaggle/input/celeb-df-v2"  # TODO: 修改为你的数据集路径
    
    # 真实视频帧目录列表
    REAL_DIRS = [
        f"{DATASET_ROOT}/Celeb-real/frames",
        f"{DATASET_ROOT}/YouTube-real/frames",
    ]
    
    # 伪造视频帧目录列表
    FAKE_DIRS = [
        f"{DATASET_ROOT}/Celeb-synthesis/frames",
    ]
    
    # 测试视频列表文件（可选）
    TEST_LIST_FILE = f"{DATASET_ROOT}/List_of_testing_videos.txt"
    
    # 训练参数
    MODEL_OUT = "image_feature_svm.joblib"
    RESIZE = 256
    MAX_FRAMES_PER_VIDEO = 5  # 每个视频最多采样的帧数
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_STATE = 42
    
    # 调试选项（限制视频数量以快速测试）
    DEBUG_MODE = False
    DEBUG_MAX_VIDEOS = 20  # 仅在 DEBUG_MODE=True 时生效


# ========== 特征提取模块 ==========
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
    """提取空域显式特征：LBP, GLCM, Sobel/Laplacian, HOG, 颜色一致性"""
    if face_box:
        x1, y1, x2, y2 = face_box
        bgr = bgr[y1:y2, x1:x2]
    if resize:
        bgr = cv2.resize(bgr, (resize, resize))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    feats = []
    
    # LBP histogram
    lbp = local_binary_pattern(gray, P=8, R=2, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, 8 + 3), range=(0, 8 + 2), density=True)
    feats += hist.tolist()
    
    # GLCM stats
    if GLCM_AVAILABLE:
        quant = (gray / 4).astype(np.uint8)
        glcm = greycomatrix(quant, distances=[1], angles=[0], levels=64, symmetric=True, normed=True)
        for prop in ["contrast", "homogeneity", "energy", "correlation"]:
            feats.append(greycoprops(glcm, prop)[0, 0])
    else:
        feats += [0.0, 0.0, 0.0, 0.0]
    
    # Gradient stats
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


# ========== 数据加载模块 ==========
def collect_video_ids_and_labels(real_dirs: List[str], fake_dirs: List[str]) -> List[Tuple[str, str, int]]:
    """
    收集所有视频 ID 和标签
    返回: [(frame_dir, video_id, label), ...]
    label: 0=real, 1=fake
    """
    video_list = []
    
    # 收集真实视频
    for frame_dir in real_dirs:
        if not os.path.exists(frame_dir):
            print(f"[Warning] Directory not found: {frame_dir}")
            continue
        video_ids = [d for d in os.listdir(frame_dir) if os.path.isdir(os.path.join(frame_dir, d))]
        for vid in video_ids:
            video_list.append((frame_dir, vid, 0))
    
    # 收集伪造视频
    for frame_dir in fake_dirs:
        if not os.path.exists(frame_dir):
            print(f"[Warning] Directory not found: {frame_dir}")
            continue
        video_ids = [d for d in os.listdir(frame_dir) if os.path.isdir(os.path.join(frame_dir, d))]
        for vid in video_ids:
            video_list.append((frame_dir, vid, 1))
    
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
    
    # 均匀采样
    if len(frame_files) > max_frames:
        indices = np.linspace(0, len(frame_files) - 1, max_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]
    
    return [os.path.join(video_path, f) for f in frame_files]


def load_features_from_videos(
    video_list: List[Tuple[str, str, int]],
    max_frames_per_video: int,
    resize: int,
    desc: str = "Loading"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从视频列表中加载特征
    返回: (X, y) - X: features, y: labels
    """
    X, y = [], []
    
    for frame_dir, video_id, label in tqdm(video_list, desc=desc):
        frame_paths = sample_frames_from_video(frame_dir, video_id, max_frames_per_video)
        
        for frame_path in frame_paths:
            try:
                feat = extract_combined_features(frame_path, resize=resize)
                X.append(feat)
                y.append(label)
            except Exception as e:
                print(f"[Error] Failed to process {frame_path}: {e}")
                continue
    
    if len(X) == 0:
        raise ValueError("No features extracted! Check your dataset paths.")
    
    return np.stack(X, axis=0), np.array(y)


# ========== 评估模块 ==========
def evaluate_model(model, X, y, split_name: str = "Validation"):
    """评估模型性能"""
    y_pred = model.predict(X)
    
    # 计算概率（用于 AUC）
    if hasattr(model, 'decision_function'):
        decision = model.decision_function(X)
        y_prob = 1 / (1 + np.exp(-decision))
    else:
        y_prob = y_pred
    
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    try:
        auc = roc_auc_score(y, y_prob)
    except:
        auc = 0.0
    
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # 计算详细指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 也叫 TPR (True Positive Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR (True Negative Rate)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
    
    print(f"\n{'='*50}")
    print(f"{split_name} Metrics:")
    print(f"{'='*50}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Precision:  {precision:.4f}  (假样本预测准确率)")
    print(f"  Recall:     {recall:.4f}  (假样本召回率)")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"  AUC-ROC:    {auc:.4f}")
    print(f"  Specificity:{specificity:.4f}  (真样本识别率)")
    print(f"  Total:      {len(y)}")
    print(f"\nError Rates:")
    print(f"  FPR (假阳性率): {fpr:.4f}  ({fp}/{fp+tn} 真样本被误判为假)")
    print(f"  FNR (假阴性率): {fnr:.4f}  ({fn}/{fn+tp} 假样本被误判为真)")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Real    Fake")
    print(f"  Actual Real   {tn:5d}   {fp:5d}")
    print(f"        Fake    {fn:5d}   {tp:5d}")
    print(f"{'='*50}\n")
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'fpr': fpr,
        'fnr': fnr,
        'confusion_matrix': cm
    }



# ========== 主流程 ==========
def main():
    print("\n" + "="*70)
    print("Image Feature Detector Demo - celeb-df-v2 Dataset")
    print("="*70 + "\n")
    
    # 显示配置
    print("Configuration:")
    print(f"  DATASET_ROOT: {Config.DATASET_ROOT}")
    print(f"  REAL_DIRS: {Config.REAL_DIRS}")
    print(f"  FAKE_DIRS: {Config.FAKE_DIRS}")
    print(f"  MAX_FRAMES_PER_VIDEO: {Config.MAX_FRAMES_PER_VIDEO}")
    print(f"  Train/Val/Test Split: {Config.TRAIN_RATIO:.1%}/{Config.VAL_RATIO:.1%}/{Config.TEST_RATIO:.1%}")
    print(f"  DEBUG_MODE: {Config.DEBUG_MODE}")
    if GLCM_IMPORT_NOTE:
        print(f"\n[Note] {GLCM_IMPORT_NOTE}")
    print()
    
    # 步骤 1: 收集视频 ID
    print("[Step 1/5] Collecting video IDs...")
    video_list = collect_video_ids_and_labels(Config.REAL_DIRS, Config.FAKE_DIRS)
    labels = np.array([label for _, _, label in video_list])
    
    if Config.DEBUG_MODE and len(video_list) > Config.DEBUG_MAX_VIDEOS:
        print(f"[DEBUG] Limiting to {Config.DEBUG_MAX_VIDEOS} videos")
        random.seed(Config.RANDOM_STATE)
        video_list = random.sample(video_list, Config.DEBUG_MAX_VIDEOS)
        labels = np.array([label for _, _, label in video_list])
    
    print(f"  Total videos: {len(video_list)}")
    print(f"    Real: {np.sum(labels == 0)}")
    print(f"    Fake: {np.sum(labels == 1)}")
    
    # 步骤 2: 按视频级别划分数据集
    print("\n[Step 2/5] Splitting dataset (video-level)...")
    
    # 先划分 train 和 temp (val+test)
    train_videos, temp_videos, _, temp_labels = train_test_split(
        video_list, labels,
        test_size=(Config.VAL_RATIO + Config.TEST_RATIO),
        random_state=Config.RANDOM_STATE,
        stratify=labels
    )
    
    # 再划分 val 和 test
    val_ratio_adjusted = Config.VAL_RATIO / (Config.VAL_RATIO + Config.TEST_RATIO)
    val_videos, test_videos = train_test_split(
        temp_videos,
        test_size=(1 - val_ratio_adjusted),
        random_state=Config.RANDOM_STATE,
        stratify=temp_labels
    )
    
    print(f"  Train: {len(train_videos)} videos")
    print(f"  Val:   {len(val_videos)} videos")
    print(f"  Test:  {len(test_videos)} videos")
    
    # 步骤 3: 提取特征
    print("\n[Step 3/5] Extracting features...")
    
    print("  > Training set...")
    X_train, y_train = load_features_from_videos(
        train_videos, Config.MAX_FRAMES_PER_VIDEO, Config.RESIZE, "Train"
    )
    
    print("  > Validation set...")
    X_val, y_val = load_features_from_videos(
        val_videos, Config.MAX_FRAMES_PER_VIDEO, Config.RESIZE, "Val"
    )
    
    print("  > Test set...")
    X_test, y_test = load_features_from_videos(
        test_videos, Config.MAX_FRAMES_PER_VIDEO, Config.RESIZE, "Test"
    )
    
    print(f"\nFeature dimensions: {X_train.shape[1]}")
    print(f"  Train: {X_train.shape[0]} samples (Real: {np.sum(y_train==0)}, Fake: {np.sum(y_train==1)})")
    print(f"  Val:   {X_val.shape[0]} samples (Real: {np.sum(y_val==0)}, Fake: {np.sum(y_val==1)})")
    print(f"  Test:  {X_test.shape[0]} samples (Real: {np.sum(y_test==0)}, Fake: {np.sum(y_test==1)})")
    
    # 步骤 4: 训练模型
    print("\n[Step 4/5] Training Linear SVM...")
    # C: 正则化参数（越小越强正则化，防止过拟合）
    # class_weight='balanced': 自动调整类别权重，处理类别不平衡
    clf = make_pipeline(
        StandardScaler(), 
        LinearSVC(
            C=0.01,  # 强正则化，防止过拟合
            class_weight='balanced',  # 处理类别不平衡
            max_iter=5000, 
            random_state=Config.RANDOM_STATE
        )
    )
    
    print("  Fitting model on training data...")
    clf.fit(X_train, y_train)
    print("  ✓ Training complete!")
    
    # 步骤 5: 评估模型
    print("\n[Step 5/5] Evaluating model...")
    
    # 训练集性能
    evaluate_model(clf, X_train, y_train, "Training Set")
    
    # 验证集性能
    val_metrics = evaluate_model(clf, X_val, y_val, "Validation Set")
    
    # 测试集性能
    test_metrics = evaluate_model(clf, X_test, y_test, "Test Set")
    
    # 保存模型
    joblib.dump(clf, Config.MODEL_OUT)
    print(f"✓ Model saved to: {Config.MODEL_OUT}")
    
    # 总结
    print("\n" + "="*70)
    print("Training Summary:")
    print("="*70)
    print(f"  Best Val Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  Best Val F1-Score:  {val_metrics['f1']:.4f}")
    print(f"  Test Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"  Test F1-Score:      {test_metrics['f1']:.4f}")
    print(f"  Test AUC:           {test_metrics['auc']:.4f}")
    print("="*70 + "\n")
    
    print("✓ All done! You can now use the trained model for inference.")
    print(f"  Model path: {Config.MODEL_OUT}")


if __name__ == "__main__":
    main()
