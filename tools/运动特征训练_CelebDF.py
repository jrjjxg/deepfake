"""
运动特征检测器 - Celeb-DF-v2 训练专用脚本
Motion Feature Detector - Celeb-DF-v2 Training Script

数据集结构：
  /kaggle/input/celeb-df-v2/
    ├── Celeb-real/frames/
    ├── Celeb-synthesis/frames/
    └── YouTube-real/frames/

使用方法:
1. 在 Kaggle 添加数据集: celeb-df-v2
2. 修改 Config.DATASET_ROOT (默认已适配)
3. 直接运行
"""

import os
import sys
from collections import defaultdict
import random
from typing import Optional, Tuple, List, Dict
import time

import cv2
import numpy as np
from scipy.stats import kurtosis, skew
from tqdm import tqdm

import mediapipe as mp

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib


# ========== Configuration ==========
class Config:
    # Kaggle dataset path
    DATASET_ROOT = "/kaggle/input/celeb-df-v2"
    
    # Training parameters
    MODEL_OUT = "motion_feature_model_celebdf.joblib"
    MAX_FRAMES_PER_VIDEO = 60
    MIN_FRAMES_PER_VIDEO = 10
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_STATE = 42
    
    # Feature extraction
    EAR_THRESHOLD = 0.21
    BLINK_CONSEC_FRAMES = 2
    
    # Debug
    DEBUG_MODE = False
    DEBUG_MAX_VIDEOS = 50


# ========== MediaPipe Setup ==========
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

JITTER_LANDMARKS = {
    'nose_tip': 1,
    'left_eye_outer': 263,
    'right_eye_outer': 33,
    'left_mouth': 61,
    'right_mouth': 291,
    'chin': 152,
    'left_eyebrow': 70,
    'right_eyebrow': 300
}


# ========== Data Collection ==========
def collect_videos_from_celebdf(dataset_root: str) -> List[Tuple[str, List[str], int]]:
    """
    Collect videos from Celeb-DF-v2 dataset.
    Returns: [(video_id, frame_paths, label), ...]
    """
    video_list = []
    
    # Process fake videos (Celeb-synthesis)
    fake_dir = os.path.join(dataset_root, "Celeb-synthesis", "frames")
    if os.path.exists(fake_dir):
        print(f"  Scanning fake videos in: {fake_dir}")
        
        video_folders = [d for d in os.listdir(fake_dir) 
                       if os.path.isdir(os.path.join(fake_dir, d))]
        
        valid_count = 0
        for video_id in video_folders:
            video_path = os.path.join(fake_dir, video_id)
            
            frame_files = sorted([
                os.path.join(video_path, f)
                for f in os.listdir(video_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            
            if len(frame_files) >= Config.MIN_FRAMES_PER_VIDEO:
                video_list.append((f"fake_{video_id}", frame_files, 1))
                valid_count += 1
        
        print(f"    Found {valid_count} fake videos")
    
    # Process real videos (Celeb-real + YouTube-real)
    for real_subdir in ["Celeb-real", "YouTube-real"]:
        real_dir = os.path.join(dataset_root, real_subdir, "frames")
        if os.path.exists(real_dir):
            print(f"  Scanning real videos in: {real_dir}")
            
            video_folders = [d for d in os.listdir(real_dir) 
                           if os.path.isdir(os.path.join(real_dir, d))]
            
            valid_count = 0
            for video_id in video_folders:
                video_path = os.path.join(real_dir, video_id)
                
                frame_files = sorted([
                    os.path.join(video_path, f)
                    for f in os.listdir(video_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                
                if len(frame_files) >= Config.MIN_FRAMES_PER_VIDEO:
                    video_list.append((f"real_{real_subdir}_{video_id}", frame_files, 0))
                    valid_count += 1
            
            print(f"    Found {valid_count} real videos")
    
    return video_list


# ========== Feature Extraction ==========
def compute_ear(eye_landmarks: np.ndarray) -> float:
    """Compute Eye Aspect Ratio."""
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    if h < 1e-6:
        return 0.0
    
    return (v1 + v2) / (2.0 * h)


def extract_eye_landmarks(face_landmarks, indices: List[int], img_w: int, img_h: int) -> np.ndarray:
    """Extract eye landmarks from MediaPipe."""
    points = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        points.append([int(lm.x * img_w), int(lm.y * img_h)])
    return np.array(points, dtype=np.float32)


def extract_landmarks_from_frame(frame: np.ndarray, face_mesh) -> Optional[Dict]:
    """Extract facial landmarks from frame."""
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return None
    
    face_landmarks = results.multi_face_landmarks[0]
    
    left_eye = extract_eye_landmarks(face_landmarks, LEFT_EYE_INDICES, w, h)
    right_eye = extract_eye_landmarks(face_landmarks, RIGHT_EYE_INDICES, w, h)
    
    jitter_points = {}
    for name, idx in JITTER_LANDMARKS.items():
        lm = face_landmarks.landmark[idx]
        jitter_points[name] = np.array([lm.x * w, lm.y * h], dtype=np.float32)
    
    return {
        'left_eye': left_eye,
        'right_eye': right_eye,
        'jitter_points': jitter_points
    }


def extract_video_features(frame_paths: List[str], face_mesh) -> Optional[np.ndarray]:
    """Extract motion features from video frames."""
    if len(frame_paths) < Config.MIN_FRAMES_PER_VIDEO:
        return None
    
    # Sample frames if too many
    if len(frame_paths) > Config.MAX_FRAMES_PER_VIDEO:
        indices = np.linspace(0, len(frame_paths) - 1, Config.MAX_FRAMES_PER_VIDEO, dtype=int)
        frame_paths = [frame_paths[i] for i in indices]
    
    ear_values = []
    jitter_sequences = defaultdict(list)
    
    for frame_path in frame_paths:
        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            landmarks = extract_landmarks_from_frame(frame, face_mesh)
            if landmarks is None:
                continue
            
            # EAR
            left_ear = compute_ear(landmarks['left_eye'])
            right_ear = compute_ear(landmarks['right_eye'])
            avg_ear = (left_ear + right_ear) / 2.0
            ear_values.append(avg_ear)
            
            # Jitter points
            for name, point in landmarks['jitter_points'].items():
                jitter_sequences[name].append(point)
        except:
            continue
    
    if len(ear_values) < Config.MIN_FRAMES_PER_VIDEO:
        return None
    
    # EAR features
    ear_array = np.array(ear_values)
    ear_mean = np.mean(ear_array)
    ear_std = np.std(ear_array)
    ear_min = np.min(ear_array)
    ear_max = np.max(ear_array)
    ear_range = ear_max - ear_min
    
    # Blink detection
    blink_count = 0
    below_threshold_count = 0
    for ear in ear_array:
        if ear < Config.EAR_THRESHOLD:
            below_threshold_count += 1
        else:
            if below_threshold_count >= Config.BLINK_CONSEC_FRAMES:
                blink_count += 1
            below_threshold_count = 0
    
    blink_rate = blink_count / (len(ear_array) / 60.0) if len(ear_array) > 0 else 0
    ear_skewness = skew(ear_array) if len(ear_array) > 2 else 0
    ear_kurtosis = kurtosis(ear_array) if len(ear_array) > 3 else 0
    
    ear_diff = np.diff(ear_array)
    ear_velocity_mean = np.mean(np.abs(ear_diff)) if len(ear_diff) > 0 else 0
    ear_velocity_std = np.std(ear_diff) if len(ear_diff) > 0 else 0
    
    ear_features = [
        ear_mean, ear_std, ear_min, ear_max, ear_range,
        blink_count, blink_rate,
        ear_skewness, ear_kurtosis,
        ear_velocity_mean, ear_velocity_std
    ]
    
    # Jitter features
    jitter_features = []
    for name in JITTER_LANDMARKS.keys():
        if name not in jitter_sequences or len(jitter_sequences[name]) < 3:
            jitter_features.extend([0.0] * 6)
            continue
        
        points = np.array(jitter_sequences[name])
        
        # Global motion compensation
        if 'nose_tip' in jitter_sequences and len(jitter_sequences['nose_tip']) == len(points):
            nose_points = np.array(jitter_sequences['nose_tip'])
            points = points - nose_points
        
        displacements = np.linalg.norm(np.diff(points, axis=0), axis=1)
        
        if len(displacements) > 0:
            disp_mean = np.mean(displacements)
            disp_std = np.std(displacements)
            disp_max = np.max(displacements)
            
            if len(displacements) > 1:
                accelerations = np.abs(np.diff(displacements))
                accel_mean = np.mean(accelerations)
                accel_std = np.std(accelerations)
                accel_max = np.max(accelerations)
            else:
                accel_mean, accel_std, accel_max = 0, 0, 0
        else:
            disp_mean, disp_std, disp_max = 0, 0, 0
            accel_mean, accel_std, accel_max = 0, 0, 0
        
        jitter_features.extend([
            disp_mean, disp_std, disp_max,
            accel_mean, accel_std, accel_max
        ])
    
    # Concatenate all features
    return np.array(ear_features + jitter_features, dtype=np.float32)


def load_features_from_videos(
    video_list: List[Tuple[str, List[str], int]],
    desc: str = "Extracting",
    cache_file: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Load features from video list with caching."""
    # Try to load from cache
    if cache_file and os.path.exists(cache_file):
        print(f"  Loading cached features from {cache_file}...")
        data = np.load(cache_file)
        return data['X'], data['y']
    
    X, y = [], []
    success_count = 0
    fail_count = 0
    
    print(f"  Processing {len(video_list)} videos...")
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        for idx, (video_id, frame_paths, label) in enumerate(video_list):
            features = extract_video_features(frame_paths, face_mesh)
            
            if features is not None:
                X.append(features)
                y.append(label)
                success_count += 1
            else:
                fail_count += 1
            
            # Progress update every 50 videos
            if (idx + 1) % 50 == 0 or (idx + 1) == len(video_list):
                success_rate = success_count / (idx + 1) * 100
                print(f"    [{idx+1}/{len(video_list)}] Success: {success_count}, Failed: {fail_count}, Rate: {success_rate:.1f}%")
    
    if len(X) == 0:
        raise ValueError("No features extracted!")
    
    X_array = np.stack(X, axis=0)
    y_array = np.array(y)
    
    # Save to cache
    if cache_file:
        print(f"  Saving features to cache: {cache_file}")
        np.savez_compressed(cache_file, X=X_array, y=y_array)
    
    return X_array, y_array


# ========== Evaluation ==========
def evaluate_model(model, X, y, split_name: str = "Validation"):
    """Evaluate model."""
    y_pred = model.predict(X)
    
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, 'decision_function'):
        decision = model.decision_function(X)
        y_prob = 1 / (1 + np.exp(-decision))
    else:
        y_prob = y_pred.astype(float)
    
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    try:
        auc = roc_auc_score(y, y_prob)
    except:
        auc = 0.0
    
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    print(f"\n{'='*50}")
    print(f"{split_name} Metrics:")
    print(f"{'='*50}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  Specificity: {specificity:.4f}")
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
        'specificity': specificity
    }


# ========== Main ==========
def main():
    print("\n" + "="*70)
    print("Motion Feature Detector - Celeb-DF-v2 Training")
    print("="*70 + "\n")
    
    # Collect videos
    print("[Step 1/5] Collecting videos from dataset...")
    video_list = collect_videos_from_celebdf(Config.DATASET_ROOT)
    
    if Config.DEBUG_MODE:
        print(f"[DEBUG] Limiting to {Config.DEBUG_MAX_VIDEOS} videos")
        random.seed(Config.RANDOM_STATE)
        video_list = random.sample(video_list, min(len(video_list), Config.DEBUG_MAX_VIDEOS))
    
    labels = [v[2] for v in video_list]
    print(f"  Total videos: {len(video_list)}")
    print(f"    Real: {labels.count(0)}")
    print(f"    Fake: {labels.count(1)}")
    
    # Split
    print("\n[Step 2/5] Splitting dataset...")
    train_videos, temp_videos, _, temp_labels = train_test_split(
        video_list, labels,
        test_size=(Config.VAL_RATIO + Config.TEST_RATIO),
        random_state=Config.RANDOM_STATE,
        stratify=labels
    )
    
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
    
    # Extract features
    print("\n[Step 3/5] Extracting motion features...")
    print("  (This may take a while. Features will be cached for reuse.)")
    
    print("  > Training set...")
    t0 = time.time()
    X_train, y_train = load_features_from_videos(
        train_videos, "Train", cache_file="train_features_celebdf.npz"
    )
    print(f"    Time elapsed: {time.time() - t0:.1f}s")
    
    print("  > Validation set...")
    t0 = time.time()
    X_val, y_val = load_features_from_videos(
        val_videos, "Val", cache_file="val_features_celebdf.npz"
    )
    print(f"    Time elapsed: {time.time() - t0:.1f}s")
    
    print("  > Test set...")
    t0 = time.time()
    X_test, y_test = load_features_from_videos(
        test_videos, "Test", cache_file="test_features_celebdf.npz"
    )
    print(f"    Time elapsed: {time.time() - t0:.1f}s")
    
    # Train
    print("\n[Step 4/5] Optimizing SVM with GridSearchCV...")
    
    # Define SVM Pipeline
    svm_pipeline = make_pipeline(
        StandardScaler(),
        SVC(
            class_weight='balanced',
            probability=True,
            random_state=Config.RANDOM_STATE
        )
    )
    
    # Parameter Grid
    param_grid = [
        # Strategy 1: RBF Kernel
        {
            'svc__kernel': ['rbf'],
            'svc__C': [0.1, 1, 10, 100],
            'svc__gamma': ['scale', 'auto', 0.01, 0.1]
        },
        # Strategy 2: Linear Kernel
        {
            'svc__kernel': ['linear'],
            'svc__C': [0.1, 1, 10, 100]
        }
    ]
    
    print("  Searching for best hyperparameters...")
    t0 = time.time()
    
    grid = GridSearchCV(
        svm_pipeline,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    training_time = time.time() - t0
    best_model = grid.best_estimator_
    
    print(f"    Optimization completed in {training_time:.1f}s")
    print(f"    Best CV AUC: {grid.best_score_:.4f}")
    print(f"    Best Parameters: {grid.best_params_}")
    
    # Evaluate Best Model
    print("\n[Step 5/5] Evaluating Best SVM Model...")
    
    # evaluate_model(best_model, X_train, y_train, "Training Set")
    val_metrics = evaluate_model(best_model, X_val, y_val, "Validation Set")
    test_metrics = evaluate_model(best_model, X_test, y_test, "Test Set")
    
    # Save Model
    joblib.dump(best_model, Config.MODEL_OUT)
    print(f"\nModel saved to: {Config.MODEL_OUT}")
    
    # Summary
    print("\n" + "="*70)
    print("FINAL TRAINING SUMMARY (Celeb-DF-v2)")
    print("="*70)
    print(f"  Best Kernel:   {grid.best_params_['svc__kernel']}")
    print(f"  Best C:        {grid.best_params_['svc__C']}")
    print("-" * 30)
    print(f"  Val AUC:       {val_metrics['auc']:.4f}")
    print(f"  Test AUC:      {test_metrics['auc']:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
