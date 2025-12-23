"""
运动特征检测模型评测脚本

目标:
- 使用 MediaPipe FaceMesh 提取与训练脚本一致的 59 维运动特征（EAR + Jitter）
- 分别加载两个 joblib 模型（Celeb-DF-v2 训练、FaceForensics++ 训练）
- 在 Celeb-DF-v1（testing list 指定的视频）与 UADFV（real/fake 全量）上测试并输出指标

数据集（本仓库 datasets/rgb 预处理格式）:
- Celeb-DF-v1:
    datasets/rgb/Celeb-DF-v1/
      - Celeb-real/frames/<video_id>/*.png
      - YouTube-real/frames/<video_id>/*.png
      - Celeb-synthesis/frames/<video_id>/*.png
      - List_of_testing_videos.txt   # 每行: "<label> <rel_path_to_mp4>"
        其中 label=1 表示 Real, label=0 表示 Fake
- UADFV:
    datasets/rgb/UADFV/
      - real/frames/<video_id>/*.png
      - fake/frames/<video_id>_fake/*.png

用法示例:
  python tools/eval_motion_models.py ^
    --model-celebdf E:\\DeepfakeBench\\motion_feature_model_celebdf.joblib ^
    --model-ffpp E:\\DeepfakeBench\\motion_feature_model.joblib ^
    --celebdfv1-root datasets\\rgb\\Celeb-DF-v1 ^
    --uadfv-root datasets\\rgb\\UADFV
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import joblib
import numpy as np


# =========================
# MediaPipe setup (lazy import)
# =========================

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

JITTER_LANDMARKS: Dict[str, int] = {
    "nose_tip": 1,
    "left_eye_outer": 263,
    "right_eye_outer": 33,
    "left_mouth": 61,
    "right_mouth": 291,
    "chin": 152,
    "left_eyebrow": 70,
    "right_eyebrow": 300,
}


def _skew_bias(x: np.ndarray) -> float:
    """
    近似 scipy.stats.skew(x, bias=True) 的定义:
      skew = E[(X-mu)^3] / sigma^3
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    mu = float(x.mean())
    v = x - mu
    m2 = float(np.mean(v * v))
    if m2 <= 1e-12:
        return 0.0
    m3 = float(np.mean(v * v * v))
    return float(m3 / (m2 ** 1.5))


def _kurtosis_fisher_bias(x: np.ndarray) -> float:
    """
    近似 scipy.stats.kurtosis(x, fisher=True, bias=True) 的定义:
      kurt = E[(X-mu)^4] / sigma^4 - 3
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    mu = float(x.mean())
    v = x - mu
    m2 = float(np.mean(v * v))
    if m2 <= 1e-12:
        return 0.0
    m4 = float(np.mean((v * v) * (v * v)))
    return float(m4 / (m2 * m2) - 3.0)


def compute_ear(eye_landmarks: np.ndarray) -> float:
    """
    Eye Aspect Ratio，要求 eye_landmarks 形状为 (6,2):
      p0..p5，其中 (p1,p5) 与 (p2,p4) 为两条竖直距离，(p0,p3) 为水平距离
    """
    eye_landmarks = np.asarray(eye_landmarks, dtype=np.float32)
    if eye_landmarks.shape != (6, 2):
        return 0.0
    v1 = float(np.linalg.norm(eye_landmarks[1] - eye_landmarks[5]))
    v2 = float(np.linalg.norm(eye_landmarks[2] - eye_landmarks[4]))
    h = float(np.linalg.norm(eye_landmarks[0] - eye_landmarks[3]))
    if h < 1e-6:
        return 0.0
    return float((v1 + v2) / (2.0 * h))


def _extract_eye_landmarks(face_landmarks, indices: Sequence[int], img_w: int, img_h: int) -> np.ndarray:
    pts = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        pts.append([int(lm.x * img_w), int(lm.y * img_h)])
    return np.asarray(pts, dtype=np.float32)


def _extract_landmarks_from_frame(frame_bgr: np.ndarray, face_mesh) -> Optional[Dict]:
    h, w, _ = frame_bgr.shape
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    face_landmarks = results.multi_face_landmarks[0]

    left_eye = _extract_eye_landmarks(face_landmarks, LEFT_EYE_INDICES, w, h)
    right_eye = _extract_eye_landmarks(face_landmarks, RIGHT_EYE_INDICES, w, h)

    jitter_points: Dict[str, np.ndarray] = {}
    for name, idx in JITTER_LANDMARKS.items():
        lm = face_landmarks.landmark[idx]
        jitter_points[name] = np.array([lm.x * w, lm.y * h], dtype=np.float32)

    return {"left_eye": left_eye, "right_eye": right_eye, "jitter_points": jitter_points}


@dataclass(frozen=True)
class FeatureConfig:
    max_frames: int = 60
    min_frames: int = 10
    ear_threshold: float = 0.21
    blink_consec_frames: int = 2


def extract_video_features_from_frames(
    frame_paths: Sequence[str],
    face_mesh,
    cfg: FeatureConfig,
) -> Optional[np.ndarray]:
    """
    与 tools/运动特征训练.py、tools/运动特征训练_CelebDF.py 保持一致的 59 维特征:
      - EAR: 11 维
      - Jitter: 8 个点 * 6 统计量 = 48 维
    """
    if len(frame_paths) < cfg.min_frames:
        return None

    # Sample frames if too many
    if len(frame_paths) > cfg.max_frames:
        idx = np.linspace(0, len(frame_paths) - 1, cfg.max_frames, dtype=int)
        frame_paths = [frame_paths[i] for i in idx]

    ear_values: List[float] = []
    jitter_sequences: Dict[str, List[np.ndarray]] = {k: [] for k in JITTER_LANDMARKS.keys()}

    for p in frame_paths:
        try:
            frame = cv2.imread(p)
            if frame is None:
                continue
            lm = _extract_landmarks_from_frame(frame, face_mesh)
            if lm is None:
                continue

            left_ear = compute_ear(lm["left_eye"])
            right_ear = compute_ear(lm["right_eye"])
            ear_values.append((left_ear + right_ear) / 2.0)

            for name, point in lm["jitter_points"].items():
                jitter_sequences[name].append(point)
        except Exception:
            continue

    if len(ear_values) < cfg.min_frames:
        return None

    ear_array = np.asarray(ear_values, dtype=np.float64)
    ear_mean = float(np.mean(ear_array))
    ear_std = float(np.std(ear_array))
    ear_min = float(np.min(ear_array))
    ear_max = float(np.max(ear_array))
    ear_range = ear_max - ear_min

    # Blink detection
    blink_count = 0
    below = 0
    for ear in ear_array:
        if ear < cfg.ear_threshold:
            below += 1
        else:
            if below >= cfg.blink_consec_frames:
                blink_count += 1
            below = 0

    # 重要: 这里保持训练脚本的写法（假设 60 帧 ~= 1 秒）
    blink_rate = float(blink_count / (len(ear_array) / 60.0)) if len(ear_array) > 0 else 0.0
    ear_skewness = _skew_bias(ear_array) if len(ear_array) > 2 else 0.0
    ear_kurtosis = _kurtosis_fisher_bias(ear_array) if len(ear_array) > 3 else 0.0

    ear_diff = np.diff(ear_array)
    ear_velocity_mean = float(np.mean(np.abs(ear_diff))) if ear_diff.size > 0 else 0.0
    ear_velocity_std = float(np.std(ear_diff)) if ear_diff.size > 0 else 0.0

    ear_features = [
        ear_mean,
        ear_std,
        ear_min,
        ear_max,
        ear_range,
        float(blink_count),
        blink_rate,
        ear_skewness,
        ear_kurtosis,
        ear_velocity_mean,
        ear_velocity_std,
    ]

    # Jitter features
    jitter_features: List[float] = []
    for name in JITTER_LANDMARKS.keys():
        seq = jitter_sequences.get(name, [])
        if seq is None or len(seq) < 3:
            jitter_features.extend([0.0] * 6)
            continue

        points = np.asarray(seq, dtype=np.float64)

        # Global motion compensation using nose tip
        nose_seq = jitter_sequences.get("nose_tip", [])
        if nose_seq is not None and len(nose_seq) == len(seq):
            points = points - np.asarray(nose_seq, dtype=np.float64)

        displacements = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if displacements.size > 0:
            disp_mean = float(np.mean(displacements))
            disp_std = float(np.std(displacements))
            disp_max = float(np.max(displacements))
            if displacements.size > 1:
                accel = np.abs(np.diff(displacements))
                accel_mean = float(np.mean(accel))
                accel_std = float(np.std(accel))
                accel_max = float(np.max(accel)) if accel.size > 0 else 0.0
            else:
                accel_mean = accel_std = accel_max = 0.0
        else:
            disp_mean = disp_std = disp_max = 0.0
            accel_mean = accel_std = accel_max = 0.0

        jitter_features.extend([disp_mean, disp_std, disp_max, accel_mean, accel_std, accel_max])

    feats = np.asarray(ear_features + jitter_features, dtype=np.float32)
    if feats.shape != (59,):
        return None
    return feats


def _list_images_sorted(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files.sort()
    return [os.path.join(folder, f) for f in files]


def collect_celebdf_v1_from_testing_list(dataset_root: str, list_path: str) -> List[Tuple[str, List[str], int]]:
    """
    使用 Celeb-DF-v1 的 List_of_testing_videos.txt 生成测试列表。
    该 list 使用 mp4 相对路径，但本仓库为 extracted frames 格式，因此将 mp4 映射到 frames 子目录。
    """
    if not os.path.isfile(list_path):
        raise FileNotFoundError(f"找不到 testing list: {list_path}")

    out: List[Tuple[str, List[str], int]] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            label_raw = parts[0]
            rel_mp4 = parts[1].replace("\\", "/")
            # list 标注: 1=Real, 0=Fake；而我们的模型约定: 0=Real, 1=Fake
            y = 0 if label_raw == "1" else 1

            rel_dir = rel_mp4
            if rel_dir.lower().endswith(".mp4"):
                rel_dir = rel_dir[:-4]
            rel_parent = os.path.dirname(rel_dir)
            base = os.path.basename(rel_dir)
            frames_dir = os.path.join(dataset_root, rel_parent, "frames", base)

            frame_paths = _list_images_sorted(frames_dir)
            video_id = rel_dir
            out.append((video_id, frame_paths, y))
    return out


def collect_uadfv(dataset_root: str) -> List[Tuple[str, List[str], int]]:
    out: List[Tuple[str, List[str], int]] = []

    real_root = os.path.join(dataset_root, "real", "frames")
    fake_root = os.path.join(dataset_root, "fake", "frames")

    if os.path.isdir(real_root):
        for vid in sorted(os.listdir(real_root)):
            vdir = os.path.join(real_root, vid)
            if not os.path.isdir(vdir):
                continue
            frames = _list_images_sorted(vdir)
            out.append((f"real/{vid}", frames, 0))

    if os.path.isdir(fake_root):
        for vid in sorted(os.listdir(fake_root)):
            vdir = os.path.join(fake_root, vid)
            if not os.path.isdir(vdir):
                continue
            frames = _list_images_sorted(vdir)
            out.append((f"fake/{vid}", frames, 1))

    return out


def maybe_load_cache(cache_path: Optional[str]) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    if not cache_path:
        return None
    if not os.path.isfile(cache_path):
        return None
    data = np.load(cache_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    ids = list(data["ids"].tolist())
    return X, y, ids


def save_cache(cache_path: Optional[str], X: np.ndarray, y: np.ndarray, ids: List[str]) -> None:
    if not cache_path:
        return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, ids=np.asarray(ids, dtype=object))


def extract_dataset_features(
    video_list: Sequence[Tuple[str, List[str], int]],
    feat_cfg: FeatureConfig,
    cache_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, int]]:
    cached = maybe_load_cache(cache_path)
    if cached is not None:
        X, y, ids = cached
        stats = {"total": int(len(ids)), "ok": int(X.shape[0]), "skipped": int(len(ids) - X.shape[0])}
        return X, y, ids, stats

    try:
        import mediapipe as mp
    except Exception as e:
        raise RuntimeError(f"无法导入 mediapipe: {e}")

    mp_face_mesh = mp.solutions.face_mesh
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    ids: List[str] = []

    total = 0
    ok = 0
    skipped = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for vid, frames, label in video_list:
            total += 1
            feats = extract_video_features_from_frames(frames, face_mesh, feat_cfg)
            ids.append(vid)
            if feats is None:
                skipped += 1
                continue
            X_list.append(feats)
            y_list.append(int(label))
            ok += 1

    if ok == 0:
        raise RuntimeError("特征提取失败：没有任何视频成功产生特征。")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    save_cache(cache_path, X, y, ids)
    return X, y, ids, {"total": total, "ok": ok, "skipped": skipped}


def _model_scores_and_preds(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回:
      - score: 用于 AUC 的连续分数（优先 decision_function，其次 proba[:,1]）
      - pred: 预测标签（model.predict）
    """
    pred = model.predict(X)
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        score = np.asarray(score).reshape(-1)
        return score, np.asarray(pred).reshape(-1)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        score = np.asarray(proba)[:, 1]
        return score, np.asarray(pred).reshape(-1)
    # fallback: 用预测标签做“分数”
    return np.asarray(pred).reshape(-1), np.asarray(pred).reshape(-1)


def _model_score_fake(model, X: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    返回每个样本的“fake 方向”的连续分数（可排序）。
    约定：分数越大越偏 fake(=1)。
    - 优先使用 decision_function，并根据 classes_ 统一方向。
    - 若没有 decision_function：退化为 logit(P(fake))（仅用于可排序，不当作概率）。
    - 再不行：退化为硬分类(0/1)。
    """
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(X)).reshape(-1).astype(np.float64, copy=False)
        classes = getattr(model, "classes_", None)
        if classes is not None:
            try:
                classes = list(classes)
                # sklearn 二分类 decision_function 的符号默认是相对 classes_[1]
                if len(classes) >= 2 and classes[1] != 1:
                    score = -score
            except Exception:
                pass
        return score, "decision_function"

    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X)).astype(np.float64, copy=False)
        classes = getattr(model, "classes_", None)
        if classes is not None:
            try:
                classes = list(classes)
                idx_fake = classes.index(1)
                p = proba[:, idx_fake]
            except Exception:
                p = proba[:, 1]
        else:
            p = proba[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)), "logit(predict_proba)"

    pred = np.asarray(model.predict(X)).reshape(-1).astype(np.float64, copy=False)
    return pred, "hard_pred"


def _describe_scores(s: np.ndarray) -> Dict[str, float]:
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "ge0": float("nan"),
            "ge05": float("nan"),
            "ge1": float("nan"),
        }
    q50, q90, q95, q99 = np.quantile(s, [0.5, 0.9, 0.95, 0.99])
    return {
        "n": int(s.size),
        "mean": float(np.mean(s)),
        "std": float(np.std(s)),
        "min": float(np.min(s)),
        "p50": float(q50),
        "p90": float(q90),
        "p95": float(q95),
        "p99": float(q99),
        "max": float(np.max(s)),
        "ge0": float(np.mean(s >= 0.0)),
        "ge05": float(np.mean(s >= 0.5)),
        "ge1": float(np.mean(s >= 1.0)),
    }


def _print_score_report(title: str, score_fake: np.ndarray) -> None:
    d = _describe_scores(score_fake)
    if d["n"] == 0:
        print(f"  {title}: n=0")
        return
    print(
        "  {}: n={} mean±std={:.3f}±{:.3f} min={:.3f} p50={:.3f} p90={:.3f} p95={:.3f} p99={:.3f} max={:.3f} | "
        "P(score>=0)={:.1f}% P>=0.5={:.1f}% P>=1.0={:.1f}%".format(
            title,
            d["n"],
            d["mean"],
            d["std"],
            d["min"],
            d["p50"],
            d["p90"],
            d["p95"],
            d["p99"],
            d["max"],
            100.0 * d["ge0"],
            100.0 * d["ge05"],
            100.0 * d["ge1"],
        )
    )


def _confusion_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    return tn, fp, fn, tp


def _metrics_from_pred(y_true: np.ndarray, y_pred: np.ndarray, score: Optional[np.ndarray] = None) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    if score is not None:
        try:
            auc = float(roc_auc_score(y_true, score))
        except Exception:
            auc = float("nan")
    else:
        auc = float("nan")

    tn, fp, fn, tp = _confusion_from_pred(y_true, y_pred)
    return {"acc": acc, "f1": f1, "auc": auc, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _search_best_threshold(
    y_true: np.ndarray,
    score: np.ndarray,
    objective: str = "acc",
) -> Tuple[float, Dict[str, float]]:
    """
    在 score 上搜索阈值 t，使得 pred = (score >= t) 最优化指定指标。
    - objective: 'acc' | 'f1' | 'youden'  (youden = TPR - FPR)
    """
    y_true = np.asarray(y_true).reshape(-1)
    score = np.asarray(score).reshape(-1)
    if y_true.size != score.size:
        raise ValueError("y_true 与 score 长度不一致")

    # 候选阈值：用分位点近似，避免遍历全部 unique（速度更稳）
    qs = np.linspace(0.0, 1.0, 401)
    thresholds = np.quantile(score, qs)

    best_t = float(thresholds[0])
    best_val = -1e18
    best_metrics: Dict[str, float] = {}

    for t in thresholds:
        pred = (score >= t).astype(np.int64)
        m = _metrics_from_pred(y_true, pred, score=score)

        if objective == "acc":
            val = m["acc"]
        elif objective == "f1":
            val = m["f1"]
        elif objective == "youden":
            tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            val = tpr - fpr
        else:
            raise ValueError(f"未知 objective: {objective}")

        if val > best_val:
            best_val = float(val)
            best_t = float(t)
            best_metrics = m

    return best_t, best_metrics


def evaluate_on_dataset(
    name: str,
    model_path: str,
    X: np.ndarray,
    y: np.ndarray,
    show_prob_dist: bool = True,
) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score

    # 兼容：模型保存时 sklearn=1.2.2，而当前环境可能更老，会不断刷 warning
    warnings.filterwarnings("once", category=UserWarning, module=r"sklearn\.base")

    model = joblib.load(model_path)
    score, pred = _model_scores_and_preds(model, X)

    # 默认阈值（即 model.predict 的结果）
    base = _metrics_from_pred(y, pred, score=score)

    # AUC 用连续 score 统一计算（更可比）
    try:
        base["auc"] = float(roc_auc_score(y, score))
    except Exception:
        base["auc"] = float("nan")

    # 如果 score 是连续的（decision_function 或 proba），额外搜索“最优阈值”帮助定位阈值偏移问题
    tuned_acc = None
    tuned_f1 = None
    tuned_youden = None

    is_continuous = True
    if np.unique(score).size <= 2:
        is_continuous = False

    if is_continuous:
        t_acc, m_acc = _search_best_threshold(y, score, objective="acc")
        t_f1, m_f1 = _search_best_threshold(y, score, objective="f1")
        t_yj, m_yj = _search_best_threshold(y, score, objective="youden")
        tuned_acc = (t_acc, m_acc)
        tuned_f1 = (t_f1, m_f1)
        tuned_youden = (t_yj, m_yj)

    # 打印
    print(f"\n========== {name} ==========")
    print(f"Model: {model_path}")
    print(f"Samples: {len(y)} (real=0: {(y==0).sum()} | fake=1: {(y==1).sum()})")
    print(f"[Default]  Accuracy: {base['acc']:.4f} | F1(fake=1): {base['f1']:.4f} | AUC: {base['auc']:.4f}")
    print("Confusion Matrix (rows=true [real=0,fake=1], cols=pred [0,1]):")
    print(f"  TN={base['tn']}  FP={base['fp']}")
    print(f"  FN={base['fn']}  TP={base['tp']}")

    # 附加：score 分布（帮助判断整体偏移）
    s0 = score[y == 0]
    s1 = score[y == 1]
    if s0.size > 0 and s1.size > 0 and is_continuous:
        print(
            "[Score] real(mean±std)={:.3f}±{:.3f} | fake(mean±std)={:.3f}±{:.3f}".format(
                float(np.mean(s0)),
                float(np.std(s0)),
                float(np.mean(s1)),
                float(np.std(s1)),
            )
        )

    if tuned_acc is not None and tuned_f1 is not None and tuned_youden is not None:
        t_acc, m_acc = tuned_acc
        t_f1, m_f1 = tuned_f1
        t_yj, m_yj = tuned_youden

        print(
            f"[Tuned@ACC]  t={t_acc:.4f} | acc={m_acc['acc']:.4f} f1={m_acc['f1']:.4f}  "
            f"(TN={m_acc['tn']} FP={m_acc['fp']} FN={m_acc['fn']} TP={m_acc['tp']})"
        )
        print(
            f"[Tuned@F1 ]  t={t_f1:.4f} | acc={m_f1['acc']:.4f} f1={m_f1['f1']:.4f}  "
            f"(TN={m_f1['tn']} FP={m_f1['fp']} FN={m_f1['fn']} TP={m_f1['tp']})"
        )
        print(
            f"[Tuned@YJ ]  t={t_yj:.4f} | acc={m_yj['acc']:.4f} f1={m_yj['f1']:.4f}  "
            f"(TN={m_yj['tn']} FP={m_yj['fp']} FN={m_yj['fn']} TP={m_yj['tp']})"
        )

    if show_prob_dist:
        score_fake, score_src = _model_score_fake(model, X)
        y_arr = np.asarray(y).reshape(-1)
        pred_arr = np.asarray(pred).reshape(-1)
        print("\n[Score(fake=1 direction)] source:", score_src)
        _print_score_report("All", score_fake)
        _print_score_report("True real (y=0)", score_fake[y_arr == 0])
        _print_score_report("True fake (y=1)", score_fake[y_arr == 1])
        _print_score_report("FP (y=0,pred=1)", score_fake[(y_arr == 0) & (pred_arr == 1)])
        _print_score_report("TP (y=1,pred=1)", score_fake[(y_arr == 1) & (pred_arr == 1)])
        _print_score_report("FN (y=1,pred=0)", score_fake[(y_arr == 1) & (pred_arr == 0)])
        _print_score_report("TN (y=0,pred=0)", score_fake[(y_arr == 0) & (pred_arr == 0)])

    return base


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate motion-feature joblib models on Celeb-DF-v1 and UADFV.")
    ap.add_argument("--model-celebdf", required=True, help="Celeb-DF-v2 训练得到的 joblib 模型路径")
    ap.add_argument("--model-ffpp", required=True, help="FaceForensics++ 训练得到的 joblib 模型路径")
    ap.add_argument("--celebdfv1-root", default=os.path.join("datasets", "rgb", "Celeb-DF-v1"))
    ap.add_argument("--celebdfv1-testing-list", default=os.path.join("datasets", "rgb", "Celeb-DF-v1", "List_of_testing_videos.txt"))
    ap.add_argument("--uadfv-root", default=os.path.join("datasets", "rgb", "UADFV"))
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--min-frames", type=int, default=10)
    ap.add_argument("--ear-threshold", type=float, default=0.21)
    ap.add_argument("--blink-consec", type=int, default=2)
    ap.add_argument("--cache-dir", default=None, help="可选：保存提取好的特征 npz 以加速重复评测")
    ap.add_argument("--no-prob-dist", action="store_true", help="不打印 decision_function 分数分布统计")
    args = ap.parse_args()

    show_prob_dist = not bool(args.no_prob_dist)

    feat_cfg = FeatureConfig(
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        ear_threshold=args.ear_threshold,
        blink_consec_frames=args.blink_consec,
    )

    # Celeb-DF-v1 (testing list)
    celeb_list = collect_celebdf_v1_from_testing_list(args.celebdfv1_root, args.celebdfv1_testing_list)
    celeb_cache = None
    if args.cache_dir:
        celeb_cache = os.path.join(args.cache_dir, "celebdfv1_motion_feats_mp.npz")
    X_celeb, y_celeb, _, celeb_stats = extract_dataset_features(celeb_list, feat_cfg, cache_path=celeb_cache)
    print(f"[Celeb-DF-v1] total={celeb_stats['total']} ok={celeb_stats['ok']} skipped={celeb_stats['skipped']} (使用 MediaPipe)")

    # UADFV (full)
    uadfv_list = collect_uadfv(args.uadfv_root)
    uadfv_cache = None
    if args.cache_dir:
        uadfv_cache = os.path.join(args.cache_dir, "uadfv_motion_feats_mp.npz")
    X_uadfv, y_uadfv, _, uadfv_stats = extract_dataset_features(uadfv_list, feat_cfg, cache_path=uadfv_cache)
    print(f"[UADFV] total={uadfv_stats['total']} ok={uadfv_stats['ok']} skipped={uadfv_stats['skipped']} (使用 MediaPipe)")

    # Evaluate both models on both datasets
    evaluate_on_dataset("Celeb-DF-v1 / Celeb-DF-v2模型", args.model_celebdf, X_celeb, y_celeb, show_prob_dist=show_prob_dist)
    evaluate_on_dataset("Celeb-DF-v1 / FF++模型", args.model_ffpp, X_celeb, y_celeb, show_prob_dist=show_prob_dist)
    evaluate_on_dataset("UADFV / Celeb-DF-v2模型", args.model_celebdf, X_uadfv, y_uadfv, show_prob_dist=show_prob_dist)
    evaluate_on_dataset("UADFV / FF++模型", args.model_ffpp, X_uadfv, y_uadfv, show_prob_dist=show_prob_dist)


if __name__ == "__main__":
    main()
