"""
Batch Testing Script for DeepfakeBench
功能：批量测试所有模型在 Celeb-DF-v1 数据集上的表现
特性：
  - 自动跳过出错的模型
  - 收集并汇总所有结果
  - 保存结果到 CSV 文件
"""

import os
import sys
import yaml
import torch
import random
import traceback
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Windows multiprocessing fix - 必须在其他导入之前设置
import multiprocessing
if sys.platform == 'win32':
    multiprocessing.set_start_method('spawn', force=True)
import pandas as pd
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

# 尝试导入 seaborn，如果失败则使用纯 matplotlib
try:
    import seaborn as sns
    USE_SEABORN = True
    print("✓ Seaborn loaded successfully")
except (ImportError, AttributeError) as e:
    USE_SEABORN = False
    print(f"⚠ Seaborn not available ({e}), using matplotlib only")

# 设置matplotlib支持中文显示
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
except:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置样式
try:
    if USE_SEABORN:
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    else:
        plt.style.use('ggplot')
except:
    # 如果没有 seaborn 样式，使用经典样式
    try:
        plt.style.use('ggplot')
    except:
        pass  # 使用默认样式

# 添加 training 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
from metrics.utils import get_test_metrics

# ==================== 配置 ====================
# 测试数据集 - 修改这里来测试不同的数据集
TEST_DATASET = "Celeb-DF-v1"  # 可选: "Celeb-DF-v1", "UADFV", "CelebDFv2", "FF-DF", etc

# 模型权重和配置映射
# 已修复：启用所有模型，添加了OOM保护
MODELS = {
    # === Naive Models ===
    "xception": {
        "config": "./training/config/detector/xception.yaml",
        "weights": "./training/weights/xception_best.pth",
    },
    "efficientnetb4": {
        "config": "./training/config/detector/efficientnetb4.yaml",
        "weights": "./training/weights/effnb4_best.pth",
    },
    "meso4": {
        "config": "./training/config/detector/meso4.yaml",
        "weights": "./training/weights/meso4_best.pth",
    },
    "meso4Inception": {
        "config": "./training/config/detector/meso4Inception.yaml",
        "weights": "./training/weights/meso4Incep_best.pth",
    },
    # === Spatial Models ===
    "capsule_net": {
        "config": "./training/config/detector/capsule_net.yaml",
        "weights": "./training/weights/capsule_best.pth",
    },
    "ffd": {
        "config": "./training/config/detector/ffd.yaml",
        "weights": "./training/weights/ffd_best.pth",
    },
    "core": {
        "config": "./training/config/detector/core.yaml",
        "weights": "./training/weights/core_best.pth",
    },
    "recce": {
        "config": "./training/config/detector/recce.yaml",
        "weights": "./training/weights/recce_best.pth",
    },
    "ucf": {
        "config": "./training/config/detector/ucf.yaml",
        "weights": "./training/weights/ucf_best.pth",
    },
    # === Frequency Models ===
    "f3net": {
        "config": "./training/config/detector/f3net.yaml",
        "weights": "./training/weights/f3net_best.pth",
    },
    "spsl": {
        "config": "./training/config/detector/spsl.yaml",
        "weights": "./training/weights/spsl_best.pth",
    },
    "srm": {
        "config": "./training/config/detector/srm.yaml",
        "weights": "./training/weights/srm_best.pth",
    },
}

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 辅助函数 ====================

def init_seed(config):
    """初始化随机种子"""
    if config.get('manualSeed') is None:
        config['manualSeed'] = 1024
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config.get('cuda', True):
        torch.cuda.manual_seed_all(config['manualSeed'])


def load_config(detector_path, test_dataset):
    """加载并合并配置"""
    with open(detector_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载测试配置
    test_config_path = './training/config/test_config.yaml'
    with open(test_config_path, 'r', encoding='utf-8') as f:
        test_config = yaml.safe_load(f)
    
    # 合并配置
    config.update(test_config)
    config['test_dataset'] = [test_dataset]
    
    # 强制设置 workers 为 0，避免 Windows 多进程内存问题
    config['workers'] = 0
    
    # 大幅减小 batch size 以节省显存（改为 2 以支持大模型如 SRM, RECCE）
    config['test_batchSize'] = 2
    print(f"  Set batch size to 2 to save GPU memory", flush=True)
    
    # 减小 frame_num 以节省内存
    if 'frame_num' in config:
        config['frame_num'] = {'train': 8, 'test': 8}
    
    return config


def prepare_testing_data(config):
    """准备测试数据加载器"""
    def get_test_data_loader(config, test_name):
        config = config.copy()
        config['test_dataset'] = test_name
        test_set = DeepfakeAbstractBaseDataset(
            config=config,
            mode='test',
        )
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config.get('test_batchSize', 32),
            shuffle=False,
            num_workers=int(config.get('workers', 0)),
            collate_fn=test_set.collate_fn,
            drop_last=False
        )
        return test_data_loader, test_set

    test_data_loaders = {}
    test_datasets = {}
    for one_test_name in config['test_dataset']:
        loader, dataset = get_test_data_loader(config, one_test_name)
        test_data_loaders[one_test_name] = loader
        test_datasets[one_test_name] = dataset
    return test_data_loaders, test_datasets


@torch.no_grad()
def inference(model, data_dict):
    """模型推理"""
    predictions = model(data_dict, inference=True)
    return predictions


def test_one_dataset(model, data_loader):
    """测试单个数据集"""
    prediction_lists = []
    label_lists = []
    
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader), desc="Testing"):
        # 获取数据
        data, label, mask, landmark = \
            data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        
        # 移动数据到 GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # 模型推理
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
    
    return np.array(prediction_lists), np.array(label_lists)


# ==================== 可视化函数 ====================

def create_visualizations(df, test_dataset, timestamp):
    """创建所有可视化图表"""
    output_dir = f"./results_{TEST_DATASET}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 Generating visualizations...")
    
    # 1. AUC 对比柱状图
    create_auc_bar_chart(df, output_dir)
    
    # 2. 多指标对比图
    create_multi_metric_comparison(df, output_dir)
    
    # 3. 排名可视化
    create_ranking_visualization(df, output_dir)
    
    # 4. 综合性能雷达图（前5名模型）
    create_radar_chart(df, output_dir)
    
    # 5. 详细对比热力图
    create_heatmap(df, output_dir)
    
    print(f"✓ All visualizations saved to: {output_dir}/")
    return output_dir


def create_auc_bar_chart(df, output_dir):
    """创建AUC对比柱状图"""
    if 'auc' not in df.columns or df['auc'].isna().all():
        return
    
    # 按AUC排序
    df_sorted = df.sort_values('auc', ascending=True)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_sorted)))
    
    bars = plt.barh(df_sorted['model'], df_sorted['auc'], color=colors, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, df_sorted['auc'])):
        if not np.isnan(val):
            plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{val:.4f}', va='center', fontsize=10, weight='bold')
    
    plt.xlabel('AUC Score', fontsize=14, weight='bold')
    plt.ylabel('Model', fontsize=14, weight='bold')
    plt.title(f'Model Performance Comparison - AUC on {TEST_DATASET}', 
              fontsize=16, weight='bold', pad=20)
    plt.xlim(0, 1.0)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '01_auc_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def create_multi_metric_comparison(df, output_dir):
    """创建多指标对比图"""
    metrics = ['auc', 'acc', 'eer', 'ap']
    available_metrics = [m for m in metrics if m in df.columns and not df[m].isna().all()]
    
    if not available_metrics:
        return
    
    # 按第一个可用指标排序
    df_sorted = df.sort_values(available_metrics[0], ascending=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        data = df_sorted.sort_values(metric, ascending=False)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(data)))
        bars = ax.bar(range(len(data)), data[metric], color=colors, 
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, data[metric])):
            if not np.isnan(val):
                ax.text(i, val + 0.01, f'{val:.3f}', 
                       ha='center', va='bottom', fontsize=9, weight='bold')
        
        ax.set_xlabel('Model', fontsize=12, weight='bold')
        ax.set_ylabel(metric.upper(), fontsize=12, weight='bold')
        ax.set_title(f'{metric.upper()} Comparison', fontsize=14, weight='bold')
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data['model'], rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.0)
    
    # 隐藏多余的子图
    for idx in range(len(available_metrics), 4):
        axes[idx].axis('off')
    
    plt.suptitle(f'Multi-Metric Performance Comparison on {TEST_DATASET}', 
                 fontsize=18, weight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '02_multi_metric_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def create_ranking_visualization(df, output_dir):
    """创建排名可视化"""
    if 'auc' not in df.columns or df['auc'].isna().all():
        return
    
    df_sorted = df.sort_values('auc', ascending=False).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 创建排名
    ranks = np.arange(1, len(df_sorted) + 1)
    scores = df_sorted['auc'].values
    
    # 绘制排名线
    colors = ['gold', 'silver', '#CD7F32']  # 金、银、铜
    for i in range(min(3, len(df_sorted))):
        color = colors[i] if i < 3 else 'skyblue'
        ax.scatter(ranks[i], scores[i], s=800, c=color, edgecolors='black', 
                  linewidth=3, zorder=3, alpha=0.9)
        ax.text(ranks[i], scores[i], f'#{i+1}', ha='center', va='center', 
               fontsize=14, weight='bold', color='black')
    
    # 其他模型
    for i in range(3, len(df_sorted)):
        ax.scatter(ranks[i], scores[i], s=500, c='steelblue', edgecolors='black', 
                  linewidth=2, zorder=3, alpha=0.7)
        ax.text(ranks[i], scores[i], f'#{i+1}', ha='center', va='center', 
               fontsize=11, weight='bold', color='white')
    
    # 连接线
    ax.plot(ranks, scores, 'o--', color='gray', alpha=0.5, linewidth=2, zorder=1)
    
    # 添加模型名称
    for i, (rank, score, model) in enumerate(zip(ranks, scores, df_sorted['model'])):
        offset = 0.02 if i % 2 == 0 else -0.02
        ax.text(rank, score + offset, model, ha='center', va='bottom' if i % 2 == 0 else 'top',
               fontsize=10, weight='bold', bbox=dict(boxstyle='round,pad=0.3', 
               facecolor='white', edgecolor='gray', alpha=0.8))
    
    ax.set_xlabel('Rank', fontsize=14, weight='bold')
    ax.set_ylabel('AUC Score', fontsize=14, weight='bold')
    ax.set_title(f'Model Ranking by AUC on {TEST_DATASET}', 
                fontsize=16, weight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(min(scores) - 0.05, max(scores) + 0.1)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '03_ranking_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def create_radar_chart(df, output_dir):
    """创建雷达图（前5名模型）"""
    metrics = ['auc', 'acc', 'ap']
    available_metrics = [m for m in metrics if m in df.columns and not df[m].isna().all()]
    
    if len(available_metrics) < 2:
        return
    
    # 选择前5名模型
    df_top = df.nlargest(min(5, len(df)), available_metrics[0])
    
    # 准备数据
    categories = [m.upper() for m in available_metrics]
    N = len(categories)
    
    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(df_top)))
    
    for idx, (_, row) in enumerate(df_top.iterrows()):
        values = [row[m] for m in available_metrics]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], 
               color=colors[idx], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, weight='bold')
    ax.set_ylim(0, 1)
    ax.set_title(f'Top 5 Models - Multi-Metric Radar Chart\n{TEST_DATASET}', 
                fontsize=16, weight='bold', pad=30)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '04_radar_chart_top5.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def create_heatmap(df, output_dir):
    """创建热力图"""
    metrics = ['auc', 'acc', 'eer', 'ap']
    available_metrics = [m for m in metrics if m in df.columns and not df[m].isna().all()]
    
    if not available_metrics:
        return
    
    # 准备数据
    df_heat = df[['model'] + available_metrics].set_index('model')
    
    # 按第一个指标排序
    df_heat = df_heat.sort_values(available_metrics[0], ascending=False)
    
    plt.figure(figsize=(10, len(df_heat) * 0.5 + 2))
    
    if USE_SEABORN:
        # 使用 seaborn 创建热力图
        sns.heatmap(df_heat.T, annot=True, fmt='.4f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Score'}, linewidths=1.5, linecolor='black',
                   vmin=0, vmax=1, annot_kws={'fontsize': 10, 'weight': 'bold'})
    else:
        # 使用纯 matplotlib 创建热力图
        data_array = df_heat.T.values
        im = plt.imshow(data_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # 添加颜色条
        cbar = plt.colorbar(im)
        cbar.set_label('Score', fontsize=12, weight='bold')
        
        # 设置刻度
        ax = plt.gca()
        ax.set_xticks(np.arange(len(df_heat.index)))
        ax.set_yticks(np.arange(len(df_heat.columns)))
        ax.set_xticklabels(df_heat.index, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(df_heat.columns, rotation=0, fontsize=12, weight='bold')
        
        # 添加数值标签
        for i in range(len(df_heat.columns)):
            for j in range(len(df_heat.index)):
                value = data_array[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.4f}',
                                 ha="center", va="center", color="black",
                                 fontsize=10, weight='bold')
        
        # 添加网格线
        ax.set_xticks(np.arange(len(df_heat.index))-0.5, minor=True)
        ax.set_yticks(np.arange(len(df_heat.columns))-0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
    
    plt.title(f'Performance Heatmap - All Metrics on {TEST_DATASET}', 
             fontsize=16, weight='bold', pad=20)
    plt.xlabel('Model', fontsize=14, weight='bold')
    plt.ylabel('Metric', fontsize=14, weight='bold')
    
    if not USE_SEABORN:
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=12, weight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '05_performance_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")





def aggressive_memory_cleanup():
    """强力清理GPU内存"""
    import gc
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # 打印当前GPU内存使用情况
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  [GPU Memory] Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")


def test_single_model(model_name, model_info, test_dataset):
    """测试单个模型"""
    
    # 在开始测试前强力清理GPU内存
    print("  Cleaning GPU memory before test...")
    aggressive_memory_cleanup()
    
    print(f"\n{'='*60}")
    print(f"Testing Model: {model_name}")
    print(f"{'='*60}")
    
    try:
        # 检查文件是否存在
        if not os.path.exists(model_info['config']):
            raise FileNotFoundError(f"Config file not found: {model_info['config']}")
        if not os.path.exists(model_info['weights']):
            raise FileNotFoundError(f"Weights file not found: {model_info['weights']}")
        
        # 加载配置
        print(f"Loading config from: {model_info['config']}", flush=True)
        config = load_config(model_info['config'], test_dataset)
        
        # 初始化种子
        init_seed(config)
        
        # 设置 cudnn
        if config.get('cudnn', True):
            cudnn.benchmark = True
        
        # 准备测试数据
        print(f"Preparing test data for {test_dataset}...", flush=True)
        test_data_loaders, test_datasets = prepare_testing_data(config)
        print(f"✓ Test data prepared", flush=True)
        
        # 准备模型
        print(f"Initializing model: {config['model_name']}...", flush=True)
        model_class = DETECTOR[config['model_name']]
        print(f"  Model class loaded: {model_class}", flush=True)
        print(f"  Creating model instance...", flush=True)
        model = model_class(config)
        print(f"  Moving model to device: {device}...", flush=True)
        model = model.to(device)
        print(f"✓ Model initialized on {device}", flush=True)
        
        # 加载权重
        print(f"Loading weights from: {model_info['weights']}")
        ckpt = torch.load(model_info['weights'], map_location=device)
        
        # 智能加载权重函数
        def smart_load_state_dict(model, state_dict):
            """自动处理形状不匹配的加载函数"""
            model_state = model.state_dict()
            filtered_state = {}
            skipped_keys = []
            
            for k, v in state_dict.items():
                if k in model_state:
                    if v.shape == model_state[k].shape:
                        filtered_state[k] = v
                    else:
                        skipped_keys.append(f"{k} (ckpt: {v.shape} vs model: {model_state[k].shape})")
                else:
                    # 包含 unexpected keys，反正 strict=False 会忽略它们，或者如果幸运的话匹配上
                    filtered_state[k] = v
            
            if skipped_keys:
                print(f"  ⚠ Skipped {len(skipped_keys)} layers due to shape mismatch:")
                for sk in skipped_keys[:3]:
                    print(f"    - {sk}")
                if len(skipped_keys) > 3: print(f"    - ... and {len(skipped_keys)-3} more")
            
            return model.load_state_dict(filtered_state, strict=False)

        # 尝试加载权重
        try:
            model.load_state_dict(ckpt, strict=True)
            print(f"✓ Loaded weights (strict mode)")
        except RuntimeError as e:
            print(f"⚠ Strict loading failed, trying smart mode (ignoring shape mismatches)...")
            # 使用智能加载
            msg = smart_load_state_dict(model, ckpt)
            print(f"✓ Loaded weights (smart non-strict mode)")
            if msg.missing_keys:
                print(f"  Missing: {len(msg.missing_keys)} keys")
            if msg.unexpected_keys:
                print(f"  Unexpected: {len(msg.unexpected_keys)} keys")
        
        # 设置为评估模式
        model.eval()
        
        # 测试
        results = {}
        for key in test_data_loaders.keys():
            print(f"\nTesting on {key}...")
            data_dict = test_data_loaders[key].dataset.data_dict
            predictions_nps, label_nps = test_one_dataset(model, test_data_loaders[key])
            
            # 计算指标
            metrics = get_test_metrics(
                y_pred=predictions_nps,
                y_true=label_nps,
                img_names=data_dict['image']
            )
            results[key] = metrics
            
            print(f"\n--- Results for {key} ---")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        
        # 显式清理模型和数据 - 更彻底的清理
        print("\n  Cleaning up model and data...")
        try:
            model.cpu()  # 先移动到CPU
        except:
            pass
        del model
        del ckpt
        del test_data_loaders
        del test_datasets
        
        # 强力清理GPU内存
        aggressive_memory_cleanup()
        
        return results
        
    except Exception as e:
        # 如果在函数内部发生任何异常，重新抛出以便外层捕获
        print(f"\n❌ Error in test_single_model: {str(e)}")
        traceback.print_exc()
        
        # 强力清理可能的残留资源
        print("  Performing aggressive memory cleanup after error...")
        aggressive_memory_cleanup()
        
        raise


def main():
    """主函数"""
    print("=" * 70)
    print("DeepfakeBench Batch Testing Script")
    print(f"Test Dataset: {TEST_DATASET}")
    print(f"Device: {device}")
    print(f"Number of Models: {len(MODELS)}")
    print("=" * 70)
    
    # 结果收集
    all_results = []
    successful_models = []
    failed_models = []
    
    # 逐个测试模型
    total_models = len(MODELS)
    for idx, (model_name, model_info) in enumerate(MODELS.items(), 1):
        print(f"\n{'#'*70}")
        print(f"# Progress: {idx}/{total_models} - Testing {model_name}")
        print(f"{'#'*70}")
        
        try:
            results = test_single_model(model_name, model_info, TEST_DATASET)
            
            # 提取指标
            for dataset_name, metrics in results.items():
                result_entry = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "auc": metrics.get('auc', None),
                    "acc": metrics.get('acc', None),
                    "eer": metrics.get('eer', None),
                    "ap": metrics.get('ap', None),
                }
                all_results.append(result_entry)
            
            successful_models.append(model_name)
            print(f"\n✓ {model_name}: Test completed successfully!")
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️ User interrupted the testing process!")
            print(f"Processed {idx-1}/{total_models} models before interruption.")
            break
            
        except Exception as e:
            print(f"\n✗ {model_name}: FAILED!")
            print(f"  Error Type: {type(e).__name__}")
            print(f"  Error Message: {str(e)[:300]}")
            
            # 只打印最后几行的traceback，避免过长输出
            import sys
            import io
            f = io.StringIO()
            traceback.print_exc(limit=5, file=f)
            error_trace = f.getvalue()
            print(f"\n  Stack Trace (last 5 frames):")
            for line in error_trace.split('\n')[-15:]:
                if line.strip():
                    print(f"    {line}")
            
            failed_models.append({
                "model": model_name,
                "error": f"{type(e).__name__}: {str(e)[:200]}"
            })
            print(f"\n  ⏭️  Skipping {model_name} and continuing with next model...")
        
        # 清理 GPU 内存和 Python 垃圾回收
        import gc
        gc.collect()  # Python 垃圾回收
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"  🧹 GPU cache cleared", flush=True)
    
    
    # ==================== 汇总结果 ====================
    print("\n")
    print("=" * 70)
    print("BATCH TESTING SUMMARY")
    print("=" * 70)
    
    print(f"\n✓ Successful Models ({len(successful_models)}/{len(MODELS)}):")
    for m in successful_models:
        print(f"  - {m}")
    
    if failed_models:
        print(f"\n✗ Failed Models ({len(failed_models)}/{len(MODELS)}):")
        for m in failed_models:
            print(f"  - {m['model']}: {m['error'][:80]}...")
    
    # 保存结果到 CSV
    if all_results:
        df = pd.DataFrame(all_results)
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"./batch_test_results_{TEST_DATASET}_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n📊 Results saved to: {output_file}")
        
        # ==================== 打印排名表格 ====================
        print("\n" + "=" * 70)
        print("MODEL RANKING (by AUC)")
        print("=" * 70)
        
        if 'auc' in df.columns:
            df_sorted = df.sort_values('auc', ascending=False).reset_index(drop=True)
            
            # 打印表头
            print(f"\n{'Rank':<6} {'Medal':<8} {'Model':<20} {'AUC':<10} {'ACC':<10} {'EER':<10} {'AP':<10}")
            print("-" * 70)
            
            # 打印排名
            medals = ['🥇', '🥈', '🥉']
            for i, row in df_sorted.iterrows():
                rank = i + 1
                medal = medals[i] if i < 3 else '  '
                model = row['model']
                auc = f"{row['auc']:.4f}" if pd.notna(row['auc']) else 'N/A'
                acc = f"{row['acc']:.4f}" if 'acc' in row and pd.notna(row['acc']) else 'N/A'
                eer = f"{row['eer']:.4f}" if 'eer' in row and pd.notna(row['eer']) else 'N/A'
                ap = f"{row['ap']:.4f}" if 'ap' in row and pd.notna(row['ap']) else 'N/A'
                
                print(f"{rank:<6} {medal:<8} {model:<20} {auc:<10} {acc:<10} {eer:<10} {ap:<10}")
            
            # 打印统计信息
            print("\n" + "=" * 70)
            print("STATISTICS")
            print("=" * 70)
            if not df_sorted['auc'].isna().all():
                print(f"  Best AUC:    {df_sorted['auc'].max():.4f}  ({df_sorted.iloc[0]['model']})")
                print(f"  Worst AUC:   {df_sorted['auc'].min():.4f}  ({df_sorted.iloc[-1]['model']})")
                print(f"  Mean AUC:    {df_sorted['auc'].mean():.4f}")
                print(f"  Median AUC:  {df_sorted['auc'].median():.4f}")
                print(f"  Std Dev:     {df_sorted['auc'].std():.4f}")
        
        # ==================== 生成可视化 ====================
        viz_dir = create_visualizations(df, TEST_DATASET, timestamp)
        
        # ==================== 打印完整结果表格 ====================
        print("\n" + "=" * 70)
        print("COMPLETE RESULTS TABLE")
        print("=" * 70)
        print(df.to_string(index=False))
        
        # ==================== 最终总结 ====================
        print("\n" + "=" * 70)
        print("📁 OUTPUT FILES")
        print("=" * 70)
        print(f"  CSV Results:         {output_file}")
        print(f"  Visualizations:      {viz_dir}/")
        print(f"    - 01_auc_comparison.png")
        print(f"    - 02_multi_metric_comparison.png")
        print(f"    - 03_ranking_visualization.png")
        print(f"    - 04_radar_chart_top5.png")
        print(f"    - 05_performance_heatmap.png")
    
    print("\n" + "=" * 70)
    print("✅ Batch Testing Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
