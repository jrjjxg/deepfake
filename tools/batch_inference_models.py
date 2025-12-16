"""
批量推理脚本 - 使用所有 DeepfakeBench 模型测试 UADFV 数据集

功能：
1. 自动遍历所有训练好的模型权重
2. 逐个对 UADFV 数据集进行推理
3. 汇总所有模型的性能指标
4. 生成对比报告

使用方法：
python tools/batch_inference_models.py
"""

import os
import sys
import yaml
import torch
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training.test import prepare_testing_data, test_epoch, init_seed
from training.detectors import DETECTOR

# ========== 配置 ==========
class BatchInferenceConfig:
    # 项目根目录
    PROJECT_ROOT = "E:/DeepfakeBench"
    
    # 模型权重目录
    WEIGHTS_DIR = "E:/DeepfakeBench/training/weights"
    
    # 测试数据集
    TEST_DATASET = ["UADFV"]  # 可以添加多个: ["UADFV", "CelebDFv2"]
    
    # 配置文件目录
    CONFIG_DIR = "E:/DeepfakeBench/training/config"
    
    # 结果保存目录
    RESULTS_DIR = "E:/DeepfakeBench/batch_inference_results"
    
    # 模型配置映射（模型名 -> 配置文件）
    MODEL_CONFIG_MAP = {
        'capsule': 'detector/capsule_net.yaml',
        'cnnaug': 'detector/resnet34.yaml',  # CNN Augmentation 使用 ResNet34
        'core': 'detector/core.yaml',
        'effnb4': 'detector/efficientnetb4.yaml',
        'f3net': 'detector/f3net.yaml',
        'ffd': 'detector/ffd.yaml',
        'meso4': 'detector/meso4.yaml',
        'meso4Incep': 'detector/meso4Inception.yaml',
        'recce': 'detector/recce.yaml',
        'spsl': 'detector/spsl.yaml',
        'srm': 'detector/srm.yaml',
        'ucf': 'detector/ucf.yaml',
        'xception': 'detector/xception.yaml',
    }


def get_model_list(weights_dir: str):
    """获取所有可用的模型权重"""
    model_weights = []
    
    for weight_file in os.listdir(weights_dir):
        if weight_file.endswith('.pth'):
            model_name = weight_file.replace('_best.pth', '')
            weight_path = os.path.join(weights_dir, weight_file)
            
            # 检查是否有对应的配置文件
            if model_name in BatchInferenceConfig.MODEL_CONFIG_MAP:
                config_file = BatchInferenceConfig.MODEL_CONFIG_MAP[model_name]
                config_path = os.path.join(BatchInferenceConfig.CONFIG_DIR, config_file)
                
                if os.path.exists(config_path):
                    model_weights.append({
                        'name': model_name,
                        'weight_path': weight_path,
                        'config_path': config_path
                    })
                else:
                    print(f"[Warning] Config not found for {model_name}: {config_path}")
            else:
                print(f"[Warning] No config mapping for model: {model_name}")
    
    return model_weights


def load_model(model_info: dict, config: dict, device):
    """加载单个模型"""
    model_name = config['model_name']
    weight_path = model_info['weight_path']
    
    # 创建模型
    model_class = DETECTOR[model_name]
    model = model_class(config).to(device)
    
    # 加载权重
    try:
        ckpt = torch.load(weight_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print(f"✓ Loaded weights: {weight_path}")
        return model
    except Exception as e:
        print(f"✗ Failed to load weights for {model_info['name']}: {e}")
        return None


def test_single_model(model_info: dict, test_datasets: list, device):
    """测试单个模型"""
    print(f"\n{'='*70}")
    print(f"Testing Model: {model_info['name']}")
    print(f"{'='*70}")
    
    # 加载配置
    with open(model_info['config_path'], 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载测试配置
    test_config_path = os.path.join(BatchInferenceConfig.CONFIG_DIR, 'test_config.yaml')
    with open(test_config_path, 'r') as f:
        test_config = yaml.safe_load(f)
    
    config.update(test_config)
    config['test_dataset'] = test_datasets
    config['weights_path'] = model_info['weight_path']
    
    # 初始化随机种子
    init_seed(config)
    
    # 准备测试数据
    print("Preparing test data...")
    test_data_loaders = prepare_testing_data(config)
    
    # 加载模型
    model = load_model(model_info, config, device)
    if model is None:
        return None
    
    # 测试
    print("Running inference...")
    model.eval()
    metrics = test_epoch(model, test_data_loaders)
    
    # 清理内存
    del model
    torch.cuda.empty_cache()
    
    return metrics


def save_results(all_results: dict, output_dir: str):
    """保存结果到 CSV 和文本文件"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 准备数据框
    results_data = []
    
    for model_name, metrics in all_results.items():
        if metrics is None:
            continue
        
        for dataset_name, dataset_metrics in metrics.items():
            row = {
                'Model': model_name,
                'Dataset': dataset_name,
                'AUC': dataset_metrics.get('auc', 0.0),
                'ACC': dataset_metrics.get('acc', 0.0),
                'EER': dataset_metrics.get('eer', 0.0),
                'AP': dataset_metrics.get('ap', 0.0),
            }
            results_data.append(row)
    
    # 保存为 CSV
    df = pd.DataFrame(results_data)
    csv_file = os.path.join(output_dir, f'batch_inference_results_{timestamp}.csv')
    df.to_csv(csv_file, index=False)
    print(f"\n✓ Results saved to CSV: {csv_file}")
    
    # 保存为格式化文本
    txt_file = os.path.join(output_dir, f'batch_inference_results_{timestamp}.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Batch Inference Results - All Models on UADFV\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 按数据集分组
        for dataset in df['Dataset'].unique():
            f.write(f"\n{'='*70}\n")
            f.write(f"Dataset: {dataset}\n")
            f.write(f"{'='*70}\n\n")
            
            dataset_df = df[df['Dataset'] == dataset].sort_values('AUC', ascending=False)
            f.write(dataset_df.to_string(index=False))
            f.write("\n\n")
            
            # 最佳模型
            if len(dataset_df) > 0:
                best_model = dataset_df.iloc[0]
                f.write(f"Best Model: {best_model['Model']}\n")
                f.write(f"  AUC: {best_model['AUC']:.4f}\n")
                f.write(f"  ACC: {best_model['ACC']:.4f}\n")
                f.write(f"  EER: {best_model['EER']:.4f}\n")
                f.write(f"  AP:  {best_model['AP']:.4f}\n\n")
    
    print(f"✓ Results saved to TXT: {txt_file}")
    
    # 显示汇总表格
    print(f"\n{'='*70}")
    print("Summary Table:")
    print(f"{'='*70}")
    print(df.to_string(index=False))


def main():
    print("\n" + "="*70)
    print("Batch Inference - All DeepfakeBench Models on UADFV")
    print("="*70 + "\n")
    
    # 检查 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 获取模型列表
    print("Scanning for available models...")
    model_list = get_model_list(BatchInferenceConfig.WEIGHTS_DIR)
    print(f"Found {len(model_list)} models with valid configurations:\n")
    for i, model_info in enumerate(model_list, 1):
        print(f"  {i}. {model_info['name']}")
    print()
    
    # 测试所有模型
    all_results = {}
    
    for i, model_info in enumerate(model_list, 1):
        print(f"\n[{i}/{len(model_list)}] Testing {model_info['name']}...")
        
        try:
            metrics = test_single_model(
                model_info,
                BatchInferenceConfig.TEST_DATASET,
                device
            )
            all_results[model_info['name']] = metrics
            
            # 显示当前模型结果
            if metrics:
                for dataset, dataset_metrics in metrics.items():
                    print(f"\n  {dataset}:")
                    print(f"    AUC: {dataset_metrics.get('auc', 0.0):.4f}")
                    print(f"    ACC: {dataset_metrics.get('acc', 0.0):.4f}")
                    print(f"    EER: {dataset_metrics.get('eer', 0.0):.4f}")
        
        except Exception as e:
            print(f"✗ Error testing {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_info['name']] = None
    
    # 保存结果
    print(f"\n{'='*70}")
    print("Saving results...")
    save_results(all_results, BatchInferenceConfig.RESULTS_DIR)
    
    print(f"\n{'='*70}")
    print("✓ Batch inference complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
