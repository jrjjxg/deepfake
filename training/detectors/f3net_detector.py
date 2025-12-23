'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the F3netDetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{qian2020thinking,
  title={Thinking in frequency: Face forgery detection by mining frequency-aware clues},
  author={Qian, Yuyang and Yin, Guojun and Sheng, Lu and Chen, Zixuan and Shao, Jing},
  booktitle={European conference on computer vision},
  pages={86--103},
  year={2020},
  organization={Springer}
}

GitHub Reference:
https://github.com/yyk-wew/F3Net

Notes:
We replicate the results by solely utilizing the FAD branch, following the reference GitHub implementation (https://github.com/yyk-wew/F3Net).
'''

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='f3net')
class F3netDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        # modules only use in FAD
        img_size = config['resolution']
        self.FAD_head = FAD_Head(img_size)

    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        
        # 检查是否是已训练的检测器模型（通过 "best" 或 "FAD" 关键词判断）
        pretrained_path = config['pretrained']
        state_dict = torch.load(pretrained_path)
        
        # 判断是否是已训练的F3Net检测器（包含FAD_head等键）
        is_trained_detector = any('FAD' in k for k in state_dict.keys()) or any('backbone.conv1' in k for k in state_dict.keys())
        
        if is_trained_detector:
            # 从已训练的F3Net检测器加载 - 不需要修改conv1
            logger.info(f'Detected trained F3Net detector weights')
            # 提取backbone部分
            backbone_state = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_key = k.replace('backbone.', '')
                    backbone_state[new_key] = v
            
            if backbone_state:
                # === 关键修复：过滤掉形状不匹配的键 ===
                model_state = backbone.state_dict()
                filtered_state = {}
                for k, v in backbone_state.items():
                    if k in model_state:
                        if v.shape == model_state[k].shape:
                            filtered_state[k] = v
                        else:
                            logger.warning(f"Skipping layer {k} due to shape mismatch: ckpt {v.shape} vs model {model_state[k].shape}")
                    else:
                        # 这是一个unexpected key，会被strict=False忽略，但我们在filtered中加上也没事
                        pass
                
                missing, unexpected = backbone.load_state_dict(filtered_state, strict=False)
                if missing:
                    logger.info(f'Missing keys: {missing[:5]}...')
                logger.info(f'Loaded backbone weights from trained detector (safe mode)')
            return backbone
        
        # 原始逻辑：从ImageNet预训练的Xception加载
        # 处理 pointwise 层
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        
        # 移除 fc 层
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        
        # 提取 conv1 数据（支持两种格式）
        conv1_data = None
        if 'conv1.weight' in state_dict:
            conv1_data = state_dict['conv1.weight'].clone()
            del state_dict['conv1.weight']
        elif 'backbone.conv1.weight' in state_dict:
            conv1_data = state_dict['backbone.conv1.weight'].clone()
            # 移除所有 backbone. 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('backbone.', '')
                if new_key != 'conv1.weight':  # 跳过 conv1
                    new_state_dict[new_key] = v
            state_dict = new_state_dict
        
        if conv1_data is None:
            logger.warning("Cannot find conv1.weight in state_dict, using random init")
            # 使用随机初始化
            backbone.conv1 = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
            return backbone
        
        # 先创建新的 conv1（12通道输入）
        backbone.conv1 = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
        
        # 初始化新 conv1 的权重
        for i in range(4):
            backbone.conv1.weight.data[:, i*3:(i+1)*3, :, :] = conv1_data / 4.0
        
        # 加载其余权重（使用 strict=False 忽略不匹配的层）
        missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.info(f'Missing keys (expected): {missing_keys[:5]}...')
        if unexpected_keys:
            logger.info(f'Unexpected keys: {unexpected_keys[:5]}...')
        
        logger.info('Load pretrained model from {}'.format(config['pretrained']))
        logger.info('Initialized conv1 for 12-channel input')
        
        return backbone
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        fea_FAD = self.FAD_head(data_dict['image']) # [B, 12, 256, 256]
        return self.backbone.features(fea_FAD)

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}

        return pred_dict


# ===================================== other modules for F3Net # =====================================


# Filter Module
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


# FAD Module
class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 2.82)
        middle_filter = Filter(size, size // 2.82, size // 2)
        high_filter = Filter(size, size // 2, size * 2)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 299, 299]

        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
            y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 299, 299]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)    # [N, 12, 299, 299]
        return out

# utils
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.
