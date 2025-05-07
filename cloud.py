from util import *

import time
import random
import warnings
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader
import os.path as osp

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    parser = argparse.ArgumentParser(description='eval global model cloud')
    parser.add_argument('--data_path', type=str, default='/home/chao/workspace/DeviceRec-simple/data')
    parser.add_argument('--cloud_model_path', type=str, default='/home/chao/workspace/DeviceRec-simple/cloud_model')
    parser.add_argument('--exp_result_path', type=str, default='/home/chao/workspace/DeviceRec-simple/test')
    parser.add_argument('--model_name', type=str, default='LR')
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    # 设置日志
    result_path = osp.join(args.exp_result_path, f'cloud_{args.model_name}')
    os.makedirs(result_path, exist_ok=True)

    log_dir = osp.join(result_path, 'log')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=osp.join(log_dir, f'{args.model_name}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode = 'w'
    )
    logger = logging.getLogger(__name__)

    logger.info(f'args: {args}')

    # 固定随机种子
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f'随机种子设置完毕')

    # 全局模型参数
    cloud_model_state_path = osp.join(args.cloud_model_path, f'{args.model_name}.pth')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # 加载模型
    model = get_model_by_name(args.model_name)
    model.load_state_dict(torch.load(cloud_model_state_path))
    model = model.to(device)

    # 读取用户
    all_user_fp = get_all_subdirectories(args.data_path)

    start_time = time.time()  # 记录训练开始时间

    cloud_auc = compute_user_level_auc(model, args.data_path, all_user_fp, device, logger, True)

    logger.info(f'{args.model_name} Cloud AUC: {cloud_auc}')

    end_time = time.time()

    # 计算训练总时长（小时和分钟）
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    logger.info(f"总训练时间: {hours} 小时 {minutes} 分钟 {seconds} 秒")

if __name__ == '__main__':
    main() # nohup python cloud.py --model_name=LR --device=cuda:4 2>&1 &