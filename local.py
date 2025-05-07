from util import *

import time
import random
import warnings
import logging
import argparse
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
    parser = argparse.ArgumentParser(description='eval global model local')
    parser.add_argument('--data_path', type=str, default='/home/chao/workspace/DeviceRec-simple/data')
    parser.add_argument('--cloud_model_path', type=str, default='/home/chao/workspace/DeviceRec-simple/cloud_model')
    parser.add_argument('--exp_result_path', type=str, default='/home/chao/workspace/DeviceRec-simple/exp_result')
    parser.add_argument('--model_name', type=str, default='LR')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()

    # 设置日志
    result_path = osp.join(args.exp_result_path, f'local_{args.model_name}')
    os.makedirs(result_path, exist_ok=True)

    log_path = osp.join(result_path, 'log')
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        filename=osp.join(log_path, f'local_{args.model_name}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode = 'w'
    )
    logger = logging.getLogger(__name__)

    # 打印配置信息
    logger.info(f'args: {args}')
    
    # 固定随机种子
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info('随机种子设置完毕')

    # 全局模型参数
    cloud_model_state_path = osp.join(args.cloud_model_path, f'{args.model_name}.pth')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # 读取用户
    all_user_fp = get_all_subdirectories(args.data_path)

    start_time = time.time()  # 记录训练开始时间

    # 评估local auc
    user_id_list = []
    local_auc_list = []
    train_epoch_list = []
    user_train_data_len_list = []
    for user in all_user_fp:
        user_id_list.append(user)
        user_fp = osp.join(args.data_path, user)
        # 加载模型
        model = get_model_by_name(args.model_name)
        model.load_state_dict(torch.load(cloud_model_state_path))
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.BCELoss()

        user_train_data, user_valid_data, user_test_data = get_user_data(user_fp)

        user_train_dataset = MovieLensDataset(user_train_data)
        user_train_dataloader = DataLoader(user_train_dataset, batch_size=args.batch_size, shuffle=True)

        user_valid_dataset = MovieLensDataset(user_valid_data)
        user_valid_dataloader = DataLoader(user_valid_dataset, batch_size=len(user_valid_data), shuffle=False)

        user_test_dataset = MovieLensDataset(user_test_data)
        user_test_dataloader = DataLoader(user_test_dataset, batch_size=len(user_test_data), shuffle=False)

        D = len(user_train_data)
        user_train_data_len_list.append(D)

        # 在本地训练集上训练至拟合
        train_epoch, best_valid_auc = train_model_on_dataloader_with_valid(model, user_train_dataloader, user_valid_dataloader, optimizer, criterion, device)
        
        # 在验证集上测试auc
        test_auc = compute_auc(model, user_test_dataloader, device)
        local_auc_list.append(test_auc)
        train_epoch_list.append(train_epoch)
        logger.info(f'{user}, auc: {test_auc} train_epoch: {train_epoch}, D: {D} best_valid_auc: {best_valid_auc}, test_auc: {test_auc}')

    # 保存结果
    result = {"user_id": user_id_list, "local_auc": local_auc_list, "local_train_epoch": train_epoch_list, 'D': user_train_data_len_list}
    df = pd.DataFrame(result)
    csv_path = osp.join(result_path, f'{args.model_name}.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV 文件已保存至{csv_path}")

    local_auc_list = [x for x in local_auc_list if x != -1]
    local_auc = sum(local_auc_list) / len(local_auc_list)
    avg_train_epoch = sum(train_epoch_list) / len(train_epoch_list)
    logger.info(f'{args.model_name} local auc: {local_auc}, average train epoch: {avg_train_epoch}')

    end_time = time.time() # 记录结束时间

    merge_result_path = osp.join(result_path, f'result.txt')
    with open(merge_result_path, 'a') as f:
        f.write(f'time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}\n\nargs:\nmodel_name: {args.model_name}\nlr: {args.lr}\nbatch_size: {args.batch_size}\ndevice: {args.device}\n\nresult:\nLocal AUC: {local_auc}\nAverage train epoch: {avg_train_epoch}\n')

    # 计算训练总时长（小时和分钟）
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    logger.info(f"总训练时间: {hours} 小时 {minutes} 分钟 {seconds} 秒")

if __name__ == '__main__':
    main() # nohup python local.py --model_name=LR --device=cuda:2 2>&1 &