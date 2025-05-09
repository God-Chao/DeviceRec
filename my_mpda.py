from util import *

import os
from copy import deepcopy
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
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import os.path as osp

warnings.simplefilter(action='ignore', category=FutureWarning)

def run_exp_for_user(args, user, all_user_fp, device, logger, user_id_to_index, index_to_user_id, similarity_matrix=None):
    # 加载模型
    model = get_model_by_name(args.model_name)
    cloud_model_state_path = osp.join(args.cloud_model_path, f'{args.model_name}.pth')
    model.load_state_dict(torch.load(cloud_model_state_path, map_location=device))
    model = model.to(device)
    criterion = nn.BCELoss()

    # 加载本地数据
    user_fp = osp.join(args.data_path, user)
    local_train_data, local_valid_data, local_test_data = get_user_data(user_fp)
    local_train_dataset = MovieLensDataset(local_train_data)
    local_train_dataloader = DataLoader(local_train_dataset, batch_size=args.local_batch_size, shuffle=True)
    local_valid_dataset = MovieLensDataset(local_valid_data)
    local_valid_dataloader = DataLoader(local_valid_dataset, batch_size=len(local_valid_data), shuffle=False)
    local_test_dataset = MovieLensDataset(local_test_data)
    local_test_dataloader = DataLoader(local_test_dataset, batch_size=len(local_test_data), shuffle=False)

    # 计算本地数据量
    D = len(local_train_data)

    # 匹配外部用户
    user_id = int(user.split("_")[-1])
    external_users = get_top_k_similar_users(user_id, args.recall_num, similarity_matrix, user_id_to_index, index_to_user_id)
    logger.info(f'external_users: {external_users}')

    # 先本地微调
    optimizer = optim.Adam(model.parameters(), lr=args.local_lr)
    mpda_local_train_epoch, _ = train_model_on_dataloader_with_valid(model, local_train_dataloader, local_valid_dataloader, optimizer, criterion, device)
    mpda_minus_auc = compute_auc(model, local_test_dataloader, device)
    logger.info(f'mpda- auc after local finetuning: {mpda_minus_auc}')

    # 计算初始 AUC
    best_valid_auc = compute_auc(model, local_valid_dataloader, device)
    logger.info(f'init post_local_finetune valid auc: {best_valid_auc}')
    best_model_state = deepcopy(model.state_dict())

    # 再增强数据微调
    optimizer = torch.optim.Adam(model.parameters(), lr=args.external_lr)
    all_external_data = merge_external_data(args.data_path, external_users)
    all_external_dataset = MovieLensDataset(all_external_data)
    all_external_dataloader = DataLoader(all_external_dataset, batch_size=args.external_batch_size, shuffle=True)
    external_train_epoch, current_valid_auc = train_model_on_dataloader_with_valid(model, all_external_dataloader, local_valid_dataloader, optimizer, criterion, device)
    delta_auc = current_valid_auc - best_valid_auc
    if delta_auc > 1e-6:
        logger.info(f'checking external data, D: {D}, delta AUC: {delta_auc}, auc improved, current valid auc: {current_valid_auc}')
        best_valid_auc = current_valid_auc
        best_model_state = deepcopy(model.state_dict())
        ext_data_selected = 1
    else:
        logger.info(f'checking external data, D: {D}, delta AUC: {delta_auc}, auc not improve, current valid auc: {best_valid_auc}')
        model.load_state_dict(best_model_state)
        ext_data_selected = 0

    # 计算 MPDA AUC
    mpda_auc = compute_auc(model, local_test_dataloader, device)
    logger.info(f'mpda auc after external training: {mpda_auc}')

    return mpda_minus_auc, best_valid_auc, mpda_auc, ext_data_selected, mpda_local_train_epoch

def main():
    parser = argparse.ArgumentParser(description='eval global model mpda')
    parser.add_argument('--data_path', type=str, default='/home/chao/workspace/DeviceRec-simple/data')
    parser.add_argument('--cloud_model_path', type=str, default='/home/chao/workspace/DeviceRec-simple/cloud_model')
    parser.add_argument('--exp_result_path', type=str, default='/home/chao/workspace/DeviceRec-simple/exp')
    parser.add_argument('--model_name', type=str, default='LR')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--local_batch_size', type=int, default=32)
    parser.add_argument('--external_batch_size', type=int, default=32)
    parser.add_argument('--external_lr', type=float, default=1e-4)
    parser.add_argument('--local_lr', type=float, default=1e-4)
    parser.add_argument('--recall_num', type=int, default=200)
    parser.add_argument('--task_index', type=int, default=0)
    parser.add_argument('--task_count', type=int, default=15)
    parser.add_argument('--current_time', type=str)

    args = parser.parse_args()

    # 设置日志
    result_path = osp.join(args.exp_result_path, f'mpda_{args.model_name}_{args.current_time}')
    os.makedirs(result_path, exist_ok=True)

    log_dir = osp.join(result_path, 'log')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=osp.join(log_dir, f'{args.task_index}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    logger = logging.getLogger(__name__)

    logger.info(f'args: {args}')

    # 固定随机种子
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f'随机种子设置完毕')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # 写入参数配置信息
    if args.task_index == 0:
        merge_result_path = osp.join(result_path, f'args.txt')
        with open(merge_result_path, 'a') as f:
            f.write(f'time={args.current_time}\n\n')
            f.write(f'device={args.device}\n')
            f.write(f'model_name={args.model_name}\n\n')
            f.write(f'recall_num={args.recall_num}\n\n')
            f.write(f'local_lr={args.local_lr}\n')
            f.write(f'external_lr={args.external_lr}\n\n')
            f.write(f'local_batch_size={args.local_batch_size}\n')
            f.write(f'external_batch_size={args.external_batch_size}\n')

    # 读取用户
    all_user_fp = get_all_subdirectories(args.data_path)

    # 划分测试用户
    split_user_fp = np.array_split(np.array(all_user_fp), args.task_count)[args.task_index]
    split_user_fp = sorted(split_user_fp)

    logger.info(f'split_user_fp: {split_user_fp}, {len(split_user_fp)}')

    start_time = time.time()  # 记录训练开始时间

    user_id_list = []
    mpda_minus_auc_list = []
    mpda_auc_list = []
    valid_auc_list = []
    ext_data_selected_list = []
    mpda_local_train_epoch_list = []

    # 召回算法预处理
    all_user_train_path = osp.join(args.data_path, "all_users_train.csv")
    all_user_train_data = pd.read_csv(all_user_train_path)
    all_user_ids = sorted(all_user_train_data['user_id'].unique())  # 所有唯一的用户ID
    user_id_to_index = {uid: idx for idx, uid in enumerate(all_user_ids)}  # 用户 ID 映射到索引
    index_to_user_id = {idx: uid for idx, uid in enumerate(all_user_ids)}  # 反向映射
    cloud_model = get_model_by_name(args.model_name)
    cloud_model_state_path = osp.join(args.cloud_model_path, f'{args.model_name}.pth')
    cloud_model.load_state_dict(torch.load(cloud_model_state_path, map_location=device))
    cloud_model = cloud_model.to(device)

    user_embeddings = cloud_model.user_embedding.weight.data.cpu().numpy()
    user_embeddings = np.array([user_embeddings[uid] for uid in all_user_ids])
    similarity_matrix = cosine_similarity(user_embeddings)

    # 评估模型
    for index, user in enumerate(split_user_fp):
        user_id_list.append(user)
        
        mpda_minus_auc, valid_auc, mpda_auc, ext_data_selected, mpda_local_train_epoch = run_exp_for_user(args, user, all_user_fp, device, logger, user_id_to_index, index_to_user_id, similarity_matrix)

        logger.info(f'{user} {index}/{len(split_user_fp)}, ext_data_selected: {ext_data_selected}, mpda- auc: {mpda_minus_auc}, mpda auc: {mpda_auc}, valid_auc: {valid_auc}, mpda_local_train_epoch: {mpda_local_train_epoch}\n')
        mpda_minus_auc_list.append(mpda_minus_auc)
        valid_auc_list.append(valid_auc)
        mpda_auc_list.append(mpda_auc)
        ext_data_selected_list.append(ext_data_selected)
        mpda_local_train_epoch_list.append(mpda_local_train_epoch)

    end_time = time.time()

    # 保存结果
    result = {"user_id": user_id_list, "MPDA-": mpda_minus_auc_list, "MPDA": mpda_auc_list, "Valid": valid_auc_list, "ext_data_selected": ext_data_selected_list, "MPDA_local_train_epoch": mpda_local_train_epoch_list}
    csv_result_path = osp.join(result_path, 'result')
    os.makedirs(csv_result_path, exist_ok=True)

    df = pd.DataFrame(result)
    csv_path = osp.join(csv_result_path, f'{args.task_index}.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV 文件已保存至{csv_path}")

    if args.task_index == 0:
        # 等待其他任务完成，并合并结果
        while True:
            completed_tasks = len([f for f in os.listdir(csv_result_path) if f.endswith('.csv')])
            if completed_tasks == args.task_count:
                logger.info(f"all tasks completed")
                break
            time.sleep(20)
        
        # 合并结果
        result_files_path = [osp.join(csv_result_path, f'{ti}.csv') for ti in range(args.task_count)]
        result_df_list = [pd.read_csv(file) for file in result_files_path]
        merged_result_df = pd.concat(result_df_list, ignore_index=True)

        # 保存合并后的结果
        merged_result_path = osp.join(result_path, 'result.csv')
        merged_result_df.to_csv(merged_result_path, index=False)

        # 求平均
        exclude_columns = {'user_id'}
        filtered_columns = [col for col in merged_result_df.columns if col not in exclude_columns]
        column_means = merged_result_df[filtered_columns].mean()

        result_txt_path = osp.join(result_path, 'result.txt')
        with open(result_txt_path, 'a') as f:
            for col, mean_value in column_means.items():
                f.write(f'{col} = {mean_value}\n')

    # 计算训练总时长
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    logger.info(f"总训练时间: {hours} 小时 {minutes} 分钟 {seconds} 秒")

if __name__ == '__main__':
    main()