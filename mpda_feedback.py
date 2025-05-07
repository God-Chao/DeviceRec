from util import *

import os
from copy import deepcopy
import time
import json
import random
import shutil
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
# 保存每一轮的用户模型参数

# 更新用户嵌入的函数
def update_user_embeddings(args, user_embeddings, updated_user_embeddings, user_id, external_users, feedback, user_id_to_index, alpha, beta):
    user_idx = user_id_to_index[user_id]
    user_embed = user_embeddings[user_idx]
    
    # 计算增强用户嵌入与目标用户嵌入的平均差异
    ext_indices = [user_id_to_index[int(ext_user.split('_')[-1])] for ext_user in external_users]
    ext_embeddings = user_embeddings[ext_indices]  
    differences = ext_embeddings - user_embed  
    diff = np.mean(differences, axis=0)  

    if feedback == 1:
        updated_user_embeddings[user_idx] += alpha * diff  # 拉近
    else:
        updated_user_embeddings[user_idx] -= beta * diff   # 拉远
    
    # 归一化嵌入
    if args.embedding_norm == 'true':
        updated_user_embeddings[user_idx] /= np.linalg.norm(updated_user_embeddings[user_idx])

    return user_embeddings

def run_exp_for_user(args, epoch, tmp_model_path, user, device, logger, user_id_to_index, index_to_user_id, similarity_matrix=None):
    # 加载用户模型
    model = get_model_by_name(args.model_name)
    user_tmp_model_path = osp.join(tmp_model_path, f'{user}.pth')

    if epoch == 0:
        cloud_model_state_path = osp.join(args.cloud_model_path, f'{args.model_name}.pth')
        logger.info(f'loading user init model: {cloud_model_state_path}')
        model.load_state_dict(torch.load(cloud_model_state_path, map_location=device))
    else:
        logger.info(f'loading user tmp model: {user_tmp_model_path}')
        model.load_state_dict(torch.load(user_tmp_model_path, map_location=device))

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

    # 先本地微调
    if epoch == 0:
        optimizer = optim.Adam(model.parameters(), lr=args.local_lr)
        mpda_local_train_epoch, _ = train_model_on_dataloader_with_valid(model, local_train_dataloader, local_valid_dataloader, optimizer, criterion, device)
        mpda_minus_auc = compute_auc(model, local_test_dataloader, device)
        logger.info(f'mpda_local_train_epoch: {mpda_local_train_epoch}')
        # 保存模型
        torch.save(model.state_dict(), user_tmp_model_path)

    # 计算初始 AUC
    init_valid_auc = compute_auc(model, local_valid_dataloader, device)

    # 再增强数据微调
    optimizer = torch.optim.Adam(model.parameters(), lr=args.external_lr)
    all_external_data = merge_external_data(args.data_path, external_users)
    all_external_dataset = MovieLensDataset(all_external_data)
    all_external_dataloader = DataLoader(all_external_dataset, batch_size=args.external_batch_size, shuffle=True)
    external_train_epoch, current_valid_auc = train_model_on_dataloader_with_valid(model, all_external_dataloader, local_valid_dataloader, optimizer, criterion, device)
    delta_auc = current_valid_auc - init_valid_auc
    if delta_auc > 1e-6:
        best_valid_auc = current_valid_auc
        # 保存模型
        torch.save(model.state_dict(), user_tmp_model_path)
        feedback = 1
    else:
        best_valid_auc = init_valid_auc
        feedback = 0

    # 计算 MPDA AUC
    mpda_auc = compute_auc(model, local_test_dataloader, device)

    return D, init_valid_auc, current_valid_auc, delta_auc, best_valid_auc, mpda_auc, feedback, external_users

def main():
    parser = argparse.ArgumentParser(description='eval global model mpda with embedding update')
    parser.add_argument('--data_path', type=str, default='/home/chao/workspace/DeviceRec-simple/data')
    parser.add_argument('--cloud_model_path', type=str, default='/home/chao/workspace/DeviceRec-simple/cloud_model')
    parser.add_argument('--exp_result_path', type=str, default='/home/chao/workspace/DeviceRec-simple/exp')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--local_batch_size', type=int)
    parser.add_argument('--external_batch_size', type=int)
    parser.add_argument('--external_lr', type=float)
    parser.add_argument('--local_lr', type=float)
    parser.add_argument('--recall_num', type=int)
    parser.add_argument('--task_index', type=int)
    parser.add_argument('--task_count', type=int, default=15)
    parser.add_argument('--current_time', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--decay_rate', type=float, default=1)
    parser.add_argument('--decay', type=str)
    parser.add_argument('--embedding_norm', type=str)

    args = parser.parse_args()

    # 设置日志
    exp_path = osp.join(args.exp_result_path, f'mpda_feedback_{args.model_name}_{args.current_time}')
    os.makedirs(exp_path, exist_ok=True)

    log_dir = osp.join(exp_path, 'log')
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

    # 读取用户
    all_user_fp = get_all_subdirectories(args.data_path)

    # 划分测试用户
    split_user_fp = np.array_split(np.array(all_user_fp), args.task_count)[args.task_index]
    split_user_fp = sorted(split_user_fp)

    logger.info(f'split_user_fp: {split_user_fp}, {len(split_user_fp)}')

    # 记录每个用户的模型
    tmp_model_dir = osp.join(exp_path, 'tmp_model')
    os.makedirs(tmp_model_dir, exist_ok=True)

    # 写入参数配置信息
    if args.task_index == 0:
        merge_result_path = osp.join(exp_path, f'args.txt')
        with open(merge_result_path, 'a') as f:
            f.write(f'time={args.current_time}\n\n')
            f.write(f'device={args.device}\n')
            f.write(f'model_name={args.model_name}\n\n')
            f.write(f'recall_num={args.recall_num}\n\n')
            f.write(f'decay={args.decay}\n')
            f.write(f'embedding_norm={args.embedding_norm}\n')
            f.write(f'decay_rate={args.decay_rate}\n\n')
            f.write(f'local_lr={args.local_lr}\n')
            f.write(f'external_lr={args.external_lr}\n\n')
            f.write(f'local_batch_size={args.local_batch_size}\n')
            f.write(f'external_batch_size={args.external_batch_size}\n\n')
            f.write(f'epochs={args.epochs}\n\n')
            f.write(f'alpha={args.alpha}\n')
            f.write(f'beta={args.beta}\n')

        # 构建每个用户暂存模型，初始都为云端全局模型
        src_path = osp.join(args.cloud_model_path, f'{args.model_name}.pth')  # 云端全局模型参数路径
        for user in all_user_fp:
            user_tmp_model_path = osp.join(tmp_model_dir, f'{user}.pth')
            shutil.copy2(src_path, user_tmp_model_path)

    start_time = time.time()  # 记录训练开始时间

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

    # 设置decay
    if args.decay == 'false':
        args.decay_rate = 1
        logger.info(f'decay_rate 被设置为 1')

    # 迭代更新嵌入
    for epoch in range(args.epochs):
        # 当前epoch的结果目录
        current_result_path = osp.join(exp_path, 'result', f'epoch{epoch}')
        os.makedirs(current_result_path, exist_ok=True)
        logger.info(f'Epoch{epoch} current_result_path: {current_result_path}')

        # 衰减 alpha 和 beta（指数衰减）
        current_alpha = args.alpha * (args.decay_rate ** epoch)
        current_beta = args.beta * (args.decay_rate ** epoch)
        logger.info(f"Epoch{epoch} current_alpha={current_alpha:.6f}, current_beta={current_beta:.6f}")
        
        epoch_list = []
        user_id_list = []
        D_list = []
        init_valid_auc_list = []
        current_valid_auc_list = []
        best_valid_auc_list = []
        delta_auc_list = []
        mpda_auc_list = []
        feedback_list = []
        external_users_list = []

        feedbacks = {}  # 存储每个用户的反馈和外部用户

        # 评估模型并收集反馈
        for index, user in enumerate(split_user_fp):
            D, init_valid_auc, current_valid_auc, delta_auc, best_valid_auc, mpda_auc, feedback, external_users = run_exp_for_user(
                args, epoch, tmp_model_dir, user, device, logger, user_id_to_index, index_to_user_id, similarity_matrix
            )
            logger.info(f'[Epoch{epoch}] {user} {index}/{len(split_user_fp)}, feedback: {feedback}, Init Valid: {init_valid_auc}, Current Valid: {current_valid_auc}, Delta AUC: {delta_auc}, MPDA: {mpda_auc}')
            
            # 收集用户反馈
            feedbacks[user] = (feedback, external_users)
            
            epoch_list.append(epoch)
            user_id = user.split('_')[1]
            user_id_list.append(user_id)
            D_list.append(D)
            init_valid_auc_list.append(init_valid_auc)
            current_valid_auc_list.append(current_valid_auc)
            delta_auc_list.append(delta_auc)
            best_valid_auc_list.append(best_valid_auc)
            mpda_auc_list.append(mpda_auc)
            feedback_list.append(feedback)
            external_users_str = ','.join([u.split('_')[1] for u in external_users])
            external_users_list.append(external_users_str)

        # 保存当前epoch下的结果文件和feedbacks
        feedbacks_path = osp.join(current_result_path, f"{args.task_index}.jsonl")
        with open(feedbacks_path, 'w', encoding='utf-8') as f:
            for user, (feedback, external_users) in feedbacks.items():
                entry = {"user_id": user, "feedback": feedback, "external_users": external_users}
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logger.info(f"epoch {epoch} task{args.task_index} feedbacks结果已保存至: {feedbacks_path}")

        result = {"epoch": epoch_list,"user_id": user_id_list, "D": D_list, "init_valid": init_valid_auc_list, "current_valid": current_valid_auc_list, "delta_auc": delta_auc_list, "best_valid": best_valid_auc_list, "MPDA": mpda_auc_list , "feedback": feedback_list, "external_users": external_users_list}
        df = pd.DataFrame(result)
        csv_path = osp.join(current_result_path, f'{args.task_index}.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"epoch{epoch} task{args.task_index} 结果已保存至: {csv_path}")

        # 等待该epoch下其他所有任务完成
        while True:
            completed_tasks = len([f for f in os.listdir(current_result_path) if f.endswith('.csv')])
            completed_feedbacks = len([f for f in os.listdir(current_result_path) if f.endswith('.jsonl')])
            if completed_tasks == args.task_count and completed_feedbacks == args.task_count:
                logger.info(f"epoch {epoch}: all {args.task_count} tasks and feedbacks completed")
                break
            time.sleep(20)

        # 计算该epoch的结果并打印
        if args.task_index == 0:
            # 读取该epoch的结果文件并合并
            result_files_path = [osp.join(current_result_path, f'{ti}.csv') for ti in range(args.task_count)]
            result_df_list = [pd.read_csv(file) for file in result_files_path]
            merged_result_df = pd.concat(result_df_list, ignore_index=True)

            # 保存合并后的结果
            merged_result_path = osp.join(exp_path, f'result{epoch}.csv')
            merged_result_df.to_csv(merged_result_path, index=False)

            # 求平均
            exclude_columns = {'user_id', 'D', 'init_valid', 'current_valid', 'delta_auc', 'external_users'}
            filtered_columns = [col for col in merged_result_df.columns if col not in exclude_columns]
            column_means = merged_result_df[filtered_columns].mean()

            # 写入结果
            result_txt_path = osp.join(exp_path, 'result.txt')
            with open(result_txt_path, 'a') as f:
                f.write(f'Epoch {epoch}: ')
                for col, mean_value in column_means.items():
                    f.write(f'{col}= {mean_value}  ')
                f.write('\n')
            
            # 合并feedbacks
            feedbacks_files = [osp.join(current_result_path, f'{ti}.jsonl') for ti in range(args.task_count)]
            all_feedbacks = {}
            for f_file in feedbacks_files:
                with open(f_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        all_feedbacks[entry["user_id"]] = (entry["feedback"], entry["external_users"])

            # 更新全局 user_embeddings
            updated_user_embeddings = deepcopy(user_embeddings)
            logger.info(f"开始更新嵌入")
            for user, (feedback, external_users) in all_feedbacks.items():
                logger.info(f"{user}:")
                user_id = int(user.split("_")[-1])
                logger.info(f"[Epoch{epoch}] current embedding: {user_embeddings[user_id_to_index[user_id]]}")
                update_user_embeddings(args, user_embeddings, updated_user_embeddings, user_id, external_users, feedback, user_id_to_index, current_alpha, current_beta)
                logger.info(f"[Epoch{epoch}] updated embedding: {updated_user_embeddings[user_id_to_index[user_id]]}")
            user_embeddings = updated_user_embeddings

            # 保存更新后的嵌入
            embed_path = osp.join(current_result_path, f'updated_user_embeddings.npy')
            np.save(embed_path, user_embeddings)
            logger.info(f"Updated global embeddings saved to {embed_path}")

            similarity_matrix = cosine_similarity(user_embeddings)  # 更新相似度矩阵
        # 其他任务等待并读取更新后的嵌入
        else:
            embed_path = osp.join(current_result_path, f'updated_user_embeddings.npy')
            while not os.path.exists(embed_path):
                time.sleep(20)
            user_embeddings = np.load(embed_path)
            similarity_matrix = cosine_similarity(user_embeddings)  # 更新相似度矩阵
            logger.info(f"Task {args.task_index} loaded updated embeddings from {embed_path}")

    end_time = time.time()

    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    logger.info(f"总训练时间: {hours} 小时 {minutes} 分钟 {seconds} 秒")

if __name__ == '__main__':
    main()