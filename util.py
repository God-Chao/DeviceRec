import os
import copy
import random
from model.LR import LRModel
from model.WideDeep import WideDeepModel
from model.PNN import PNNModel
from model.Dataset import MovieLensDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import os.path as osp
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split

def compute_user_level_auc(model, data_path, all_user_fp, device, logger, is_log):
    auc_list = []
    for user in all_user_fp:
        user_fp = osp.join(data_path, user)
        user_test_data_path = osp.join(user_fp, 'test.csv')

        user_test_data = pd.read_csv(user_test_data_path)
        user_test_dataset = MovieLensDataset(user_test_data)
        user_test_dataloader = DataLoader(user_test_dataset, batch_size=len(user_test_data), shuffle=False)

        test_auc = compute_auc(model, user_test_dataloader, device)
        if is_log == True:
            logger.info(f'{user}, auc: {test_auc}')
        auc_list.append(test_auc)
    auc_list = [x for x in auc_list if x != -1]
    return sum(auc_list) / len(auc_list)

def get_all_subdirectories(directory):
    try:
        all_items = os.listdir(directory)
        subdirs = [item for item in all_items if os.path.isdir(os.path.join(directory, item))]
        return sorted(subdirs)
    except FileNotFoundError:
        print(f"目录 {directory} 不存在")
        return []
    except PermissionError:
        print(f"无权限访问目录 {directory}")
        return []
    
def get_model_by_name(model_name):
    if model_name == 'LR':
        model = LRModel()
    elif model_name == 'WideDeep':
        model = WideDeepModel()
    elif model_name == 'PNN':
        model = PNNModel()
    return model

def train_model_on_dataloader(model, dataloader, optimizer, criterion, device, output_loss=False):
    model.train()
    train_loss = 0.
    for batch in dataloader:
        user_ids = batch['user_id'].to(device)
        movie_ids = batch['movie_id'].to(device)
        gender = batch['gender'].to(device)
        age = batch['age'].to(device)
        occupation = batch['occupation'].to(device)
        genres = batch['genres'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        preds = model(user_ids, movie_ids, gender, age, occupation, genres)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(dataloader)
    if output_loss == True:
        return avg_train_loss
    else:
        return

# 计算 AUC 的函数
def compute_auc(model, dataloader, device):
    model.eval()  # 设置为评估模式
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # 禁用梯度计算
        for batch in dataloader:
            user_ids = batch['user_id'].to(device)
            movie_ids = batch['movie_id'].to(device)
            gender = batch['gender'].to(device)
            age = batch['age'].to(device)
            occupation = batch['occupation'].to(device)
            genres = batch['genres'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播，获取预测概率
            preds = model(user_ids, movie_ids, gender, age, occupation, genres)
            
            # 将预测和标签存储到列表中
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # 拼接所有批次的预测和标签
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    if len(np.unique(all_labels)) < 2:
        return -1

    # 计算 AUC
    auc = roc_auc_score(all_labels, all_preds)
    return auc

def get_user_data(user_fp):
    user_train_data_path = osp.join(user_fp, 'train.csv')
    user_valid_data_path = osp.join(user_fp, 'valid.csv')
    user_test_data_path = osp.join(user_fp, 'test.csv')

    user_train_data = pd.read_csv(user_train_data_path)
    user_valid_data = pd.read_csv(user_valid_data_path)
    user_test_data = pd.read_csv(user_test_data_path)

    return user_train_data, user_valid_data, user_test_data

def train_model_on_dataloader_with_valid(model, train_dataloader, valid_dataloader, optimizer, criterion, device, patience=5, max_epochs=30):
    best_valid_auc = compute_auc(model, valid_dataloader, device)
    best_model_state = copy.deepcopy(model.state_dict())  # 保存最佳模型状态
    best_epoch = 0
    counter = 0  # 早停计数器
    for epoch in range(max_epochs):
        # 在训练集上训练
        train_model_on_dataloader(model, train_dataloader, optimizer, criterion, device, False)

        # 在验证集上测试AUC
        current_valid_auc = compute_auc(model, valid_dataloader, device)
        
        # 如果当前AUC比最佳AUC好，更新最佳AUC和模型状态
        if current_valid_auc > best_valid_auc:
            best_epoch = epoch + 1
            best_valid_auc = current_valid_auc
            best_model_state = copy.deepcopy(model.state_dict())  # 保存当前模型状态
            counter = 0  # 重置计数器
        else:
            counter += 1  # AUC下降，计数器加1
        
        # 如果计数器达到耐心值，提前退出，返回最佳模型状态
        if counter >= patience:
            model.load_state_dict(best_model_state)
            return best_epoch, best_valid_auc
        
    # 如果达到最大epoch，返回最佳模型状态
    model.load_state_dict(best_model_state)
    return best_epoch, best_valid_auc

def random_recall(all_user_fp, user_fp, recall_num):
    all_user_fp = [user for user in all_user_fp if user != user_fp]
    return random.sample(all_user_fp, recall_num)

def merge_external_data(data_path, external_users):
    all_external_data = []
    for external_user in external_users:
        external_user_fp = osp.join(data_path, external_user)
        external_train_data, external_valid_data, external_test_data = get_user_data(external_user_fp)
        all_external_data.append(external_train_data)
    merged_external_train_data = pd.concat(all_external_data, ignore_index=True)
    return merged_external_train_data

def get_top_k_similar_users(user_id, k, similarity_matrix, user_id_to_index, index_to_user_id):
    if user_id not in user_id_to_index:
        raise ValueError(f"用户 ID {user_id} 不在用户列表中")

    user_index = user_id_to_index[user_id]  # 获取用户索引
    user_similarities = similarity_matrix[user_index]  # 取出该用户的相似度向量

    # 获取最相似的 k 个用户（排除自己）
    similar_user_indices = np.argsort(user_similarities)[::-1]  # 降序排序
    top_k_indices = [idx for idx in similar_user_indices if idx != user_index][:k]  # 排除自身，取前 k 个

    # 将索引转换回用户 ID
    top_k_user_ids = [f"user_{index_to_user_id[idx]}" for idx in top_k_indices]

    return top_k_user_ids

def split_df_into_batches(df, batch_size=64):
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)  # 随机打乱数据
    batches = [df.iloc[i: i + batch_size] for i in range(0, len(df), batch_size)]
    return batches