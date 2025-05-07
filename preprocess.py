import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
import numpy as np

def preprocess_ml1m(ratings_path, users_path, movies_path):
    # 加载数据
    ratings = pd.read_csv(ratings_path, sep='::', header=None, 
                          names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
    users = pd.read_csv(users_path, sep='::', header=None, 
                        names=['user_id', 'gender', 'age', 'occupation', 'zip'], engine='python')
    movies = pd.read_csv(movies_path, sep='::', header=None, 
                         names=['movie_id', 'title', 'genres'], engine='python', encoding='ISO-8859-1')

    # 数据清洗
    ratings = ratings.drop(columns=['timestamp'])
    users = users.drop(columns=['zip'])
    
    # 特征工程
    users['gender'] = users['gender'].map({'M': 0, 'F': 1})
    movies['genres'] = movies['genres'].str.split('|')
    genres_encoded = movies['genres'].str.join('|').str.get_dummies('|')
    movies = pd.concat([movies[['movie_id']], genres_encoded], axis=1)
    
    ratings['label'] = (ratings['rating'] >= 4).astype(int)
    ratings = ratings.drop(columns=['rating'])
    
    data = ratings.merge(users, on='user_id').merge(movies, on='movie_id')
    
    ratings_with_timestamp = pd.read_csv(ratings_path, sep='::', header=None, 
                                         names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
    data = data.merge(ratings_with_timestamp[['user_id', 'movie_id', 'timestamp']], 
                      on=['user_id', 'movie_id'])
    
    return data

def split_and_save_by_user(data, output_base_dir):
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    user_ids = data['user_id'].unique()
    all_train_data = []
    all_valid_data = []
    all_test_data = []
    single_class_users = []  # 记录单类用户
    
    for user_id in user_ids:
        user_data = data[data['user_id'] == user_id].copy()
        user_data = user_data.sort_values('timestamp')
        
        total_samples = len(user_data)
        train_end = int(total_samples * 0.6)
        valid_end = train_end + int(total_samples * 0.2)
        
        train_data = user_data.iloc[:train_end]
        valid_data = user_data.iloc[train_end:valid_end]
        test_data = user_data.iloc[valid_end:]
        
        all_movie_ids = data['movie_id'].unique()

        # 处理验证集单类
        if len(valid_data['label'].unique()) == 1 and len(train_data) > 0:
            missing_label = 1 - valid_data['label'].iloc[0]
            opposite_label_data = train_data[train_data['label'] == missing_label]
            if len(opposite_label_data) > 0:
                sample_to_move = opposite_label_data.sample(n=1)
                train_data = train_data.drop(sample_to_move.index)
                valid_data = pd.concat([valid_data, sample_to_move])
            elif missing_label == 0 and len(valid_data) > 0 and valid_data['label'].iloc[0] == 1:
                user_interacted_movies = user_data['movie_id'].unique()
                uninteracted_movies = set(all_movie_ids) - set(user_interacted_movies)
                if uninteracted_movies:
                    neg_movie_id = np.random.choice(list(uninteracted_movies))
                    neg_sample = valid_data.iloc[0:1].copy()
                    neg_sample['movie_id'] = neg_movie_id
                    neg_sample['label'] = 0
                    movie_features = data[data['movie_id'] == neg_movie_id].iloc[0]
                    for col in movie_features.index:
                        if col not in ['user_id', 'movie_id', 'label', 'timestamp', 'gender', 'age', 'occupation']:
                            neg_sample[col] = movie_features[col]
                    neg_sample['timestamp'] = valid_data['timestamp'].max()
                    valid_data = pd.concat([valid_data, neg_sample])

        # 处理测试集单类
        if len(test_data['label'].unique()) == 1 and len(train_data) > 0:
            missing_label = 1 - test_data['label'].iloc[0]
            opposite_label_data = train_data[train_data['label'] == missing_label]
            if len(opposite_label_data) > 0:
                sample_to_move = opposite_label_data.sample(n=1)
                train_data = train_data.drop(sample_to_move.index)
                test_data = pd.concat([test_data, sample_to_move])
            elif missing_label == 0 and len(test_data) > 0 and test_data['label'].iloc[0] == 1:
                user_interacted_movies = user_data['movie_id'].unique()
                uninteracted_movies = set(all_movie_ids) - set(user_interacted_movies)
                if uninteracted_movies:
                    neg_movie_id = np.random.choice(list(uninteracted_movies))
                    neg_sample = test_data.iloc[0:1].copy()
                    neg_sample['movie_id'] = neg_movie_id
                    neg_sample['label'] = 0
                    movie_features = data[data['movie_id'] == neg_movie_id].iloc[0]
                    for col in movie_features.index:
                        if col not in ['user_id', 'movie_id', 'label', 'timestamp', 'gender', 'age', 'occupation']:
                            neg_sample[col] = movie_features[col]
                    neg_sample['timestamp'] = test_data['timestamp'].max()
                    test_data = pd.concat([test_data, neg_sample])

        # 检查是否仍为单类
        if len(valid_data['label'].unique()) == 1 or len(test_data['label'].unique()) == 1:
            single_class_users.append(user_id)
            print(f"User {user_id} still has single class after processing: "
                  f"Valid labels={valid_data['label'].unique()}, "
                  f"Test labels={test_data['label'].unique()}")
            continue  # 跳过该用户，不加入最终数据集

        # 保存用户数据
        all_train_data.append(train_data)
        all_valid_data.append(valid_data)
        all_test_data.append(test_data)

        user_dir = os.path.join(output_base_dir, f'user_{user_id}')
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        
        train_data.to_csv(os.path.join(user_dir, 'train.csv'), index=False)
        valid_data.to_csv(os.path.join(user_dir, 'valid.csv'), index=False)
        test_data.to_csv(os.path.join(user_dir, 'test.csv'), index=False)
        
        print(f"User {user_id}: Train={len(train_data)}, Valid={len(valid_data)}, Test={len(test_data)}")

    # 输出单类用户
    if single_class_users:
        print(f"\nFollowing users were removed due to persistent single-class issue: {single_class_users}")
    else:
        print("\nNo users were removed due to single-class issues.")

    # 合并数据
    combined_train_data = pd.concat(all_train_data, ignore_index=True)
    combined_valid_data = pd.concat(all_valid_data, ignore_index=True)
    combined_test_data = pd.concat(all_test_data, ignore_index=True)
    
    combined_train_output_path = os.path.join(output_base_dir, 'all_users_train.csv')
    combined_valid_output_path = os.path.join(output_base_dir, 'all_users_valid.csv')
    combined_test_output_path = os.path.join(output_base_dir, 'all_users_test.csv')

    combined_train_data.to_csv(combined_train_output_path, index=False)
    combined_valid_data.to_csv(combined_valid_output_path, index=False)
    combined_test_data.to_csv(combined_test_output_path, index=False)
    
    print(f"Combined train data saved to {combined_train_output_path} with {len(combined_train_data)} samples")
    print(f"Combined valid data saved to {combined_valid_output_path} with {len(combined_valid_data)} samples")
    print(f"Combined test data saved to {combined_test_output_path} with {len(combined_test_data)} samples")

    all_users_data = pd.concat([combined_train_data, combined_valid_data, combined_test_data], ignore_index=True)
    all_users_data_output_path = os.path.join(output_base_dir, 'all_users_data.csv')
    all_users_data.to_csv(all_users_data_output_path, index=False)
    print(f"合并后的数据已保存到 {all_users_data_output_path}, 共 {len(all_users_data)} 条记录")

# 文件路径
ratings_path = '/home/chao/workspace/DeviceRec/data/ml-1m/ratings.dat'
users_path = '/home/chao/workspace/DeviceRec/data/ml-1m/users.dat'
movies_path = '/home/chao/workspace/DeviceRec/data/ml-1m/movies.dat'

# 执行
data = preprocess_ml1m(ratings_path, users_path, movies_path)
output_base_dir = '/home/chao/workspace/DeviceRec-simple/data'
split_and_save_by_user(data, output_base_dir)