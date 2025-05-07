import torch
import pandas as pd
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data

        # 定义 genres 列名（从 CSV 的列中排除其他已知列）
        self.genre_columns = [col for col in data.columns if col not in 
                             ['user_id', 'movie_id', 'label', 'gender', 'age', 'occupation', 'timestamp']]
    
        # 提取所有特征并转为 PyTorch 张量
        self.user_ids = torch.tensor(self.data['user_id'].values, dtype=torch.long)
        self.movie_ids = torch.tensor(self.data['movie_id'].values, dtype=torch.long)
        self.gender = torch.tensor(self.data['gender'].values, dtype=torch.float32)
        self.age = torch.tensor(self.data['age'].values, dtype=torch.float32)
        self.occupation = torch.tensor(self.data['occupation'].values, dtype=torch.float32)
        self.genres = torch.tensor(self.data[self.genre_columns].values, dtype=torch.float32)
        self.labels = torch.tensor(self.data['label'].values, dtype=torch.float32)
        self.timestamps = torch.tensor(self.data['timestamp'].values, dtype=torch.long)  # 添加 timestamp
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],        # 用户 ID
            'movie_id': self.movie_ids[idx],      # 电影 ID
            'gender': self.gender[idx],           # 性别（0 或 1）
            'age': self.age[idx],                 # 年龄（分桶后的数值）
            'occupation': self.occupation[idx],   # 职业（0-20）
            'genres': self.genres[idx],           # 电影类型（独热编码向量）
            'label': self.labels[idx],            # 二值标签
            'timestamp': self.timestamps[idx]     # 时间戳
        }