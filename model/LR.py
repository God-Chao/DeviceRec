import torch
import torch.nn as nn

# 逻辑回归模型
class LRModel(nn.Module):
    def __init__(self, num_users=6041, num_movies=3953, embedding_dim=32, num_genres=18):
        super(LRModel, self).__init__()
        
        # 嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # 输入维度 = 用户嵌入 + 电影嵌入 + gender + age + occupation + genres
        input_dim = embedding_dim * 2 + 1 + 1 + 1 + num_genres
        
        # 线性层（逻辑回归）
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user_ids, movie_ids, gender, age, occupation, genres):
        # 获取嵌入
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        movie_emb = self.movie_embedding(movie_ids)  # [batch_size, embedding_dim]
        
        # 将标量特征转为二维张量
        gender = gender.unsqueeze(1)  # [batch_size, 1]
        age = age.unsqueeze(1)  # [batch_size, 1]
        occupation = occupation.unsqueeze(1)  # [batch_size, 1]
        
        # 拼接所有特征
        features = torch.cat([user_emb, movie_emb, gender, age, occupation, genres], dim=1)
        
        # 线性层输出
        logits = self.linear(features)  # [batch_size, 1]
        probs = self.sigmoid(logits)  # [batch_size, 1]
        
        return probs.squeeze(1)  # [batch_size]
