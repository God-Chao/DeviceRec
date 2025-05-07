import torch
import torch.nn as nn

class WideDeepModel(nn.Module):
    def __init__(self, num_users=6041, num_movies=3953, embedding_dim=32, num_genres=18, hidden_units=[64, 32, 16]):
        super(WideDeepModel, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        wide_input_dim = num_users + num_movies + 1 + 1 + 1 + num_genres
        self.wide_linear = nn.Linear(wide_input_dim, 1, bias=True)

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Deep部分的输入维度：用户嵌入 + 电影嵌入 + gender + age + occupation + genres
        deep_input_dim = embedding_dim * 2 + 1 + 1 + 1 + num_genres
        
        # Deep部分的神经网络层
        layers = []
        prev_dim = deep_input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # 最后一层输出1维
        self.deep_net = nn.Sequential(*layers)
        
        # === 输出层 ===
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, movie_ids, gender, age, occupation, genres, timestamp=None):
        # === Wide 部分 ===
        # 将类别特征转为 one-hot 编码形式（这里假设输入是数值化的索引）
        user_onehot = nn.functional.one_hot(user_ids, num_classes=self.num_users).float()
        movie_onehot = nn.functional.one_hot(movie_ids, num_classes=self.num_movies).float()
        gender = gender.unsqueeze(1)  # [batch_size, 1]
        age = age.unsqueeze(1)  # [batch_size, 1]
        occupation = occupation.unsqueeze(1)  # [batch_size, 1]
        
        # Wide部分的输入拼接
        wide_input = torch.cat([user_onehot, movie_onehot, gender, age, occupation, genres], dim=1)
        wide_output = self.wide_linear(wide_input)  # [batch_size, 1]
        
        # === Deep 部分 ===
        # 获取嵌入
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        movie_emb = self.movie_embedding(movie_ids)  # [batch_size, embedding_dim]
        
        # Deep部分的输入拼接
        deep_input = torch.cat([user_emb, movie_emb, gender, age, occupation, genres], dim=1)
        deep_output = self.deep_net(deep_input)  # [batch_size, 1]
        
        # === 组合 Wide 和 Deep ===
        final_output = wide_output + deep_output  # 简单相加，也可以加权重
        probs = self.sigmoid(final_output)  # [batch_size, 1]
        
        return probs.squeeze(1)  # [batch_size]
