import torch
import torch.nn as nn

class PNNModel(nn.Module):
    def __init__(self, num_users=6041, num_movies=3953, num_occupations=21, num_genres=18, embedding_dim=32, hidden_dims=[64, 32]):
        super(PNNModel, self).__init__()
        
        # 嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.occupation_embedding = nn.Embedding(num_occupations, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim)
        
        self.embedding_dim = embedding_dim
        self.num_genres = num_genres
        
        # 输入维度
        self.num_emb_fields = 4  # user, movie, occupation, genres
        self.input_dim_linear = (embedding_dim * 4) + 2  # 4 个嵌入 + gender, age
        self.num_product_terms = (self.num_emb_fields * (self.num_emb_fields - 1)) // 2  # C(4, 2) = 6
        self.input_dim = self.input_dim_linear + self.num_product_terms
        
        # MLP
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, user_ids, movie_ids, gender, age, occupation, genres):
        user_emb = self.user_embedding(user_ids)                    # [batch_size, embedding_dim]
        movie_emb = self.movie_embedding(movie_ids)                 # [batch_size, embedding_dim]
        occupation_emb = self.occupation_embedding(occupation.long())  # [batch_size, embedding_dim]
        genre_embs = self.genre_embedding(torch.arange(self.num_genres).to(genres.device))  # [num_genres, embedding_dim]
        genres_emb = torch.matmul(genres, genre_embs)              # [batch_size, embedding_dim]
        
        gender = gender.unsqueeze(1)  # [batch_size, 1]
        age = age.unsqueeze(1)        # [batch_size, 1]
        
        # 线性特征拼接
        linear_features = torch.cat([user_emb, movie_emb, occupation_emb, genres_emb, gender, age], dim=1)  # [batch_size, input_dim_linear]
        
        # 内积交互
        vector_features = [user_emb, movie_emb, occupation_emb, genres_emb]
        product_terms = [torch.sum(vector_features[i] * vector_features[j], dim=1, keepdim=True)
                         for i in range(len(vector_features)) for j in range(i + 1, len(vector_features))]
        product_features = torch.cat(product_terms, dim=1)  # [batch_size, num_product_terms]
        
        # 拼接所有特征
        features = torch.cat([linear_features, product_features], dim=1)  # [batch_size, input_dim]
        
        # MLP 输出
        logits = self.mlp(features)  # [batch_size, 1]
        return self.sigmoid(logits).squeeze(1)  # [batch_size]