# DeviceRec

## 数据预处理
**movielens-1m基本特征**
movielens-1m数据集包含3个文件（总共包含6040个用户和3952个电影）
- 用户评分表 ratings.dat，格式为`UserID::MovieID::Rating::Timestamp`
- 用户信息表 users.dat，格式为`UserID::Gender::Age::Occupation::Zip-code`
- 电影信息表 movies.dat，格式为`MovieID::Title::Genres`

**合并数据集**
- 删除字段：ratings.dat中的`timestamp`字段和users.dat中的`zip`字段
- 特征工程：users.dat中`gender`字段二值化，ratings.dat中`rating`字段转换为二分类标签（>=4为1，<4为0），movies.dat中`genres`字段转换为one-hot编码
- 记录格式为`user_id, movie_id, label, user_info, movie_info, timestamp`
- 最终一条记录格式为`user_id,movie_id,label,(gender,age,occupation),(Action,Adventure,Animation,Children's,Comedy,Crime,Documentary,Drama,Fantasy,Film-Noir,Horror,Musical,Mystery,Romance,Sci-Fi,Thriller,War,Western),timestamp`

**划分训练集，验证集和测试集**
每个用户创建一个自己的目录，并按照6:2:2划分训练集，验证集和测试集

**筛选无效用户**
为了避免计算AUC时出现单类用户（即一个用户的标签全为0/1），检查用户的验证集和测试集中是否只包含单一标签，对单类用户处理如下
- 从训练集移动不同标签样本到单类集合
- 若训练集也不存在，为只有正样本的用户伪造负样本（选择用户未交互过的电影）
- 最终，删除验证集/测试集中仍然为单类的用户，筛选后的用户数量为6037（删除3个用户：3598, 4486, 5850）

## 模型选取
目前选择了三个比较基础的模型：LR，WideDeep和PNN
- LR：拼接所有的特征，然后输入一个两层MLP
- WideDeep：Wide部分为用户和电影的one-hot编码+用户属性+电影属性，Deep部分为用户和电影的嵌入向量+用户属性+电影属性。最后Wide和Deep部分分别输入各自的MLP，最后输出相加
- PNN：线性部分为拼接所有特征，内积交互部分将用户，电影，职业，性别四个嵌入做内积。拼接线性和内积交互部分输入MLP。

## 具体实现
### 训练全局云端模型Cloud
参数设置如下：
- batch_size = 32   # 全体训练集的batch
- lr = 0.0001       # 模型学习率
- patience = 5      # 早停的耐心值

在全体训练集上训练，并且以全体验证集上的AUC指标作为早停指标，最终训练好的全局模型参数保存至`/cloud_model`

最终得到的每个云端模型的测试集上用户平均AUC指标如下：
- LR = 0.6909240355875058
- WideDeep = 0.6895638356875761
- PNN = 0.6858376096914948

### 训练本地微调模型Local
参数设置如下：
- batch_size = 32   # 本地训练集的batch
- lr = 0.0001       # 模型学习率
- patience = 5      # 早停的耐心值

每个用户加载全局云端模型作为初始模型，在本地训练集上训练，并以本地验证集的AUC作为早停指标
最终得到的本地微调模型在测试集上用户平均AUC指标如下：
- LR = 0.6909866975949882
- WideDeep = 0.6898554805778412
- PNN = 0.6871874010988587

### MPDA
原始MPDA的做法如下：
- 使用全局云端模型作为初始模型
- 随机匹配k个外部用户
- 遍历外部用户，若在外边用户的训练集上训练后，在本地训练集的auc指标提升则保留，否则回退模型
- 在本地训练集训练一个epoch

我的改进的MPDA的做法如下：
- 使用全局云端模型作为初始模型
- 在本地训练集训练，并以本地验证集的AUC作为早停指标
- 根据用户嵌入，使用余弦相似度给用户匹配k个外部用户
- 将所有外部用户的训练集合并
- 在合并的外边用户的训练集上训练后，在本地验证集的auc指标提升则保留，否则回退模型

主要改进为：
- 微调顺序从原来的`外部-本地`改为`本地-外部`
- 原来选取外部数据为用户级别，现在为数据级别，即直接选择所有外部用户的数据或不选，降低实验时间
- 训练不是固定为一个epoch，而是改为验证集上早停

各个模型的参数设置如下：
| Model    | Recall_num | Local_lr | External_lr | Local_batch_size | External_bach_size |
| -------- | ---------- | -------- | ----------- | ---------------- | ------------------ |
| LR       | 200        | 1e-3     | 5e-5        | 16               | 256                |
| WideDeep | 20         | 5e-4     | 5e-5        | 128              | 64                 |
| PNN      | 150        | 1e-4     | 5e-5        | 256              | 256                |


各个模型的结果如下：
| model    | Valid              | Test               | improved_users_num |
| -------- | ------------------ | ------------------ | ------------------ |
| LR       | 0.6977626149353882 | 0.6912387526231878 | 868                |
| WideDeep | 0.6983983826144139 | 0.6905310503034009 | 1070               |
| PNN      | 0.6991687780992718 | 0.6873370908092477 | 1476               |

### MPDA_feedback
基于反馈式的用户数据增强的端云协同推荐范式

云端维持一个全局的用户特征向量表（初始化为模型嵌入）

端侧算法伪代码如下：
```txt
on_device_exp:

for epoch in range(epochs):
    # 根据云端用户嵌入向量给用户匹配k个相似用户
    external_users = match(cloud_user_embeddings)
    # 合并所有外部数据
    external_data = merge(external_users)
    
    # 本地微调
    if epoch == 0:
        model = cloud_model  # 云端初始模型
        train(model, local_train_data)
        save(model, user_model_path)  # 保存本地微调后的模型
    else:
        model = load(user_model_path)

    # 计算初始验证集AUC
    init_valid_auc = calc_auc(model, local_valid_data)

    # 外部数据微调
    train(model, external_data)

    # 外部数据微调后计算验证集auc
    current_valid_auc = calc_auc(model, local_valid_data)

    delta_auc = current_valid_auc - init_valid_auc
    if (delta_auc > 1e-6):
        feedback = 1
        # 保存模型
        save(model, user_model_path)
    else:
        feedback = 0

    # 反馈结果给云端
    return feedback, external_users
```

云端伪代码：
```txt
on_cloud_exp:

# 初始化全局用户特征向量为模型用户嵌入
cloud_user_embeddings = model.user_embedding.weight

for epoch in range(epochs):
    feedbacks = []

    # 收集用户反馈
    for user in users:
        # 根据云端用户嵌入向量给用户匹配k个相似用户
        external_users = match(cloud_user_embeddings)

        feedback, external_users = on_device_exp(user, external_users)
        feedbacks[user] = {feedback, external_users}
    
    # 更新全局用户特征向量
    for user, (feedback, external_users) in feedbacks:
        # 拉近嵌入
        if feedbakc == 1:
            cloud_user_embeddings[user] += alpha * (cloud_user_embeddings[user] - cloud_user_embeddings[external_users]).mean()
        # 拉远嵌入
        else if feedback == 0:
            cloud_user_embeddings[user] -= beta * (cloud_user_embeddings[user] - cloud_user_embeddings[external_users]).mean()

        # 归一化嵌入
        cloud_user_embeddings[user] /= norm(cloud_user_embeddings[user])

    # alpha和beta衰减
    alpha *= decay_rate
    beta *= decay_rate
```
