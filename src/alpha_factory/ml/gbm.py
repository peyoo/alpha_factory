from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb


def fit_with_tscv(df, features, label_col):
    # 二分类任务参数示例
    params = {
        # 1. 任务性质
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": 42,
        # 2. 树结构控制 (防止过拟合的核心)
        "num_leaves": 31,  # 核心参数。金融数据建议设小点 (15-63)，防止树太深捕捉噪声
        "max_depth": -1,  # 通常由 num_leaves 控制即可
        "min_data_in_leaf": 100,  # 重要！每个叶子节点最少样本数。越大越能防止过拟合
        "max_bin": 255,  # 特征分箱数，默认 255 足够
        # 3. 学习速率
        "learning_rate": 0.01,  # 建议设小 (0.01 - 0.05)，配合较大的 num_boost_round
        # 4. 随机采样 (引入扰动，增加鲁棒性)
        "feature_fraction": 0.8,  # 每次建树随机选 80% 的因子。防止模型过度依赖某几个强因子
        "bagging_fraction": 0.8,  # 每次迭代随机选 80% 的数据
        "bagging_freq": 5,  # 每 5 次迭代执行一次 bagging
        # 5. 正则化 (量化模型的“救命稻草”)
        "lambda_l1": 0.1,  # L1 正则，能让一些不重要因子的权重归零
        "lambda_l2": 0.1,  # L2 正则，防止权重过大
        "min_gain_to_split": 0.01,  # 分裂需要的最小增益，剔除微弱信号
        # 6. 其他
        "n_jobs": -1,  # 使用全部 CPU 核心
    }

    # 1. 定义时间序列切分器
    # n_splits: 切成几段
    # gap: 训练集和验证集之间的间隔（防止由于 Label 跨期导致的数据泄露）
    # max_train_size: 设为 None 则是扩张窗口，设为具体数值则是滚动窗口
    tscv = TimeSeriesSplit(n_splits=5, gap=5)

    models = []
    X = df[features]
    y = df[label_col]

    # 2. 遍历切分出来的索引
    for i, (train_index, val_index) in enumerate(tscv.split(X)):
        logger.info(f"正在训练第 {i + 1} 个分段...")

        # 3. 按索引取数据
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_val, y_val = X.iloc[val_index], y.iloc[val_index]

        # 4. 原生 API 数据封装
        train_ds = lgb.Dataset(X_train, label=y_train)
        valid_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

        # 5. 原生训练
        model = lgb.train(
            params,
            train_set=train_ds,
            valid_sets=[train_ds, valid_ds],
            valid_names=["train", "valid"],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)],
        )
        models.append(model)

    return models
