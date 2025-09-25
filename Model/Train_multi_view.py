import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 导入多视图检索增强模型
from Multi_view import MultiViewRetrievalModel

# 设置设备
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

def create_sliding_windows(data, lookback=21, forecast_steps=7, start_index=20, end_index=90):
    """
    创建滑动窗口数据集（限定在指定时间范围内）
    
    Args:
        data: (B, L) 时间序列数据（L=97）
        lookback: 历史窗口大小（默认21天）
        forecast_steps: 预测步长（默认7天）
        start_index: 开始索引（默认20）
        end_index: 结束索引（默认90）
        
    Returns:
        X: (num_windows, lookback) 历史窗口数据
        y: (num_windows, forecast_steps) 未来预测目标
    """
    # 实际上是20-89，划分出70份
    B, L = data.shape
    X_list, y_list = [], []
    
    for i in range(B):
        series = data[i]
        # 
        effective_end = end_index 
        
        for t in range(start_index, effective_end):

            # 历史窗口：从t-20到t（共21天）
            X = series[t-lookback+1:t+1]
            # 未来窗口：从t+1到t+7（共7天）
            y = series[t+1:t+1+forecast_steps]
            X_list.append(X)
            y_list.append(y)
    
    return np.array(X_list), np.array(y_list)

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """创建PyTorch DataLoader"""
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_one_epoch(model, train_loader, optimizer, criterion):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
    
    return total_loss / len(train_loader.dataset)

def eval_model(model, data_loader, criterion):
    """评估模型"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
    
    return total_loss / len(data_loader.dataset)

def predict(model, data_loader):
    """生成预测"""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(DEVICE)
            outputs = model(batch_x)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_targets)

def RMSE_MAE_MAPE(preds, targets):
    """计算评估指标"""
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    mape = np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100  # 避免除零
    return rmse, mae, mape

def print_log(*args, log=None):
    """打印日志"""
    message = ' '.join(str(arg) for arg in args)
    print(message)
    if log:
        log.write(message + '\n')

def extract_multi_view_features(data, lookback=21):
    """
    从时间序列数据中提取多视图特征
    
    Args:
        data: (B, L) 时间序列数据
        lookback: 回溯窗口大小
        
    Returns:
        multi_view_features: (B, C, D) 多视图特征
    """
    B, L = data.shape
    C = 4  # 假设4个视图
    D = 64  # 假设特征维度为64
    
    # 这里使用简单的统计特征作为多视图特征
    multi_view_features = np.zeros((B, C, D))
    
    for i in range(B):
        series = data[i]
        
        # 视图1: 统计特征
        stats_features = np.zeros(D)
        stats_features[0] = np.mean(series)
        stats_features[1] = np.std(series)
        stats_features[2] = np.max(series) - np.min(series)
        stats_features[3] = np.median(series)
        # 可以添加更多统计特征...
        multi_view_features[i, 0, :len(stats_features)] = stats_features
        
        # 视图2: 最近lookback天的特征
        recent_features = series[-lookback:] if L >= lookback else series
        multi_view_features[i, 1, :len(recent_features)] = recent_features
        
        # 视图3: 傅里叶变换特征
        if L >= 10:
            fft_features = np.abs(np.fft.fft(series)[:D//2])
            multi_view_features[i, 2, :len(fft_features)] = fft_features
        
        # 视图4: 差分特征
        diff_features = np.diff(series)
        multi_view_features[i, 3, :len(diff_features)] = diff_features
    
    return multi_view_features
def extract_multi_view_features_from_windows(window_data, original_data_shape):
    """
    从滑动窗口数据中提取多视图特征
    
    Args:
        window_data: (num_windows, lookback) 滑动窗口数据
        original_data_shape: (B, L) 原始数据形状
        
    Returns:
        multi_view_features: (num_windows, C, D) 多视图特征
    """
    num_windows, lookback = window_data.shape
    C = 4  # 假设4个视图
    D = 64  # 假设特征维度为64
    
    multi_view_features = np.zeros((num_windows, C, D))
    
    for i in range(num_windows):
        window = window_data[i]
        
        # 视图1: 统计特征
        stats_features = np.zeros(D)
        stats_features[0] = np.mean(window)
        stats_features[1] = np.std(window)
        stats_features[2] = np.max(window) - np.min(window)
        stats_features[3] = np.median(window)
        stats_features[4] = np.sum(np.diff(window) > 0) / (len(window) - 1)  # 上升趋势比例
        multi_view_features[i, 0, :len(stats_features)] = stats_features
        
        # 视图2: 整个窗口特征
        multi_view_features[i, 1, :lookback] = window
        
        # 视图3: 傅里叶变换特征
        if lookback >= 10:
            fft_features = np.abs(np.fft.fft(window))[:D//2]
            multi_view_features[i, 2, :len(fft_features)] = fft_features
        
        # 视图4: 差分和变化率特征
        diff_features = np.diff(window)
        multi_view_features[i, 3, :len(diff_features)] = diff_features
    
    return multi_view_features

def enhance_window_features_with_multi_view(original_data, window_data, retrieval_db, retrieval_sales, model, forecast_steps = 7):
    """
    为滑动窗口数据使用多视图检索增强
    
    Args:
        window_data: (num_windows, lookback) 滑动窗口数据
        retrieval_db: (N, L) 检索数据库（原始数据）
        retrieval_sales: (N, 7) 检索数据库的销量特征
        model: 多视图检索增强模型
        
    Returns:
        enhanced_windows: (num_windows, lookback, 8) 增强后的窗口特征
    """
    B, L = original_data.shape
    num_windows, lookback = window_data.shape
    N, _ = retrieval_db.shape
    
    # 提取滑动窗口的多视图特征
    window_mv_features = extract_multi_view_features_from_windows(window_data, (num_windows, lookback))
    
    # 提取检索数据库的多视图特征
    retrieval_mv_features = extract_multi_view_features_from_windows(retrieval_db, (num_windows, lookback))

    
    # 使用多视图检索模型进行增强
    with torch.no_grad():
        predicted_sales, aggregated_features = model.predict(
            torch.FloatTensor(window_mv_features).to(DEVICE),
            torch.FloatTensor(retrieval_mv_features).to(DEVICE),
            torch.FloatTensor(retrieval_sales).to(DEVICE)
        )
    predicted_sales = predicted_sales.reshape(B, -1, forecast_steps) 


    # 构建增强特征 (B, L, 8)
    enhanced_data = np.zeros((B,L-forecast_steps, 8)) # 最后预测的几天是y值，不特征增强了
    
    for i in range(B):
        for t in range(L-forecast_steps):
            enhanced_data[i, t, 0] = original_data[i, t]  # 原始值
            
            if t < 20:  # 前20天使用未来真实值填充
                # 获取未来1-7天的真实值
                for step in range(1, 8):
                    if t + step < L:
                        enhanced_data[i, t, step] = original_data[i, t + step]
                    else:
                        enhanced_data[i, t, step] = 0  # 超出序列长度补零
            elif t < L-forecast_steps:  # 20天后使用模型预测值
                enhanced_data[i, t, 1:8] = predicted_sales[i,t-20,:].cpu().numpy()
    return enhanced_data

def train_multi_view_model_on_windows(train_windows, train_sales, val_windows, val_sales):
    """
    在滑动窗口数据上训练多视图检索增强模型
    
    Args:
        train_windows: (num_train_windows, lookback) 训练窗口数据
        train_sales: (num_train_windows, 7) 训练窗口对应的销量标签
        val_windows: (num_val_windows, lookback) 验证窗口数据
        val_sales: (num_val_windows, 7) 验证窗口对应的销量标签
        
    Returns:
        model: 训练好的多视图检索增强模型
    """
    num_train, lookback = train_windows.shape
    num_val, _ = val_windows.shape
    
    # 提取多视图特征
    train_mv_features = extract_multi_view_features_from_windows(train_windows, (num_train, lookback))
    val_mv_features = extract_multi_view_features_from_windows(val_windows, (num_val, lookback))
    
    # 初始化模型
    model = MultiViewRetrievalModel(
        input_dim=64,
        hidden_dim=64,
        output_dim=64,
        num_views=4,
        k1=5,
        k2=3,
        sales_feature_dim=7,
        dropout_rate=0.2
    ).to(DEVICE)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 转换为Tensor
    train_features = torch.FloatTensor(train_mv_features).to(DEVICE)
    train_sales_tensor = torch.FloatTensor(train_sales).to(DEVICE)
    val_features = torch.FloatTensor(val_mv_features).to(DEVICE)
    val_sales_tensor = torch.FloatTensor(val_sales).to(DEVICE)
    
    # 训练循环
    print("在滑动窗口数据上训练多视图检索增强模型...")
    best_val_loss = float('inf')
    patience = 10
    wait = 0
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        pred_sales, _ = model(train_features, train_features, train_sales_tensor)
        
        # 计算损失
        loss = criterion(pred_sales, train_sales_tensor)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(val_features, train_features, train_sales_tensor)
            val_loss = criterion(val_pred, val_sales_tensor)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_state_dict = model.state_dict().copy()
        else:
            wait += 1
            if wait >= patience:
                print(f"早停触发于第 {epoch+1} 轮")
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    # 加载最佳模型
    model.load_state_dict(best_state_dict)
    print("多视图检索增强模型训练完成!")
    
    return model

def main_enhancement_pipeline_with_windows():
    """主增强流程（处理滑动窗口数据）"""
    # 创建示例数据
    np.random.seed(42)
    B_train, B_val, B_test = 60, 20, 20
    L = 97
    lookback = 21
    forecast_steps = 7
    
    # 原始数据
    train_data = np.random.randn(B_train, L)
    val_data = np.random.randn(B_val, L)
    test_data = np.random.randn(B_test, L)
    
    # 创建滑动窗口
    X_train, y_train = create_sliding_windows(train_data, lookback, forecast_steps, 20)
    X_val, y_val = create_sliding_windows(val_data, lookback, forecast_steps, 20)
    X_test, y_test = create_sliding_windows(test_data, lookback, forecast_steps, 20)
    
    print("滑动窗口数据形状:")
    print(f"训练窗口: {X_train.shape} -> 标签: {y_train.shape}")
    print(f"验证窗口: {X_val.shape} -> 标签: {y_val.shape}")
    print(f"测试窗口: {X_test.shape} -> 标签: {y_test.shape}")
    
    # 训练多视图检索增强模型（在训练窗口上）
    multi_view_model = train_multi_view_model_on_windows(X_train, y_train, X_val, y_val)
    
    # 为训练窗口生成增强特征（在训练窗口中检索）
    print("\n为训练窗口进行多视图检索增强...")
    enhanced_train = enhance_window_features_with_multi_view(
        train_data, X_train, X_train, y_train, multi_view_model
    )
    
    # 为验证窗口生成增强特征（在训练窗口中检索）
    print("为验证窗口进行多视图检索增强...")
    enhanced_val = enhance_window_features_with_multi_view(
        val_data, X_val, X_train, y_train, multi_view_model
    )
    
    # 为测试窗口生成增强特征（在训练+验证窗口中检索）
    print("为测试窗口进行多视图检索增强...")
    combined_db = np.concatenate([X_train, X_val], axis=0)
    combined_sales = np.concatenate([y_train, y_val], axis=0)
    enhanced_test = enhance_window_features_with_multi_view(
        test_data, X_test, combined_db, combined_sales, multi_view_model
    )
    
    # 输出结果
    print(f"\n增强后窗口特征形状:")
    print(f"训练窗口: {enhanced_train.shape}")
    print(f"验证窗口: {enhanced_val.shape}")
    print(f"测试窗口: {enhanced_test.shape}")
    
    # 验证特征增强效果
    print(f"\n示例增强窗口（第一个窗口的前5个时间点）:")
    for t in range(5):
        features = enhanced_train[0, t]
        print(f"时间点 {t}: 当前值={features[0]:6.2f}, 未来预测={features[1:].round(2)}")
    
    return enhanced_train, enhanced_val, enhanced_test, multi_view_model


if __name__ == "__main__":
    enhanced_train, enhanced_val, enhanced_test, multi_view_model = main_enhancement_pipeline_with_windows()
    
    # 保存增强后的数据
    np.save('enhanced_train_mv.npy', enhanced_train)
    np.save('enhanced_val_mv.npy', enhanced_val)
    np.save('enhanced_test_mv.npy', enhanced_test)
    
    print("\n增强数据已保存为:")
    print("enhanced_train_mv.npy")
    print("enhanced_val_mv.npy")
    print("enhanced_test_mv.npy")