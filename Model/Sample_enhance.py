'''
2025.09.11
lsh
内容：样本增强
'''
import numpy as np
from scipy.signal import correlate
from statsmodels.tsa.seasonal import STL
# from timegan import TimeGAN
import torch
import warnings
warnings.filterwarnings('ignore')

def extract_temporal_features(data):
    """
    使用时序大模型提取序列表征
    
    Args:
        data: (B, L) 时间序列数据
        
    Returns:
        features: (B, D) 序列表征特征
    """
    # 这里使用简单的统计特征作为示例，可以替换为更复杂的时序模型
    B, L = data.shape
    features = np.zeros((B, 4))  # 4个统计特征
    
    for i in range(B):
        series = data[i]
        features[i, 0] = np.mean(series)      # 均值
        features[i, 1] = np.std(series)       # 标准差
        features[i, 2] = np.max(series) - np.min(series)  # 极差
        features[i, 3] = np.median(series)    # 中位数
    
    return features

def find_similar_sequences(features, k=2):
    """
    找到最相似的k个序列
    
    Args:
        features: (B, D) 序列特征
        k: 相似序列数量
        
    Returns:
        similarity_matrix: (B, B) 相似度矩阵
        topk_indices: (B, k) 每个序列的最相似序列索引
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 计算余弦相似度
    similarity_matrix = cosine_similarity(features)
    
    # 排除自身相似度
    np.fill_diagonal(similarity_matrix, -np.inf)
    
    # 获取top-k相似序列
    topk_indices = np.argsort(similarity_matrix, axis=1)[:, -k:]
    
    return similarity_matrix, topk_indices

def stl_decomposition(series, period=7):
    """
    STL分解时间序列
    
    Args:
        series: (L,) 时间序列
        period: 季节周期
        
    Returns:
        trend: 趋势分量
        seasonal: 季节分量
        residual: 残差分量
    """
    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    
    return result.trend, result.seasonal, result.resid

def mixup(series1, series2):
    """
    趋势项融合 - MixUp方法
    
    Args:
        series1: 趋势分量1
        series2: 趋势分量2
        
    Returns:
        mixed_trend: 融合后的趋势
        series1: 原始趋势1
        series2: 原始趋势2
    """
    # 生成一个随机的lambda值
    lambda_value = np.random.uniform(0.3, 0.7)  # 限制在0.3-0.7之间避免过度混合
    
    # 创建 mixup 时间序列
    mixed_trend = lambda_value * series1 + (1 - lambda_value) * series2
    
    return mixed_trend, series1, series2

def phase_alignment(series1, series2):
    """
    季节项融合 - 相位对准
    
    Args:
        series1: 季节分量1
        series2: 季节分量2
        
    Returns:
        aligned_series2: 对齐后的季节分量2
        series1: 原始季节分量1
        series2: 原始季节分量2
    """
    # 计算互相关
    correlation = correlate(series1, series2, mode='full')
    lag = np.arange(-len(series1) + 1, len(series2))  # 计算延迟
    max_lag = lag[np.argmax(correlation)]  # 找到最大互相关的延迟
    
    # 对齐第二条时间序列
    if max_lag > 0:  # series2 在 series1 之后
        aligned_series2 = np.roll(series2, -max_lag)
        aligned_series2[-max_lag:] = np.nan  # 尾部填充 NaN
    elif max_lag < 0:  # series2 在 series1 之前
        aligned_series2 = np.roll(series2, -max_lag)
        aligned_series2[:abs(max_lag)] = np.nan  # 头部填充 NaN
    else:
        aligned_series2 = series2  # 如果没有延迟
    
    # 处理NaN值（使用线性插值）
    nan_mask = np.isnan(aligned_series2)
    if np.any(nan_mask):
        x = np.arange(len(aligned_series2))
        aligned_series2[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], aligned_series2[~nan_mask])
    
    return aligned_series2, series1, series2

def seasonal_fusion(seasonal1, seasonal2):
    """
    季节项融合
    
    Args:
        seasonal1: 季节分量1
        seasonal2: 季节分量2
        
    Returns:
        fused_seasonal: 融合后的季节分量
    """
    # 相位对齐
    aligned_seasonal2, _, _ = phase_alignment(seasonal1, seasonal2)
    
    # 加权平均融合
    fused_seasonal = 0.5 * seasonal1 + 0.5 * aligned_seasonal2
    
    return fused_seasonal

def residual_fusion(residual1, residual2, timegan_model=None):
    """
    残差项融合
    
    Args:
        residual1: 残差分量1
        residual2: 残差分量2
        timegan_model: TimeGAN模型（可选）
        
    Returns:
        fused_residual: 融合后的残差分量
    """
    if timegan_model is not None:
        # 使用TimeGAN生成新的残差序列
        residuals = np.stack([residual1, residual2], axis=0)
        residuals = residuals.reshape(2, len(residual1), 1)
        
        # 使用TimeGAN生成新序列
        generated_residuals = timegan_model.generate(residuals)
        fused_residual = generated_residuals[0].flatten()
    else:
        # 简单的随机加权融合
        alpha = np.random.uniform(0.4, 0.6)
        fused_residual = alpha * residual1 + (1 - alpha) * residual2
    
    return fused_residual

def reconstruct_series(trend, seasonal, residual):
    """
    从分解分量重建时间序列
    
    Args:
        trend: 趋势分量
        seasonal: 季节分量
        residual: 残差分量
        
    Returns:
        reconstructed: 重建的时间序列
    """
    return trend + seasonal + residual

def augment_sample(original_series, similar_series):
    """
    增强单个样本
    
    Args:
        original_series: (L, 8) 原始序列
        similar_series: (L, 8) 相似序列
        
    Returns:
        augmented_series: (L, 8) 增强后的序列
    """
    L, num_features = original_series.shape
    augmented_series = np.zeros((L, num_features))
    
    for feature_idx in range(num_features):
        # 对每个特征维度分别进行STL分解和融合
        orig_feature = original_series[:, feature_idx]
        similar_feature = similar_series[:, feature_idx]
        
        # STL分解
        trend1, seasonal1, residual1 = stl_decomposition(orig_feature, period=7)
        trend2, seasonal2, residual2 = stl_decomposition(similar_feature, period=7)
        
        # 趋势项融合
        fused_trend, _, _ = mixup(trend1, trend2)
        
        # 季节项融合
        fused_seasonal = seasonal_fusion(seasonal1, seasonal2)
        
        # 残差项融合
        fused_residual = residual_fusion(residual1, residual2)
        
        # 重建时间序列
        augmented_feature = reconstruct_series(fused_trend, fused_seasonal, fused_residual)
        augmented_series[:, feature_idx] = augmented_feature
    
    return augmented_series

def sample_augmentation(enhanced_data, k=2):
    """
    样本增强主函数
    
    Args:
        enhanced_data: (B, L, 8) 增强后的特征数据
        k: 每个样本找k个相似样本
        augmentation_factor: 增强倍数
        
    Returns:
        augmented_data: (B_augmented, L, 8) 增强后的数据
    """
    B, L, num_features = enhanced_data.shape
    
    # 提取第0维特征用于相似度计算
    base_features = enhanced_data[:, :, 0]  # (B, L)
    
    # 提取时序表征
    temporal_features = extract_temporal_features(base_features)  # (B, D)
    
    # 找到相似序列
    _, similar_indices = find_similar_sequences(temporal_features, k=k)
    
    # 生成增强样本
    augmented_samples = []
    
    for i in range(B):
        original_sample = enhanced_data[i]  # (L, 8)
        
        # 为每个原始样本生成k个增强样本
        for aug_idx in range(k):
            # 选择一个相似样本
            similar_idx = similar_indices[i][aug_idx]
            similar_sample = enhanced_data[similar_idx]  # (L, 8)
            
            # 增强样本
            augmented_sample = augment_sample(original_sample, similar_sample)
            augmented_samples.append(augmented_sample)
    
    # 合并原始数据和增强数据
    augmented_data = np.concatenate([enhanced_data, np.array(augmented_samples)], axis=0)
    
    print(f"样本增强完成: {B} -> {len(augmented_data)} 个样本")
    
    return augmented_data

# 示例使用
if __name__ == "__main__":
    # 假设已经得到了增强后的特征数据
    B_train, L, num_features = 100, 90, 8
    enhanced_train_data = np.random.randn(B_train, L, num_features)
    
    print(f"原始训练数据形状: {enhanced_train_data.shape}")
    
    # 进行样本增强
    augmented_train_data = sample_augmentation(
        enhanced_train_data, 
        k=2, 
    )
    
    print(f"增强后训练数据形状: {augmented_train_data.shape}")
    
    # 验证增强效果
    print("\n增强样本示例:")
    print("原始样本第一个特征:", enhanced_train_data[0, :10, 0])
    print("增强样本第一个特征:", augmented_train_data[B_train, :10, 0])