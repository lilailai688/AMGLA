from transformers import TimesFmModelForPrediction
import torch

# 加载模型（建议在函数外部加载一次）
model = TimesFmModelForPrediction.from_pretrained("/workspace2/gongyan/ts/timesfm-2.0-500m-pytorch")


def create_multi_scale_series(series, lookback_windows=[21, 14, 7], scales=[1, 2]):
    """
    创建多视图和多尺度的时间序列
    
    Args:
        series: torch.Tensor, 形状为 (B, L)
        lookback_windows: list, 回望窗口长度列表
        scales: list, 尺度列表
    
    Returns:
        list: 多尺度时间序列列表
    """
    B, L = series.shape
    multi_scale_series = []
    
    # 确保输入序列足够长
    max_window = max(lookback_windows)
    if L < max_window:
        raise ValueError(f"输入序列长度{L}小于最大回望窗口{max_window}")
    
    for window in lookback_windows:
        # 从最后一步取指定窗口长度的序列
        window_series = series[:, -window:]
        
        for scale in scales:
            if scale == 1:
                # 原始尺度
                scaled_series = window_series
            else:
                # 下采样到指定尺度
                scaled_length = window // scale
                scaled_series = torch.zeros(B, scaled_length)
                
                for i in range(B):
                    # 使用平均池化进行下采样
                    scaled_data = torch.nn.functional.avg_pool1d(
                        window_series[i].unsqueeze(0).unsqueeze(0), 
                        kernel_size=scale, 
                        stride=scale
                    )
                    scaled_series[i] = scaled_data.squeeze()
            
            multi_scale_series.append(scaled_series)
    
    return multi_scale_series


def get_timesfm_embeddings(past_values_list, frequency_input=None):
    """
    获取 TimesFM 模型的 embedding 表示
    
    Args:
        past_values_list: list of torch.Tensor, 输入时间序列列表
        frequency_input: torch.Tensor, 频率标识符，如果不提供则自动生成全0
    
    Returns:
        torch.Tensor: embedding 张量，形状为 (B, P, D)
    """
    # 如果没有提供 frequency_input，则自动生成全0
    # 0是天频率及以下的
    # 1是周，月
    # 2是季度，年
    if frequency_input is None:
        frequency_input = torch.zeros(len(past_values_list), dtype=torch.long)
    
    
    with torch.no_grad():
        outputs = model(
            past_values=past_values_list, 
            freq=frequency_input, 
            return_dict=True
        )
        embedding = outputs.last_hidden_state
    
    return embedding


def extract_multi_view_multi_scale_embeddings(series, lookback_windows=[21, 14, 7], scales=[1, 2]):
    """
    提取多视图多尺度的时间序列表示
    
    Args:
        series: torch.Tensor, 形状为 (B, L)
        lookback_windows: list, 回望窗口长度列表
        scales: list, 尺度列表
    
    Returns:
        torch.Tensor: 形状为 (B, num_views * num_scales, P, D)
    """
    # 创建多尺度序列
    multi_scale_series = create_multi_scale_series(series, lookback_windows, scales)
    
    # 获取所有尺度序列的embedding
    embeddings_list = []
    for scale_series in multi_scale_series:
        # 将每个尺度的序列转换为列表格式
        series_list = [scale_series[i] for i in range(scale_series.shape[0])]
        embedding = get_timesfm_embeddings(series_list)
        embeddings_list.append(embedding)
    
    # 将所有embedding堆叠起来
    # embeddings_list 中的每个元素形状为 (B, P, D)
    # 堆叠后形状为 (num_views * num_scales, B, P, D)
    all_embeddings = torch.stack(embeddings_list, dim=0)
    
    # 调整维度顺序为 (B, num_views * num_scales, P, D)
    all_embeddings = all_embeddings.permute(1, 0, 2, 3)
    
    # 调整维度 (B, num_views * num_scales, P*D)
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], all_embeddings.shape[1], -1)

    return all_embeddings



# 使用示例
if __name__ == "__main__":
    # 创建示例数据 - 假设我们有3个批次，每个批次有100个时间点
    batch_size = 3
    seq_length = 90
    series = torch.randn(batch_size, seq_length)
    
    # 提取多尺度嵌入
    embeddings = extract_multi_view_multi_scale_embeddings(
        series, 
        lookback_windows=[21, 14, 7], 
        scales=[1, 2]
    )
    
    print(f"输入序列形状: {series.shape}")
    print(f"多尺度嵌入形状: {embeddings.shape}")
    
    # 计算多尺度的数量
    num_scales = len([21, 14, 7]) * len([1, 2])
    print(f"多尺度数量: {num_scales}")
    
    # 验证形状
    # 假设 TimesFM 输出的 P=10, D=64
    expected_shape = (batch_size, num_scales, 10 * 64)
    print(f"期望形状: {expected_shape}")
    print(f"实际形状匹配: {embeddings.shape == expected_shape}")