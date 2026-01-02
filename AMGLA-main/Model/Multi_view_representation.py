from transformers import TimesFmModelForPrediction
import torch

# Load once (recommended) to avoid repeated initialization cost
model = TimesFmModelForPrediction.from_pretrained(
    "/workspace2/gongyan/ts/timesfm-2.0-500m-pytorch"
)


def create_multi_scale_series(series, lookback_windows=[21, 14, 7], scales=[1, 2]):
    """
    Generate multi-window, multi-scale series views.

    Args:
        series: (B, L)
        lookback_windows: list of window lengths
        scales: downsampling factors

    Returns:
        list[Tensor]: each tensor is (B, window//scale)
    """
    B, L = series.shape
    max_window = max(lookback_windows)
    if L < max_window:
        raise ValueError(f"Sequence length {L} < max lookback window {max_window}")

    multi_scale_series = []

    for window in lookback_windows:
        window_series = series[:, -window:]  # (B, window)

        for scale in scales:
            if scale == 1:
                scaled_series = window_series
            else:
                # Downsample via average pooling
                scaled_length = window // scale
                scaled_series = torch.zeros(B, scaled_length)

                for i in range(B):
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
    Encode a batch of 1D series using TimesFM.

    Args:
        past_values_list: list[Tensor], length B; each tensor is (T,)
        frequency_input: LongTensor of shape (B,). If None, defaults to zeros.

    Returns:
        Tensor: (B, P, D) last_hidden_state
    """
    # Frequency id convention:
    # 0: daily or higher frequency; 1: weekly/monthly; 2: quarterly/yearly
    if frequency_input is None:
        frequency_input = torch.zeros(len(past_values_list), dtype=torch.long)

    with torch.no_grad():
        outputs = model(
            past_values=past_values_list,
            freq=frequency_input,
            return_dict=True
        )
        embedding = outputs.last_hidden_state  # (B, P, D)

    return embedding


def extract_multi_view_multi_scale_embeddings(series, lookback_windows=[21, 14, 7], scales=[1, 2]):
    """
    Extract multi-view, multi-scale embeddings and flatten per view.

    Args:
        series: (B, L)
        lookback_windows: list of window lengths
        scales: downsampling factors

    Returns:
        Tensor: (B, V, P*D), where V = len(lookback_windows) * len(scales)
    """
    multi_scale_series = create_multi_scale_series(series, lookback_windows, scales)

    embeddings_list = []
    for scale_series in multi_scale_series:
        # TimesFM expects a list of length B, each element is (T,)
        series_list = [scale_series[i] for i in range(scale_series.shape[0])]
        embedding = get_timesfm_embeddings(series_list)  # (B, P, D)
        embeddings_list.append(embedding)

    all_embeddings = torch.stack(embeddings_list, dim=0)    # (V, B, P, D)
    all_embeddings = all_embeddings.permute(1, 0, 2, 3)     # (B, V, P, D)
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], all_embeddings.shape[1], -1)  # (B, V, P*D)

    return all_embeddings


if __name__ == "__main__":
    # Example usage
    batch_size = 3
    seq_length = 90
    series = torch.randn(batch_size, seq_length)

    embeddings = extract_multi_view_multi_scale_embeddings(
        series,
        lookback_windows=[21, 14, 7],
        scales=[1, 2]
    )

    print(f"Series shape: {series.shape}")
    print(f"Embeddings shape: {embeddings.shape}")

    num_views = len([21, 14, 7]) * len([1, 2])
    print(f"Num views: {num_views}")

    # Shape check if you assume TimesFM outputs P=10, D=64
    expected_shape = (batch_size, num_views, 10 * 64)
    print(f"Expected shape (assumed): {expected_shape}")
    print(f"Match: {embeddings.shape == expected_shape}")
