import numpy as np
from scipy.signal import correlate
from statsmodels.tsa.seasonal import STL
import torch
import warnings

warnings.filterwarnings('ignore')


def extract_temporal_features(data):
    """
    Extract lightweight temporal representations (placeholder for a foundation TS model).

    Args:
        data: (B, L)

    Returns:
        features: (B, D)
    """
    B, L = data.shape
    features = np.zeros((B, 4))

    for i in range(B):
        series = data[i]
        features[i, 0] = np.mean(series)
        features[i, 1] = np.std(series)
        features[i, 2] = np.max(series) - np.min(series)
        features[i, 3] = np.median(series)

    return features


def find_similar_sequences(features, k=2):
    """
    Retrieve top-k most similar sequences based on cosine similarity.

    Args:
        features: (B, D)
        k: number of neighbors

    Returns:
        similarity_matrix: (B, B)
        topk_indices: (B, k)
    """
    from sklearn.metrics.pairwise import cosine_similarity

    similarity_matrix = cosine_similarity(features)
    np.fill_diagonal(similarity_matrix, -np.inf)
    topk_indices = np.argsort(similarity_matrix, axis=1)[:, -k:]

    return similarity_matrix, topk_indices


def stl_decomposition(series, period=7):
    """
    STL decomposition into trend/seasonal/residual components.

    Args:
        series: (L,)
        period: seasonal period

    Returns:
        trend, seasonal, residual: each (L,)
    """
    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    return result.trend, result.seasonal, result.resid


def mixup(series1, series2):
    """
    Trend fusion via MixUp.

    Args:
        series1: (L,)
        series2: (L,)

    Returns:
        mixed: (L,)
        series1, series2: original inputs (for optional debugging)
    """
    lam = np.random.uniform(0.3, 0.7)
    mixed = lam * series1 + (1 - lam) * series2
    return mixed, series1, series2


def phase_alignment(series1, series2):
    """
    Phase-align series2 to series1 using cross-correlation.

    Args:
        series1: (L,)
        series2: (L,)

    Returns:
        aligned_series2: (L,)
        series1, series2: original inputs (for optional debugging)
    """
    correlation = correlate(series1, series2, mode='full')
    lag = np.arange(-len(series1) + 1, len(series2))
    max_lag = lag[np.argmax(correlation)]

    if max_lag > 0:
        aligned_series2 = np.roll(series2, -max_lag)
        aligned_series2[-max_lag:] = np.nan
    elif max_lag < 0:
        aligned_series2 = np.roll(series2, -max_lag)
        aligned_series2[:abs(max_lag)] = np.nan
    else:
        aligned_series2 = series2

    # Fill NaNs by linear interpolation
    nan_mask = np.isnan(aligned_series2)
    if np.any(nan_mask):
        x = np.arange(len(aligned_series2))
        aligned_series2[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], aligned_series2[~nan_mask])

    return aligned_series2, series1, series2


def seasonal_fusion(seasonal1, seasonal2):
    """
    Seasonal fusion with phase alignment + averaging.

    Args:
        seasonal1: (L,)
        seasonal2: (L,)

    Returns:
        fused: (L,)
    """
    aligned_seasonal2, _, _ = phase_alignment(seasonal1, seasonal2)
    fused = 0.5 * seasonal1 + 0.5 * aligned_seasonal2
    return fused


def residual_fusion(residual1, residual2, timegan_model=None):
    """
    Residual fusion (optional TimeGAN-based generation).

    Args:
        residual1: (L,)
        residual2: (L,)
        timegan_model: optional generator

    Returns:
        fused: (L,)
    """
    if timegan_model is not None:
        residuals = np.stack([residual1, residual2], axis=0).reshape(2, len(residual1), 1)
        generated_residuals = timegan_model.generate(residuals)
        fused = generated_residuals[0].flatten()
    else:
        alpha = np.random.uniform(0.4, 0.6)
        fused = alpha * residual1 + (1 - alpha) * residual2

    return fused


def reconstruct_series(trend, seasonal, residual):
    """
    Reconstruct series from components.

    Args:
        trend, seasonal, residual: (L,)

    Returns:
        reconstructed: (L,)
    """
    return trend + seasonal + residual


def augment_sample(original_series, similar_series):
    """
    Augment one sample by component-wise fusion with a similar sample.

    Args:
        original_series: (L, F)
        similar_series: (L, F)

    Returns:
        augmented_series: (L, F)
    """
    L, num_features = original_series.shape
    augmented_series = np.zeros((L, num_features))

    for feature_idx in range(num_features):
        orig_feature = original_series[:, feature_idx]
        similar_feature = similar_series[:, feature_idx]

        trend1, seasonal1, residual1 = stl_decomposition(orig_feature, period=7)
        trend2, seasonal2, residual2 = stl_decomposition(similar_feature, period=7)

        fused_trend, _, _ = mixup(trend1, trend2)
        fused_seasonal = seasonal_fusion(seasonal1, seasonal2)
        fused_residual = residual_fusion(residual1, residual2)

        augmented_series[:, feature_idx] = reconstruct_series(fused_trend, fused_seasonal, fused_residual)

    return augmented_series


def sample_augmentation(enhanced_data, k=2):
    """
    Main augmentation pipeline.

    Args:
        enhanced_data: (B, L, F)
        k: neighbors per sample

    Returns:
        augmented_data: (B + B*k, L, F)
    """
    B, L, num_features = enhanced_data.shape

    # Use the first feature channel to compute similarity
    base_features = enhanced_data[:, :, 0]  # (B, L)

    temporal_features = extract_temporal_features(base_features)  # (B, D)
    _, similar_indices = find_similar_sequences(temporal_features, k=k)

    augmented_samples = []
    for i in range(B):
        original_sample = enhanced_data[i]  # (L, F)

        for j in range(k):
            similar_idx = similar_indices[i][j]
            similar_sample = enhanced_data[similar_idx]  # (L, F)

            augmented_samples.append(augment_sample(original_sample, similar_sample))

    augmented_data = np.concatenate([enhanced_data, np.array(augmented_samples)], axis=0)
    print(f"Augmentation complete: {B} -> {len(augmented_data)} samples")

    return augmented_data


if __name__ == "__main__":
    # Example usage
    B_train, L, num_features = 100, 90, 8
    enhanced_train_data = np.random.randn(B_train, L, num_features)

    print(f"Original data shape: {enhanced_train_data.shape}")

    augmented_train_data = sample_augmentation(
        enhanced_train_data,
        k=2,
    )

    print(f"Augmented data shape: {augmented_train_data.shape}")

    print("\nExample:")
    print("Original (feature 0):", enhanced_train_data[0, :10, 0])
    print("Augmented (feature 0):", augmented_train_data[B_train, :10, 0])
