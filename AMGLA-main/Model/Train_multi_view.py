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

# Multi-view retrieval enhancement model
from Multi_view import MultiViewRetrievalModel

# Device selection
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


def create_sliding_windows(data, lookback=21, forecast_steps=7, start_index=20, end_index=90):
    """
    Create sliding windows within a specified index range.

    Args:
        data: (B, L)
        lookback: history length
        forecast_steps: horizon length
        start_index: first center index (inclusive)
        end_index: last center index (exclusive)

    Returns:
        X: (num_windows, lookback)
        y: (num_windows, forecast_steps)
    """
    B, L = data.shape
    X_list, y_list = [], []

    effective_end = end_index
    for i in range(B):
        series = data[i]
        for t in range(start_index, effective_end):
            X = series[t - lookback + 1:t + 1]
            y = series[t + 1:t + 1 + forecast_steps]
            X_list.append(X)
            y_list.append(y)

    return np.array(X_list), np.array(y_list)


def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """Create PyTorch DataLoaders."""
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

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
    """Evaluate model loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(data_loader.dataset)


def predict(model, data_loader):
    """Run inference and return numpy arrays."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(DEVICE)
            outputs = model(batch_x)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def RMSE_MAE_MAPE(preds, targets):
    """Compute RMSE / MAE / MAPE."""
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    mape = np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100
    return rmse, mae, mape


def print_log(*args, log=None):
    """Print to stdout and optionally write to a file handle."""
    message = ' '.join(str(arg) for arg in args)
    print(message)
    if log:
        log.write(message + '\n')


def extract_multi_view_features(data, lookback=21):
    """
    Build simple multi-view features from full sequences (example implementation).

    Args:
        data: (B, L)
        lookback: recent window length

    Returns:
        multi_view_features: (B, C, D)
    """
    B, L = data.shape
    C, D = 4, 64
    multi_view_features = np.zeros((B, C, D))

    for i in range(B):
        series = data[i]

        # View 1: summary statistics
        stats = np.zeros(D)
        stats[0] = np.mean(series)
        stats[1] = np.std(series)
        stats[2] = np.max(series) - np.min(series)
        stats[3] = np.median(series)
        multi_view_features[i, 0, :len(stats)] = stats

        # View 2: most recent lookback values
        recent = series[-lookback:] if L >= lookback else series
        multi_view_features[i, 1, :len(recent)] = recent

        # View 3: FFT magnitude (truncated)
        if L >= 10:
            fft_features = np.abs(np.fft.fft(series)[:D // 2])
            multi_view_features[i, 2, :len(fft_features)] = fft_features

        # View 4: first-order differences
        diff = np.diff(series)
        multi_view_features[i, 3, :len(diff)] = diff

    return multi_view_features


def extract_multi_view_features_from_windows(window_data, original_data_shape):
    """
    Build multi-view features from sliding windows.

    Args:
        window_data: (num_windows, lookback)
        original_data_shape: kept for compatibility (unused)

    Returns:
        multi_view_features: (num_windows, C, D)
    """
    num_windows, lookback = window_data.shape
    C, D = 4, 64
    multi_view_features = np.zeros((num_windows, C, D))

    for i in range(num_windows):
        window = window_data[i]

        # View 1: summary statistics + up-trend ratio
        stats = np.zeros(D)
        stats[0] = np.mean(window)
        stats[1] = np.std(window)
        stats[2] = np.max(window) - np.min(window)
        stats[3] = np.median(window)
        stats[4] = np.sum(np.diff(window) > 0) / (len(window) - 1)
        multi_view_features[i, 0, :len(stats)] = stats

        # View 2: raw window values
        multi_view_features[i, 1, :lookback] = window

        # View 3: FFT magnitude (truncated)
        if lookback >= 10:
            fft_features = np.abs(np.fft.fft(window))[:D // 2]
            multi_view_features[i, 2, :len(fft_features)] = fft_features

        # View 4: first-order differences
        diff = np.diff(window)
        multi_view_features[i, 3, :len(diff)] = diff

    return multi_view_features


def enhance_window_features_with_multi_view(original_data, window_data, retrieval_db, retrieval_sales, model, forecast_steps=7):
    """
    Enhance window features using a multi-view retrieval model.

    Args:
        original_data: (B, L)
        window_data: (num_windows, lookback)
        retrieval_db: (N, lookback)
        retrieval_sales: (N, forecast_steps)
        model: MultiViewRetrievalModel
        forecast_steps: horizon length

    Returns:
        enhanced_data: (B, L - forecast_steps, 8)
    """
    B, L = original_data.shape
    num_windows, lookback = window_data.shape

    window_mv_features = extract_multi_view_features_from_windows(window_data, (num_windows, lookback))
    retrieval_mv_features = extract_multi_view_features_from_windows(retrieval_db, (num_windows, lookback))

    with torch.no_grad():
        predicted_sales, aggregated_features = model.predict(
            torch.FloatTensor(window_mv_features).to(DEVICE),
            torch.FloatTensor(retrieval_mv_features).to(DEVICE),
            torch.FloatTensor(retrieval_sales).to(DEVICE)
        )

    predicted_sales = predicted_sales.reshape(B, -1, forecast_steps)

    # Feature layout: [current_value, future_1..future_7]
    enhanced_data = np.zeros((B, L - forecast_steps, 8))

    for i in range(B):
        for t in range(L - forecast_steps):
            enhanced_data[i, t, 0] = original_data[i, t]

            if t < 20:
                for step in range(1, 8):
                    enhanced_data[i, t, step] = original_data[i, t + step] if t + step < L else 0
            else:
                enhanced_data[i, t, 1:8] = predicted_sales[i, t - 20, :].cpu().numpy()

    return enhanced_data


def train_multi_view_model_on_windows(train_windows, train_sales, val_windows, val_sales):
    """
    Train the retrieval model on window-level samples.

    Args:
        train_windows: (num_train_windows, lookback)
        train_sales: (num_train_windows, forecast_steps)
        val_windows: (num_val_windows, lookback)
        val_sales: (num_val_windows, forecast_steps)

    Returns:
        model: trained MultiViewRetrievalModel
    """
    num_train, lookback = train_windows.shape
    num_val, _ = val_windows.shape

    train_mv_features = extract_multi_view_features_from_windows(train_windows, (num_train, lookback))
    val_mv_features = extract_multi_view_features_from_windows(val_windows, (num_val, lookback))

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

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_features = torch.FloatTensor(train_mv_features).to(DEVICE)
    train_sales_tensor = torch.FloatTensor(train_sales).to(DEVICE)
    val_features = torch.FloatTensor(val_mv_features).to(DEVICE)
    val_sales_tensor = torch.FloatTensor(val_sales).to(DEVICE)

    print("Training multi-view retrieval model on sliding windows...")
    best_val_loss = float('inf')
    patience, wait = 10, 0

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        pred_sales, _ = model(train_features, train_features, train_sales_tensor)
        loss = criterion(pred_sales, train_sales_tensor)

        loss.backward()
        optimizer.step()

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
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    model.load_state_dict(best_state_dict)
    print("Training finished.")
    return model


def main_enhancement_pipeline_with_windows():
    """End-to-end demo pipeline for window creation, training, and feature enhancement."""
    np.random.seed(42)
    B_train, B_val, B_test = 60, 20, 20
    L = 97
    lookback = 21
    forecast_steps = 7

    train_data = np.random.randn(B_train, L)
    val_data = np.random.randn(B_val, L)
    test_data = np.random.randn(B_test, L)

    X_train, y_train = create_sliding_windows(train_data, lookback, forecast_steps, 20)
    X_val, y_val = create_sliding_windows(val_data, lookback, forecast_steps, 20)
    X_test, y_test = create_sliding_windows(test_data, lookback, forecast_steps, 20)

    print("Sliding window shapes:")
    print(f"Train: {X_train.shape} -> y: {y_train.shape}")
    print(f"Val:   {X_val.shape} -> y: {y_val.shape}")
    print(f"Test:  {X_test.shape} -> y: {y_test.shape}")

    multi_view_model = train_multi_view_model_on_windows(X_train, y_train, X_val, y_val)

    print("\nEnhancing train windows...")
    enhanced_train = enhance_window_features_with_multi_view(
        train_data, X_train, X_train, y_train, multi_view_model
    )

    print("Enhancing val windows...")
    enhanced_val = enhance_window_features_with_multi_view(
        val_data, X_val, X_train, y_train, multi_view_model
    )

    print("Enhancing test windows...")
    combined_db = np.concatenate([X_train, X_val], axis=0)
    combined_sales = np.concatenate([y_train, y_val], axis=0)
    enhanced_test = enhance_window_features_with_multi_view(
        test_data, X_test, combined_db, combined_sales, multi_view_model
    )

    print("\nEnhanced feature shapes:")
    print(f"Train: {enhanced_train.shape}")
    print(f"Val:   {enhanced_val.shape}")
    print(f"Test:  {enhanced_test.shape}")

    print("\nExample enhanced sequence (first sample, first 5 time steps):")
    for t in range(5):
        features = enhanced_train[0, t]
        print(f"t={t}: x={features[0]:6.2f}, future={features[1:].round(2)}")

    return enhanced_train, enhanced_val, enhanced_test, multi_view_model


if __name__ == "__main__":
    enhanced_train, enhanced_val, enhanced_test, multi_view_model = main_enhancement_pipeline_with_windows()

    np.save('enhanced_train_mv.npy', enhanced_train)
    np.save('enhanced_val_mv.npy', enhanced_val)
    np.save('enhanced_test_mv.npy', enhanced_test)

    print("\nSaved:")
    print("enhanced_train_mv.npy")
    print("enhanced_val_mv.npy")
    print("enhanced_test_mv.npy")
