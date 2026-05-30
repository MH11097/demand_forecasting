"""Shared PyTorch utilities for the LSTM model.

Cột target: "unit_sales". Entity key: "series_id" = factorize(store_nbr, item_nbr).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Cột liên tục (không phải ID / target / date) — chọn động bằng _select_feature_cols().
_EXCLUDE_FROM_FEATURES = {
    "unit_sales", "store_nbr", "item_nbr", "series_id",
    "id", "cluster", "class",   # ID / categorical thô — không dùng làm feature liên tục
    "store_idx", "item_idx",    # embedding index — xử lý riêng
    "date",
    "was_return", "returned_units",  # EDA-only, dẫn xuất từ target cùng ngày
}


def _select_feature_cols(df, config: dict | None = None) -> list[str]:
    """Chọn cột feature liên tục: tất cả numeric ngoài ID/target/date."""
    cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in _EXCLUDE_FROM_FEATURES
    ]
    feature_cfg = (config or {}).get("features", {})
    if not feature_cfg.get("use_promo", True):
        cols = [
            c for c in cols
            if c != "onpromotion" and not c.startswith("promo_")
            and c not in {"days_since_last_promo", "days_until_next_promo"}
        ]
    if not feature_cfg.get("use_oil", True):
        cols = [c for c in cols if c not in {"dcoilwtico", "oil_lag_7"}]
    if not feature_cfg.get("use_holiday", True):
        cols = [c for c in cols if "holiday" not in c and "event" not in c]
    if not feature_cfg.get("use_perishable", True):
        cols = [c for c in cols if c != "perishable"]
    return cols


def prepare_sequences(
    df, feature_cols=None, target_col: str = "unit_sales", log_target: bool = False,
    config: dict | None = None,
):
    """Chuẩn bị feature matrix và target từ DataFrame.

    Args:
        df: DataFrame đã có feature engineering.
        feature_cols: danh sách cột feature liên tục. None = tự chọn (lần train đầu).
        target_col: tên cột target (mặc định "unit_sales").
        log_target: không dùng trực tiếp ở đây (flag cho model biết cần inverse transform).

    Returns:
        (features: np.float32, targets: np.float32, feature_cols: list[str])
    """
    if feature_cols is None:
        feature_cols = _select_feature_cols(df, config=config)
    features = df[feature_cols].fillna(0).values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)
    return features, targets, feature_cols


class SeriesSequenceDataset(Dataset):
    """Sequence windowing dataset cho các mô hình chuỗi thời gian.

    Mỗi sample trả về (x_cont, store_idx, item_idx, y) để LSTMNet có thể
    dùng embedding riêng cho store và item (composite entity).

    Args:
        features: numpy (n, d) — chỉ cột liên tục, KHÔNG bao gồm store_idx/item_idx.
        targets: numpy (n,).
        store_idxs: numpy (n,) — chỉ số embedding store (0..n_store-1).
        item_idxs:  numpy (n,) — chỉ số embedding item (0..n_item-1).
        series_ids: numpy (n,) — series_id = factorize(store_nbr,item_nbr), dùng loại window vắt
            qua biên chuỗi. Nếu None → hành vi cũ (toàn bộ window hợp lệ).
        forecast_horizon: số bước dự đoán.
        strategy: "direct" | "multioutput" | "recursive".
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int = 30,
        forecast_horizon: int = 1,
        strategy: str = "direct",
        series_ids: np.ndarray | None = None,
        store_idxs: np.ndarray | None = None,
        item_idxs: np.ndarray | None = None,
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.strategy = strategy

        # store/item idx không bắt buộc — fallback về 0 nếu thiếu; .copy() để tensor writable
        n = len(features)
        _sidxs = store_idxs.copy() if store_idxs is not None else np.zeros(n, np.int64)
        _iidxs = item_idxs.copy()  if item_idxs  is not None else np.zeros(n, np.int64)
        self.store_idxs = torch.LongTensor(_sidxs)
        self.item_idxs  = torch.LongTensor(_iidxs)

        full_window = seq_len + forecast_horizon
        if series_ids is not None and len(series_ids) == n:
            # data sort theo [series_id, date] -> start cùng series_id với end → window hợp lệ
            valid = [
                i for i in range(n - full_window + 1)
                if series_ids[i] == series_ids[i + full_window - 1]
            ]
            self.valid_indices = np.array(valid, dtype=np.int64)
        else:
            self.valid_indices = np.arange(max(0, n - full_window + 1), dtype=np.int64)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = int(self.valid_indices[idx])
        x = self.features[start : start + self.seq_len]      # (seq_len, d)
        # store/item idx không đổi trong window (series_id cố định) → lấy giá trị đầu window
        sidx = self.store_idxs[start]
        iidx = self.item_idxs[start]
        if self.strategy == "multioutput":
            y = self.targets[start + self.seq_len : start + self.seq_len + self.forecast_horizon]
        else:
            y = self.targets[start + self.seq_len + self.forecast_horizon - 1]
        return x, sidx, iidx, y


class EarlyStopping:
    """Early stopping callback."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        # val_loss cải thiện → reset đếm; không cải thiện liên tục patience lần → dừng tránh overfit
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _mape_loss(output: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Mean Absolute Percentage Error."""
    denom = torch.clamp(torch.abs(target), min=eps)
    return torch.mean(torch.abs((output - target) / denom))


def get_loss_fn(name: str):
    """Factory trả về loss function từ tên.

    Args:
        name: "mse" | "mae" | "mape" (mse trên log target ≈ NWRMSLE)
    """
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name == "mape":
        return _mape_loss
    else:
        raise ValueError(f"loss_fn không hợp lệ: '{name}'. Chọn: mse, mae, mape")


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train one epoch, trả về average loss.

    Dataloader trả về (x_cont, sidx, iidx, y) từ SeriesSequenceDataset.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x_cont, sidx, iidx, y_batch in dataloader:
        x_cont, sidx, iidx, y_batch = (
            x_cont.to(device), sidx.to(device), iidx.to(device), y_batch.to(device)
        )
        optimizer.zero_grad()
        output = model(x_cont, sidx, iidx).squeeze(-1)
        loss = criterion(output, y_batch)
        loss.backward()
        # clip gradient tránh exploding gradient trong LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_epoch(model, dataloader, criterion, device):
    """Evaluate one epoch, trả về average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for x_cont, sidx, iidx, y_batch in dataloader:
            x_cont, sidx, iidx, y_batch = (
                x_cont.to(device), sidx.to(device), iidx.to(device), y_batch.to(device)
            )
            output = model(x_cont, sidx, iidx).squeeze(-1)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def save_checkpoint(checkpoint_path: str, model, optimizer, epoch: int, early_stop: EarlyStopping, feature_cols: list[str]):
    """Lưu checkpoint để resume training."""
    import os
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": early_stop.best_loss,
        "patience_counter": early_stop.counter,
        "feature_cols": feature_cols,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Tải checkpoint để resume training."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint
