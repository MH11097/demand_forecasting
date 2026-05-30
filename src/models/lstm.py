"""LSTM model — global model với dual embedding (store + item).

Entity là CẶP (store_nbr, item_nbr). Dùng 2 embedding riêng:
  - store embedding: vocab = số store DUY NHẤT trong subset, dim=store_emb_dim
  - item  embedding: vocab = số item DUY NHẤT trong subset, dim=item_emb_dim
Vocab SUY RA TỪ DỮ LIỆU (store_idx/item_idx do preprocessor tạo) — KHÔNG hardcode.
Cả hai nối với feature liên tục TRƯỚC LSTM để học tương tác store×item ngay từ bước đầu.

Cửa sổ (window) luôn được giới hạn theo series_id — không vắt qua biên
chuỗi khi training hay prediction.
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.base import BaseModel
from src.models.torch_utils import (
    EarlyStopping,
    SeriesSequenceDataset,
    evaluate_epoch,
    get_loss_fn,
    load_checkpoint,
    prepare_sequences,
    save_checkpoint,
    train_epoch,
)


class LSTMNet(nn.Module):
    """LSTM với dual embedding store+item.

    Input forward: (x_cont, store_idx, item_idx)
      x_cont:    (B, seq_len, n_cont) — feature liên tục
      store_idx: (B,)                 — LongTensor, range [0, store_vocab-1]
      item_idx:  (B,)                 — LongTensor, range [0, item_vocab-1]
    """

    def __init__(
        self,
        n_cont: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int = 1,
        store_vocab: int = 1,
        item_vocab: int = 1,
        store_emb_dim: int = 4,
        item_emb_dim: int = 8,
    ):
        super().__init__()
        # Hai embedding riêng biệt — store và item có thể học pattern độc lập
        self.store_emb = nn.Embedding(store_vocab, store_emb_dim)
        self.item_emb  = nn.Embedding(item_vocab,  item_emb_dim)

        # LSTM nhận đầu vào = feature liên tục + embedding hai chiều
        lstm_input_size = n_cont + store_emb_dim + item_emb_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        # output_size=1 cho direct/recursive; output_size=H cho multioutput
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_cont: torch.Tensor, store_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        x_cont:    (B, seq_len, n_cont)
        store_idx: (B,) — scalar per sample (cố định trong mỗi window)
        item_idx:  (B,) — scalar per sample
        """
        B, S, _ = x_cont.shape
        # embed → (B, emb_dim) → (B, 1, emb_dim) → broadcast → (B, S, emb_dim)
        se = self.store_emb(store_idx).unsqueeze(1).expand(B, S, -1)
        ie = self.item_emb(item_idx).unsqueeze(1).expand(B, S, -1)
        # nối theo chiều feature: (B, S, n_cont+store_emb_dim+item_emb_dim)
        x = torch.cat([x_cont, se, ie], dim=-1)
        out, _ = self.lstm(x)
        # chỉ lấy output bước cuối → tổng hợp toàn chuỗi để dự đoán
        return self.fc(out[:, -1, :])


class LSTMModel(BaseModel):
    name = "lstm"

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", {})
        self.hidden_size   = model_cfg.get("hidden_size", 128)
        self.num_layers    = model_cfg.get("num_layers", 2)
        self.seq_len       = model_cfg.get("seq_len", 90)
        self.batch_size    = model_cfg.get("batch_size", 256)
        self.epochs        = model_cfg.get("epochs", 50)
        self.lr            = model_cfg.get("learning_rate", 0.001)
        self.dropout       = model_cfg.get("dropout", 0.2)
        self.forecast_horizon = model_cfg.get("forecast_horizon", 90)
        self.store_emb_dim = model_cfg.get("store_emb_dim", 4)
        self.item_emb_dim  = model_cfg.get("item_emb_dim", 8)
        # log_target=True: unit_sales đã ở log1p space → predict() cần expm1 inverse
        self.log_target      = config.get("use_log_sales", False)
        self.forecast_strategy = config.get("forecast_strategy", "direct")
        self.loss_fn_name    = config.get("loss_fn", "mse")
        self.patience        = model_cfg.get("patience", 10)
        self.min_delta       = model_cfg.get("min_delta", 0.0)
        self.weight_decay    = model_cfg.get("weight_decay", 0.0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: LSTMNet | None = None
        self.feature_cols: list[str] = []

    # ─── helpers ──────────────────────────────────────────────────────────────

    def _get_series_arrays(self, df: pd.DataFrame):
        """Trả về (series_ids, store_idxs, item_idxs) numpy arrays từ df."""
        sids   = df["series_id"].values.astype(np.int64)  if "series_id" in df.columns else None
        sidxs  = df["store_idx"].values.astype(np.int64)  if "store_idx" in df.columns else np.zeros(len(df), np.int64)
        iidxs  = df["item_idx"].values.astype(np.int64)   if "item_idx"  in df.columns else np.zeros(len(df), np.int64)
        return sids, sidxs, iidxs

    def _make_dataset(self, feats, tgts, sidxs, iidxs, series_ids, horizon, strategy):
        return SeriesSequenceDataset(
            feats, tgts, self.seq_len, horizon, strategy,
            series_ids=series_ids, store_idxs=sidxs, item_idxs=iidxs,
        )

    # ─── training ─────────────────────────────────────────────────────────────

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict:
        start = time.time()
        features, targets, self.feature_cols = prepare_sequences(train_df, config=self.config)

        _train_h = 1 if self.forecast_strategy == "recursive" else self.forecast_horizon
        _out_sz  = self.forecast_horizon if self.forecast_strategy == "multioutput" else 1

        sids, sidxs, iidxs = self._get_series_arrays(train_df)
        train_ds = self._make_dataset(features, targets, sidxs, iidxs, sids, _train_h, self.forecast_strategy)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if val_df is not None and len(val_df) > 0:
            v_feats, v_tgts, _ = prepare_sequences(val_df, self.feature_cols)
            v_sids, v_sidxs, v_iidxs = self._get_series_arrays(val_df)
            val_ds = self._make_dataset(v_feats, v_tgts, v_sidxs, v_iidxs, v_sids, _train_h, self.forecast_strategy)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        n_cont = len(self.feature_cols)
        # vocab embedding SUY RA TỪ DỮ LIỆU: max index + 1 (store_idx/item_idx do preprocessor tạo)
        store_vocab = int(sidxs.max()) + 1 if len(sidxs) else 1
        item_vocab = int(iidxs.max()) + 1 if len(iidxs) else 1
        self.net = LSTMNet(
            n_cont=n_cont,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=_out_sz,
            store_vocab=store_vocab,
            item_vocab=item_vocab,
            store_emb_dim=self.store_emb_dim,
            item_emb_dim=self.item_emb_dim,
        ).to(self.device)

        optimizer  = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion  = get_loss_fn(self.loss_fn_name)
        early_stop = EarlyStopping(patience=self.patience, min_delta=self.min_delta)

        # resume từ checkpoint nếu có
        start_epoch = 0
        resume_from = self.config.get("resume_from")
        if resume_from and os.path.exists(resume_from):
            print(f"Loading checkpoint from {resume_from}...")
            ckpt = load_checkpoint(resume_from, device=str(self.device))
            self.net.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            early_stop.best_loss = ckpt["best_loss"]
            early_stop.counter   = ckpt["patience_counter"]
            self.feature_cols    = ckpt["feature_cols"]
            print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")

        checkpoint_dir  = self.config.get("checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{self.name}_latest.pth")
        checkpoint_best = os.path.join(checkpoint_dir, f"{self.name}_best.pth")

        print(
            f"Training LSTM | strategy={self.forecast_strategy} | device={self.device} | "
            f"epochs={self.epochs} lr={self.lr} hidden={self.hidden_size} H={self.forecast_horizon} "
            f"log={self.log_target} store_emb={self.store_emb_dim} item_emb={self.item_emb_dim}",
            flush=True,
        )
        train_losses: list[float] = []
        val_losses:   list[float] = []
        best_val_loss = float("inf")
        best_epoch    = 0

        for epoch in range(start_epoch, self.epochs):
            tl = train_epoch(self.net, train_loader, optimizer, criterion, self.device)
            train_losses.append(tl)

            vl = None
            if val_loader:
                vl = evaluate_epoch(self.net, val_loader, criterion, self.device)
                val_losses.append(vl)
                if vl < best_val_loss - self.min_delta:
                    best_val_loss = vl
                    best_epoch    = epoch + 1
                    save_checkpoint(checkpoint_best, self.net, optimizer, epoch, early_stop, self.feature_cols)
                if early_stop(vl):
                    print(f"Early stopping at epoch {epoch + 1}.")
                    break

            marker = " *" if vl is not None and vl == best_val_loss else ""
            if vl is not None:
                print(f"Epoch {epoch+1:3d}/{self.epochs} | Train: {tl:.6f} | Val: {vl:.6f}{marker}", flush=True)
            else:
                print(f"Epoch {epoch+1:3d}/{self.epochs} | Train: {tl:.6f}", flush=True)

            save_checkpoint(checkpoint_path, self.net, optimizer, epoch, early_stop, self.feature_cols)

        # khôi phục weights tốt nhất
        if val_loader and os.path.exists(checkpoint_best):
            best_ckpt = load_checkpoint(checkpoint_best, device=str(self.device))
            self.net.load_state_dict(best_ckpt["model_state_dict"])
            print(f"[best] Restored epoch {best_epoch} (val_loss={best_val_loss:.6f}).")

        self._training_time = time.time() - start
        print(f"Training complete in {self._training_time:.1f}s")
        return {
            "training_time": self._training_time,
            "n_samples": len(train_df),
            "epochs_trained": epoch + 1,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss":   val_losses[-1]   if val_losses   else None,
        }

    # ─── prediction public API ────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Dự đoán unit_sales. Trả về array (n,). Luôn per-series để tránh window cross-series."""
        raw = self._predict_series(df, all_horizons=False)
        if self.log_target:
            raw = np.expm1(raw)
        return np.clip(raw, 0, None)

    def predict_all_horizons(self, df: pd.DataFrame) -> np.ndarray:
        """Dự đoán tất cả bước 1..H. Trả về (n, H). Chỉ dùng với multioutput/recursive."""
        if self.forecast_strategy == "direct":
            raise ValueError(
                "predict_all_horizons() không dùng được với strategy='direct'. "
                "Đổi sang 'multioutput' hoặc 'recursive'."
            )
        result = self._predict_series(df, all_horizons=True)
        if self.log_target:
            result = np.expm1(result)
        return np.clip(result, 0, None)

    # ─── prediction internals ─────────────────────────────────────────────────

    def _predict_series(self, df: pd.DataFrame, all_horizons: bool) -> np.ndarray:
        """Prediction per series_id — tránh window vắt qua biên chuỗi."""
        n = len(df)
        H = self.forecast_horizon
        result = np.zeros((n, H)) if all_horizons else np.zeros(n)

        # target col "unit_sales" không có trong feature_cols (đã exclude) → sales_idx=None
        # recursive mode feed-forward là no-op (mặc định multioutput cho dataset này)
        sales_idx = None

        df_pos = df.reset_index(drop=True)
        group_col = "series_id" if "series_id" in df_pos.columns else None

        if group_col is None:
            # không có series_id → chạy toàn bộ (fallback)
            sub_feats, _, _ = prepare_sequences(df_pos, self.feature_cols)
            sidxs = df_pos["store_idx"].values.astype(np.int64) if "store_idx" in df_pos.columns else np.zeros(n, np.int64)
            iidxs = df_pos["item_idx"].values.astype(np.int64)  if "item_idx"  in df_pos.columns else np.zeros(n, np.int64)
            result = self._dispatch(sub_feats, sidxs, iidxs, sales_idx, H, all_horizons)
            return result

        for sid in df_pos[group_col].unique():
            mask = (df_pos[group_col] == sid).values
            pos  = np.where(mask)[0]
            sub  = df_pos.iloc[pos]

            sub_feats, _, _ = prepare_sequences(sub, self.feature_cols)
            sidxs = sub["store_idx"].values.astype(np.int64) if "store_idx" in sub.columns else np.zeros(len(sub), np.int64)
            iidxs = sub["item_idx"].values.astype(np.int64)  if "item_idx"  in sub.columns else np.zeros(len(sub), np.int64)

            sub_preds = self._dispatch(sub_feats, sidxs, iidxs, sales_idx, H, all_horizons)
            result[pos] = sub_preds

        return result

    def _dispatch(self, features, sidxs, iidxs, sales_idx, H, all_horizons):
        if all_horizons:
            if self.forecast_strategy == "multioutput":
                return self._run_multioutput_all(features, sidxs, iidxs, H)
            else:
                return self._run_recursive_all(features, sidxs, iidxs, sales_idx, H)
        elif self.forecast_strategy == "recursive":
            return self._run_recursive(features, sidxs, iidxs, sales_idx, H)
        elif self.forecast_strategy == "multioutput":
            return self._run_multioutput_single(features, sidxs, iidxs, H)
        else:
            return self._run_direct(features, sidxs, iidxs, H)

    # ─── runner helpers (dataset-based) ──────────────────────────────────────

    def _make_pred_loader(self, features, sidxs, iidxs, H, strategy):
        """Tạo DataLoader cho inference (targets dummy zeros)."""
        ds = SeriesSequenceDataset(
            features, np.zeros(len(features)), self.seq_len, H, strategy,
            series_ids=None, store_idxs=sidxs, item_idxs=iidxs,
        )
        return DataLoader(ds, batch_size=self.batch_size)

    @staticmethod
    def _pad_preds(raw: np.ndarray, n: int, offset: int) -> np.ndarray:
        """Pad raw predictions (length = n-offset) vào array độ dài n."""
        full = np.zeros(n)
        if len(raw) > 0 and offset < n:
            take = n - offset
            full[offset:] = np.clip(raw[:take], 0, None)
            full[:offset] = raw[0]
        return full

    def _run_direct(self, features, sidxs, iidxs, H):
        loader = self._make_pred_loader(features, sidxs, iidxs, H, "direct")
        self.net.eval()
        preds = []
        with torch.no_grad():
            for x, s, i, _ in loader:
                preds.append(np.atleast_1d(self.net(x.to(self.device), s.to(self.device), i.to(self.device)).squeeze(-1).cpu().numpy()))
        raw = np.concatenate(preds) if preds else np.array([])
        return self._pad_preds(raw, len(features), self.seq_len + H - 1)

    def _run_multioutput_single(self, features, sidxs, iidxs, H):
        """Multioutput → lấy bước H (cột cuối)."""
        loader = self._make_pred_loader(features, sidxs, iidxs, H, "multioutput")
        self.net.eval()
        preds = []
        with torch.no_grad():
            for x, s, i, _ in loader:
                preds.append(self.net(x.to(self.device), s.to(self.device), i.to(self.device)).cpu().numpy()[:, -1])
        raw = np.concatenate(preds) if preds else np.array([])
        return self._pad_preds(raw, len(features), self.seq_len + H - 1)

    def _run_multioutput_all(self, features, sidxs, iidxs, H):
        """Multioutput → trả về (n, H) tất cả bước."""
        n = len(features)
        loader = self._make_pred_loader(features, sidxs, iidxs, H, "multioutput")
        self.net.eval()
        chunks = []
        with torch.no_grad():
            for x, s, i, _ in loader:
                chunks.append(self.net(x.to(self.device), s.to(self.device), i.to(self.device)).cpu().numpy())
        raw = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, H))
        off = self.seq_len
        result = np.zeros((n, H))
        result[off : off + len(raw)] = np.clip(raw[: n - off], 0, None)
        if len(raw) > 0:
            result[:off] = raw[0]
        return result

    def _run_recursive(self, features, sidxs, iidxs, sales_idx, H):
        """Lăn bánh H bước, trả về (n,) tại bước H."""
        n, CHUNK = len(features), max(self.batch_size, 256)
        n_samp = n - self.seq_len
        if n_samp <= 0:
            return np.zeros(n)
        # sidx/iidx không đổi trong mỗi window (series_id cố định) → dùng giá trị đầu window
        finals = []
        self.net.eval()
        with torch.no_grad():
            for cs in range(0, n_samp, CHUNK):
                end = min(cs + CHUNK, n_samp)
                wins = np.stack([features[j : j + self.seq_len] for j in range(cs, end)]).copy()
                s_t = torch.LongTensor(sidxs[cs : end]).to(self.device)
                i_t = torch.LongTensor(iidxs[cs : end]).to(self.device)
                sp = None
                for step in range(H):
                    sp = np.atleast_1d(
                        self.net(torch.FloatTensor(wins).to(self.device), s_t, i_t)
                        .squeeze(-1).cpu().numpy()
                    )
                    if step < H - 1:
                        nr = wins[:, -1, :].copy()
                        if sales_idx is not None:
                            nr[:, sales_idx] = sp
                        wins = np.concatenate([wins[:, 1:, :], nr[:, np.newaxis, :]], axis=1)
                finals.append(sp)
        raw = np.concatenate(finals) if finals else np.array([])
        return self._pad_preds(raw, n, self.seq_len + H - 1)

    def _run_recursive_all(self, features, sidxs, iidxs, sales_idx, H):
        """Recursive → trả về (n, H) tất cả bước."""
        n, CHUNK = len(features), max(self.batch_size, 256)
        n_samp = n - self.seq_len
        if n_samp <= 0:
            return np.zeros((n, H))
        chunks = []
        self.net.eval()
        with torch.no_grad():
            for cs in range(0, n_samp, CHUNK):
                end = min(cs + CHUNK, n_samp)
                wins = np.stack([features[j : j + self.seq_len] for j in range(cs, end)]).copy()
                s_t = torch.LongTensor(sidxs[cs : end]).to(self.device)
                i_t = torch.LongTensor(iidxs[cs : end]).to(self.device)
                cp = np.zeros((len(wins), H))
                for step in range(H):
                    sp = np.atleast_1d(
                        self.net(torch.FloatTensor(wins).to(self.device), s_t, i_t)
                        .squeeze(-1).cpu().numpy()
                    )
                    cp[:, step] = sp
                    if step < H - 1:
                        nr = wins[:, -1, :].copy()
                        if sales_idx is not None:
                            nr[:, sales_idx] = sp
                        wins = np.concatenate([wins[:, 1:, :], nr[:, np.newaxis, :]], axis=1)
                chunks.append(cp)
        raw = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, H))
        result = np.zeros((n, H))
        result[self.seq_len : self.seq_len + len(raw)] = np.clip(raw[: n - self.seq_len], 0, None)
        if len(raw) > 0:
            result[: self.seq_len] = raw[0]
        return result
