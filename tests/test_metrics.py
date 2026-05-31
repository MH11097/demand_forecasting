"""Tests for evaluation metrics (Favorita NWRMSLE)."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    evaluate_all,
    mae,
    mape,
    nwrmsle,
    perishable_weights,
    rmse,
    rmspe,
)


def test_nwrmsle_perfect_is_zero():
    y = np.array([0, 5, 100, 1000])
    assert nwrmsle(y, y) == 0.0


def test_nwrmsle_clips_negatives():
    # unit_sales âm (trả hàng) -> clip 0; pred 0 -> log1p(0)=log1p(0) -> sai số 0
    assert nwrmsle(np.array([-5.0]), np.array([0.0])) == 0.0


def test_nwrmsle_matches_manual_rmsle():
    yt, yp = np.array([10.0, 100.0]), np.array([12.0, 90.0])
    d = (np.log1p(yp) - np.log1p(yt)) ** 2
    expected = float(np.sqrt(d.mean()))
    assert np.isclose(nwrmsle(yt, yp), expected)


def test_perishable_weights_values():
    w = perishable_weights([1, 0, 1])
    assert list(w) == [1.25, 1.0, 1.25]


def test_nwrmsle_weighting_changes_result():
    # sai số chỉ ở row0 (perishable) -> trọng số 1.25 làm metric KHÁC bản không trọng số
    yt, yp = np.array([10.0, 100.0]), np.array([20.0, 100.0])
    w = perishable_weights([1, 0])
    weighted = nwrmsle(yt, yp, w)
    unweighted = nwrmsle(yt, yp)
    assert weighted != unweighted
    assert weighted > unweighted   # trọng số dồn vào row có sai số -> tăng metric


def test_rmse_mae_mape():
    assert rmse(np.array([100, 200]), np.array([100, 200])) == 0.0
    assert mae(np.array([100, 200]), np.array([110, 210])) == 10.0
    assert 0 < mape(np.array([100, 200]), np.array([90, 210])) < 1


def test_rmspe_filters_zero_actual_values():
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([999.0, 90.0, 220.0])
    expected = np.sqrt(np.mean([0.1**2, 0.1**2]))
    assert np.isclose(rmspe(y_true, y_pred), expected)


def test_evaluate_all_keys():
    result = evaluate_all(np.array([10, 20, 30]), np.array([11, 19, 31]))
    assert set(result) == {"nwrmsle", "rmse", "mae", "mape", "rmspe"}
    # metric cũ phải BIẾN MẤT
    assert "smape" not in result


def test_evaluate_all_accepts_weights():
    w = perishable_weights([1, 0, 1])
    result = evaluate_all(np.array([10, 20, 30]), np.array([11, 19, 31]), w)
    assert result["nwrmsle"] >= 0


def test_evaluate_all_rejects_empty_arrays():
    with pytest.raises(ValueError, match="empty"):
        evaluate_all(np.array([]), np.array([]))
