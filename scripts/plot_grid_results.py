"""Vẽ biểu đồ kết quả grid search theo từng phase.

Usage:
    python scripts/plot_grid_results.py --phase A
    python scripts/plot_grid_results.py --phase B
    python scripts/plot_grid_results.py --phase AB   # so sánh A vs B
    python scripts/plot_grid_results.py --phase A --save
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


RESULTS_DIR = Path("results/lstm")


def load_phase_A_results():
    """Phase A: seq_len grid search (20260325, num_layers=1, recursive)."""
    # Lấy đúng 5 run của Phase A từ grid_lstm_seqlen_recursive.json
    target_seqlens = {14, 21, 30, 44, 60}
    # Lọc theo timestamp 20260325 và experiment_name chứa seq_len=
    runs = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not (d / "result.json").exists():
            continue
        r = json.loads((d / "result.json").read_text(encoding="utf-8"))
        ts = r.get("timestamp", "")
        exp = r.get("experiment_name", "")
        sl = r["config"]["model"]["seq_len"]
        strategy = r["config"].get("forecast_strategy", "")
        layers = r["config"]["model"]["num_layers"]
        # Chỉ lấy Phase A: ngày 25/03, layers=1, recursive, nằm trong target set
        if ("2026-03-25" in ts and layers == 1 and strategy == "recursive"
                and sl in target_seqlens and "seq_len" in exp):
            runs.append({
                "seq_len": sl,
                "val_nwrmsle": r["metrics"]["nwrmsle"],
                "test_nwrmsle": r["test_metrics"]["nwrmsle"],
                "best_epoch": r["loss_summary"]["best_val_epoch"],
                "epochs_trained": r["training_info"]["epochs_trained"],
                "best_val_loss": r["loss_summary"]["best_val_loss"],
            })
    # Dedup: giữ run đầu tiên cho mỗi seq_len (theo thứ tự timestamp)
    seen = {}
    for run in runs:
        sl = run["seq_len"]
        if sl not in seen:
            seen[sl] = run
    return sorted(seen.values(), key=lambda x: x["seq_len"])


def plot_phase_A(results: list[dict], save: bool = False):
    seq_lens = [r["seq_len"] for r in results]
    val_nwrmsle = [r["val_nwrmsle"] for r in results]
    test_nwrmsle = [r["test_nwrmsle"] for r in results]
    best_epochs = [r["best_epoch"] for r in results]
    epochs_trained = [r["epochs_trained"] for r in results]

    best_idx = int(np.argmin(val_nwrmsle))
    colors_val = ["#2196F3"] * len(results)
    colors_val[best_idx] = "#4CAF50"

    x = np.arange(len(seq_lens))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phase A — seq_len Grid Search (Recursive, H=30, hidden=64, layers=1)",
                 fontsize=13, fontweight="bold")

    # ── Left: NWRMSLE bar chart ──
    ax = axes[0]
    bars_val = ax.bar(x - width/2, val_nwrmsle, width, label="Val NWRMSLE",
                      color=colors_val, edgecolor="white", linewidth=0.8)
    bars_test = ax.bar(x + width/2, test_nwrmsle, width, label="Test NWRMSLE",
                       color=[c + "88" for c in colors_val], edgecolor="white", linewidth=0.8)

    # Annotate values
    for i, (bar_v, bar_t) in enumerate(zip(bars_val, bars_test)):
        ax.text(bar_v.get_x() + bar_v.get_width()/2, bar_v.get_height() + 0.005,
                f"{val_nwrmsle[i]:.4f}", ha="center", va="bottom", fontsize=8.5,
                fontweight="bold" if i == best_idx else "normal")
        ax.text(bar_t.get_x() + bar_t.get_width()/2, bar_t.get_height() + 0.005,
                f"{test_nwrmsle[i]:.4f}", ha="center", va="bottom", fontsize=8.5, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels([f"seq_len={sl}" for sl in seq_lens], rotation=15)
    ax.set_ylabel("NWRMSLE")
    ax.set_ylim(0, max(max(val_nwrmsle), max(test_nwrmsle)) * 1.18)
    ax.set_title("Val & Test NWRMSLE theo seq_len")
    ax.legend()
    ax.axhline(val_nwrmsle[best_idx], color="#4CAF50", linestyle="--", alpha=0.5, linewidth=1)
    ax.grid(axis="y", alpha=0.3)

    # Best marker
    ax.annotate(f"★ Best\n{val_nwrmsle[best_idx]:.4f}",
                xy=(x[best_idx] - width/2, val_nwrmsle[best_idx]),
                xytext=(x[best_idx] - width/2 + 0.5, val_nwrmsle[best_idx] + 0.04),
                arrowprops=dict(arrowstyle="->", color="#4CAF50"),
                color="#4CAF50", fontsize=9, fontweight="bold")

    # ── Right: Epoch info ──
    ax2 = axes[1]
    bar_best = ax2.bar(x, best_epochs, 0.5, label="Best epoch (checkpoint)",
                       color=colors_val, edgecolor="white", linewidth=0.8)
    ax2.bar(x, [e - b for e, b in zip(epochs_trained, best_epochs)], 0.5,
            bottom=best_epochs, label="Epochs after best (patience)",
            color="#FF980044", edgecolor="white", linewidth=0.8)

    for i, (be, et) in enumerate(zip(best_epochs, epochs_trained)):
        ax2.text(x[i], et + 0.5, f"stop={et}", ha="center", va="bottom", fontsize=8.5)
        ax2.text(x[i], be / 2, f"best={be}", ha="center", va="center",
                 fontsize=8, color="white", fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"seq_len={sl}" for sl in seq_lens], rotation=15)
    ax2.set_ylabel("Epoch")
    ax2.set_title("Best epoch & Early stopping")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        out = Path("results") / "phase_A_seqlen_grid.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.show()
    return fig


# ─── Phase B ──────────────────────────────────────────────────────────────────

def load_phase_B_results():
    """Phase B: multioutput seq_len grid (20260325 cuối ngày, num_layers=1, multioutput)."""
    target_dirs = [
        "hidden_size_64__num_layers_1__h30__log__multioutput__20260325_233055",
        "hidden_size_64__num_layers_1__h30__log__multioutput__20260325_234102",
        "hidden_size_64__num_layers_1__h30__log__multioutput__20260325_234555",
        "hidden_size_64__num_layers_1__h30__log__multioutput__20260325_235133",
    ]
    runs = []
    for d in target_dirs:
        p = RESULTS_DIR / d / "result.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        sl = r["config"]["model"]["seq_len"]
        val_windows_store674 = max(0, 64 - sl - 30 + 1)  # val_min=64, H=30
        runs.append({
            "seq_len": sl,
            "val_nwrmsle": r["metrics"]["nwrmsle"],
            "test_nwrmsle": r["test_metrics"]["nwrmsle"],
            "best_epoch": r["loss_summary"]["best_val_epoch"],
            "epochs_trained": r["training_info"]["epochs_trained"],
            "val_windows_674": val_windows_store674,  # windows Store 674 trong val_loader
        })
    return sorted(runs, key=lambda x: x["seq_len"])


def plot_phase_B(results: list[dict], save: bool = False):
    seq_lens = [r["seq_len"] for r in results]
    val_nwrmsle = [r["val_nwrmsle"] for r in results]
    test_nwrmsle = [r["test_nwrmsle"] for r in results]
    best_epochs = [r["best_epoch"] for r in results]
    epochs_trained = [r["epochs_trained"] for r in results]
    val_wins = [r["val_windows_674"] for r in results]

    best_idx = int(np.argmin(val_nwrmsle))
    # Đánh dấu run có val_windows=0 (Store 674 bị loại khỏi early stopping)
    colors_val = []
    for r in results:
        if r["val_windows_674"] == 0:
            colors_val.append("#FF5722")  # đỏ: early stopping bị ảnh hưởng
        elif r["seq_len"] == results[best_idx]["seq_len"]:
            colors_val.append("#4CAF50")  # xanh: best
        else:
            colors_val.append("#2196F3")  # blue: normal

    x = np.arange(len(seq_lens))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phase B — seq_len Grid Search (Multioutput, H=30, hidden=64, layers=1)",
                 fontsize=13, fontweight="bold")

    # ── Left: NWRMSLE bar chart ──
    ax = axes[0]
    bars_val = ax.bar(x - width/2, val_nwrmsle, width, label="Val NWRMSLE",
                      color=colors_val, edgecolor="white", linewidth=0.8)
    bars_test = ax.bar(x + width/2, test_nwrmsle, width, label="Test NWRMSLE",
                       color=[c + "88" for c in colors_val], edgecolor="white", linewidth=0.8)

    for i, (bar_v, bar_t) in enumerate(zip(bars_val, bars_test)):
        ax.text(bar_v.get_x() + bar_v.get_width()/2, bar_v.get_height() + 0.004,
                f"{val_nwrmsle[i]:.4f}", ha="center", va="bottom", fontsize=8.5,
                fontweight="bold" if i == best_idx else "normal")
        ax.text(bar_t.get_x() + bar_t.get_width()/2, bar_t.get_height() + 0.004,
                f"{test_nwrmsle[i]:.4f}", ha="center", va="bottom", fontsize=8.5, color="#555")
        # Val windows Store 674
        ax.text(x[i], 0.01, f"wins674={val_wins[i]}", ha="center", va="bottom",
                fontsize=7.5, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"seq_len={sl}" for sl in seq_lens], rotation=15)
    ax.set_ylabel("NWRMSLE")
    ax.set_ylim(0, max(max(val_nwrmsle), max(test_nwrmsle)) * 1.2)
    ax.set_title("Val & Test NWRMSLE theo seq_len\n(số trắng = val windows Store 674)")
    ax.legend()
    ax.axhline(val_nwrmsle[best_idx], color="#4CAF50", linestyle="--", alpha=0.5, linewidth=1)
    ax.grid(axis="y", alpha=0.3)

    # Legend màu
    from matplotlib.patches import Patch
    legend_extra = [
        Patch(color="#4CAF50", label="Best"),
        Patch(color="#2196F3", label="Normal"),
        Patch(color="#FF5722", label="Store 674 excluded (wins674=0)"),
    ]
    ax.legend(handles=legend_extra, fontsize=8)

    ax.annotate(f"★ Best\n{val_nwrmsle[best_idx]:.4f}",
                xy=(x[best_idx] - width/2, val_nwrmsle[best_idx]),
                xytext=(x[best_idx] + 0.5, val_nwrmsle[best_idx] + 0.03),
                arrowprops=dict(arrowstyle="->", color="#4CAF50"),
                color="#4CAF50", fontsize=9, fontweight="bold")

    # ── Right: Epoch info ──
    ax2 = axes[1]
    ax2.bar(x, best_epochs, 0.5, label="Best epoch (checkpoint)",
            color=colors_val, edgecolor="white", linewidth=0.8)
    ax2.bar(x, [e - b for e, b in zip(epochs_trained, best_epochs)], 0.5,
            bottom=best_epochs, label="Epochs after best (patience)",
            color="#FF980044", edgecolor="white", linewidth=0.8)

    for i, (be, et) in enumerate(zip(best_epochs, epochs_trained)):
        ax2.text(x[i], et + 0.3, f"stop={et}", ha="center", va="bottom", fontsize=8.5)
        ax2.text(x[i], max(be / 2, 1), f"best={be}", ha="center", va="center",
                 fontsize=8, color="white", fontweight="bold")
        if val_wins[i] == 0:
            ax2.text(x[i], et + 2, "⚠ 0 wins\nStore674", ha="center", va="bottom",
                     fontsize=7, color="#FF5722")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"seq_len={sl}" for sl in seq_lens], rotation=15)
    ax2.set_ylabel("Epoch")
    ax2.set_title("Best epoch & Early stopping\n(⚠ = Store 674 excluded from val_loader)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        out = Path("results") / "phase_B_seqlen_multioutput.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.show()
    return fig


def plot_phase_AB(res_a: list[dict], res_b: list[dict], save: bool = False):
    """So sánh Recursive (Phase A) vs Multioutput (Phase B) — best của mỗi strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phase A vs B — Recursive vs Multioutput (H=30, hidden=64, layers=1)",
                 fontsize=13, fontweight="bold")

    # ── Left: NWRMSLE theo seq_len cho cả 2 strategy ──
    ax = axes[0]
    sl_a = [r["seq_len"] for r in res_a]
    sl_b = [r["seq_len"] for r in res_b]
    val_a = [r["val_nwrmsle"] for r in res_a]
    val_b = [r["val_nwrmsle"] for r in res_b]
    test_a = [r["test_nwrmsle"] for r in res_a]
    test_b = [r["test_nwrmsle"] for r in res_b]

    ax.plot(sl_a, val_a, "o-", color="#2196F3", linewidth=2, markersize=7, label="Recursive Val")
    ax.plot(sl_a, test_a, "s--", color="#2196F3", linewidth=1.5, markersize=6, alpha=0.6, label="Recursive Test")
    ax.plot(sl_b, val_b, "o-", color="#FF5722", linewidth=2, markersize=7, label="Multioutput Val")
    ax.plot(sl_b, test_b, "s--", color="#FF5722", linewidth=1.5, markersize=6, alpha=0.6, label="Multioutput Test")

    for sl, v in zip(sl_a, val_a):
        ax.annotate(f"{v:.4f}", (sl, v), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.5, color="#2196F3")
    for sl, v in zip(sl_b, val_b):
        ax.annotate(f"{v:.4f}", (sl, v), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=7.5, color="#FF5722")

    ax.set_xlabel("seq_len")
    ax.set_ylabel("NWRMSLE")
    ax.set_title("Val NWRMSLE: Recursive vs Multioutput")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(sorted(set(sl_a + sl_b)))

    # ── Right: Best comparison bar chart ──
    ax2 = axes[1]
    best_a = min(res_a, key=lambda r: r["val_nwrmsle"])
    best_b = min(res_b, key=lambda r: r["val_nwrmsle"])

    categories = [
        f"Recursive\nseq_len={best_a['seq_len']}",
        f"Multioutput\nseq_len={best_b['seq_len']}",
    ]
    val_vals = [best_a["val_nwrmsle"], best_b["val_nwrmsle"]]
    test_vals = [best_a["test_nwrmsle"], best_b["test_nwrmsle"]]
    colors = ["#2196F3", "#FF5722"]

    xb = np.arange(2)
    bars_v = ax2.bar(xb - 0.2, val_vals, 0.35, label="Val NWRMSLE", color=colors, alpha=0.9)
    bars_t = ax2.bar(xb + 0.2, test_vals, 0.35, label="Test NWRMSLE", color=colors, alpha=0.5)

    for bar, v in zip(bars_v, val_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, v in zip(bars_t, test_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    improvement = (best_a["val_nwrmsle"] - best_b["val_nwrmsle"]) / best_a["val_nwrmsle"] * 100
    ax2.set_title(f"Best của mỗi strategy\nMultioutput cải thiện {improvement:.1f}% vs Recursive")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("NWRMSLE")
    ax2.set_ylim(0, max(val_vals + test_vals) * 1.2)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        out = Path("results") / "phase_AB_comparison.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.show()
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["A", "B", "AB"], default="A")
    parser.add_argument("--save", action="store_true", help="Lưu ảnh vào results/")
    args = parser.parse_args()

    if args.phase == "A":
        results = load_phase_A_results()
        if not results:
            print("Không tìm thấy kết quả Phase A.")
            return
        print(f"Tìm thấy {len(results)} run Phase A:")
        for r in results:
            marker = " ★" if r["val_nwrmsle"] == min(x["val_nwrmsle"] for x in results) else ""
            print(f"  seq_len={r['seq_len']:2d} | val_nwrmsle={r['val_nwrmsle']:.4f} | "
                  f"test_nwrmsle={r['test_nwrmsle']:.4f} | best_ep={r['best_epoch']}{marker}")
        plot_phase_A(results, save=args.save)

    elif args.phase == "B":
        results = load_phase_B_results()
        if not results:
            print("Không tìm thấy kết quả Phase B.")
            return
        print(f"Tìm thấy {len(results)} run Phase B (multioutput):")
        for r in results:
            marker = " ★" if r["val_nwrmsle"] == min(x["val_nwrmsle"] for x in results) else ""
            warn = " ⚠ Store674 excluded" if r["val_windows_674"] == 0 else f" (Store674 wins={r['val_windows_674']})"
            print(f"  seq_len={r['seq_len']:2d} | val_nwrmsle={r['val_nwrmsle']:.4f} | "
                  f"test_nwrmsle={r['test_nwrmsle']:.4f} | best_ep={r['best_epoch']}{marker}{warn}")
        plot_phase_B(results, save=args.save)

    elif args.phase == "AB":
        res_a = load_phase_A_results()
        res_b = load_phase_B_results()
        if not res_a or not res_b:
            print("Thiếu dữ liệu Phase A hoặc B.")
            return
        print(f"\nPhase A (recursive) best: seq_len={min(res_a, key=lambda r: r['val_nwrmsle'])['seq_len']} "
              f"val_nwrmsle={min(r['val_nwrmsle'] for r in res_a):.4f}")
        print(f"Phase B (multioutput) best: seq_len={min(res_b, key=lambda r: r['val_nwrmsle'])['seq_len']} "
              f"val_nwrmsle={min(r['val_nwrmsle'] for r in res_b):.4f}")
        plot_phase_AB(res_a, res_b, save=args.save)


if __name__ == "__main__":
    main()
