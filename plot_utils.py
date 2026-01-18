from typing import Dict, Union, List
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_train_val_curves(
    csv_by_T: Dict[int, Union[str, Path]],
    title: str = "Training curves",
    epoch_col: str = "epoch",
    train_col: str = "train_mse",
    val_col: str = "val_mae",
    best_col: str = "best_val_mae_so_far",
    save_path: Union[str, Path] = None
) -> None:
    color_by_T = {8: "tab:blue", 16: "tab:pink"}
    dfs = {T: pd.read_csv(p) for T, p in csv_by_T.items()}

    fig, (ax_val, ax_train) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for T, df in dfs.items():
        x = df[epoch_col].astype(int)
        color = color_by_T.get(T, "black")

        ax_val.plot(x, df[val_col], color=color, label=f"T={T}")
        ax_train.plot(x, df[train_col], color=color)

        best_val = df[best_col].iloc[-1]
        best_row = df[df[best_col] == best_val].iloc[0]
        best_epoch = int(best_row[epoch_col])
        best_val_mae = float(best_row[val_col])
        best_train_mse = float(best_row[train_col])

        ax_val.axvline(best_epoch, linestyle="--", color=color, linewidth=1.5)
        ax_train.axvline(best_epoch, linestyle="--", color=color, linewidth=1.5)
        ax_val.annotate(f"Best val:{best_val_mae:.3f}", xy=(best_epoch, best_val_mae),
                        xytext=(6, 150), textcoords="offset points", color=color, fontsize=9, va="center")
        ax_train.annotate(f"Best val:{best_train_mse:.3e}", xy=(best_epoch, best_train_mse),
                          xytext=(6, 150), textcoords="offset points", color=color, fontsize=9, va="center")

    ax_val.set_title("Val-MAE vs Epoch")
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Val MAE")
    ax_val.grid(True, linestyle="--", alpha=0.5)
    ax_train.set_title("Train MSE vs Epoch")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Train MSE")
    ax_train.grid(True, which="both", linestyle="--", alpha=0.5)
    handles, labels = ax_val.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), frameon=False)
    fig.suptitle(title, y=1.05)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()

def plot_bestval_grouped_bars(
    records: List[Dict],
    title: str = "Best Val by T and Model",
    ylabel: str = "best_val",
    t_order: List[int] = None,
    model_order: List[str] = None,
    save_path: Union[str, Path] = None
) -> None:
    Ts = sorted({int(r["T"]) for r in records}) if t_order is None else list(t_order)
    models = sorted({str(r["model"]) for r in records}) if model_order is None else list(model_order)
    val = {(int(r["T"]), str(r["model"])): float(r["best_val"]) for r in records}
    default_cycle = [
        "tab:blue", "tab:pink", "tab:orange", "tab:green", "tab:purple",
        "tab:brown", "tab:red", "tab:gray", "tab:olive", "tab:cyan"
    ]
    color_by_model = {m: default_cycle[i % len(default_cycle)] for i, m in enumerate(models)}
    x = np.arange(len(Ts))
    n_models = len(models)
    group_width = 0.8
    bar_w = group_width / max(1, n_models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(models):
        offsets = x - group_width / 2 + (i + 0.5) * bar_w
        heights = [val.get((T, m), np.nan) for T in Ts]
        ax.bar(offsets, heights, width=bar_w, label=m, color=color_by_model[m])
        for xi, hi in zip(offsets, heights):
            if np.isfinite(hi):
                ax.annotate(f"{hi:.4g}", (xi, hi), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("T")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T={T}" for T in Ts])
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()
