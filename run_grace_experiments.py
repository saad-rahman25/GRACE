# #!/usr/bin/env python3
# """
# Run GRACE hyperparameter sweep + automatic plotting
# """

# import subprocess
# import pickle
# from pathlib import Path
# from collections import defaultdict

# from plot_utils import (
#     plot_train_val_curves,
#     plot_bestval_grouped_bars
# )

# VISITS = "visit_matrix_large_depr_z_1.csv"
# WEIGHTS = "weights.csv"

# # --------------------------------------------------
# # 1. Run GRACE experiments
# # --------------------------------------------------
# def run_grace():
#     print("=" * 80)
#     print("Running GRACE hyperparameter sweep")
#     print("=" * 80)

#     subprocess.run(
#         [
#             "python", "hyperparam_tune_gde.py",
#             "--csv", VISITS,
#             "--weights", WEIGHTS
#         ],
#         check=True
#     )

# # --------------------------------------------------
# # 2. Load records
# # --------------------------------------------------
# def load_records():
#     with open("records.pkl", "rb") as f:
#         return pickle.load(f)

# # --------------------------------------------------
# # 3. Auto-select best CSV per T
# # --------------------------------------------------
# def select_best_histories(records):
#     """
#     Returns:
#         csv_by_T: Dict[T, csv_path]
#     """
#     best_by_T = {}

#     for r in records:
#         T = int(r["T"])
#         if T not in best_by_T or r["best_val"] < best_by_T[T]["best_val"]:
#             best_by_T[T] = r

#     csv_by_T = {}
#     for T, r in best_by_T.items():
#         model = r["model"]          # e.g. hidden24_lr0.01
#         csv_by_T[T] = Path(f"train_val_history_{model}_T{T}.csv")

#     return csv_by_T

# # --------------------------------------------------
# # 4. Plot everything
# # --------------------------------------------------
# def plot_all(records):
#     # ---- grouped bars (all models)
#     plot_bestval_grouped_bars(
#         records=records,
#         title="GRACE: Best Validation MAE by T and Model",
#         ylabel="Validation MAE",
#         t_order=sorted({r["T"] for r in records}),
#         save_path="grace_bestval_grouped.png"
#     )

#     # ---- training curves (best per T)
#     csv_by_T = select_best_histories(records)

#     plot_train_val_curves(
#         csv_by_T=csv_by_T,
#         title="GRACE: Training / Validation Curves (Best per T)",
#         save_path="grace_train_val_curves.png"
#     )

# # --------------------------------------------------
# # Main
# # --------------------------------------------------
# if __name__ == "__main__":
#     #run_grace()
#     records = load_records()
#     print(records)
#     plot_all(records)



#!/usr/bin/env python3

import pickle
from plot_utils import plot_bestval_grouped_bars
from collections import defaultdict
from pathlib import Path
import re

from plot_utils import plot_train_val_curves


def collect_grace_histories():
    histories = defaultdict(dict)
    pattern = re.compile(
        r"train_val_history_(hidden\d+_lr[0-9.]+)_T(\d+)\.csv"
    )

    for p in Path(".").glob("train_val_history_*.csv"):
        m = pattern.match(p.name)
        if m:
            model, T = m.group(1), int(m.group(2))
            histories[T][model] = p
    print(len(histories))
    return histories


import itertools
import os
from plot_utils import plot_train_val_curves


def plot_grace_train_curves():
    """
    Collect all GRACE history CSV files and plot
    train/validation curves exactly like GDE.
    """

    history_files = {}

    # Hyperparameter grid (must match training)
    for T, hidden, lr in itertools.product([8, 16], [16, 24], [0.01, 0.001]):
        csv_file = f"train_val_history_hidden{hidden}_lr{lr}_T{T}.csv"

        if not os.path.exists(csv_file):
            print(f"[WARN] Missing history file: {csv_file}")
            continue

        # if T not in history_files:
        #     history_files[T] 

        history_files[T] = csv_file

    if len(history_files) == 0:
        print("[ERROR] No history files found — skipping train/val plots.")
        return

    # Same API as GDE
    plot_train_val_curves(
        history_files,
        title="GRACE Train/Val Curves",
        save_path="train_val_curves.png"
    )


if __name__ == "__main__":
    # ---- load records
    with open("records.pkl", "rb") as f:
        records = pickle.load(f)

    # ---- plot best-val bars
    plot_bestval_grouped_bars(
        records=records,
        title="GRACE Best Validation MAE by T and Model",
        ylabel="Validation MAE",
        t_order=[8, 16],
        save_path="grace_bestval_grouped.png"
    )

    # # ---- plot training curves (ALL runs, GCDE-style)
    # histories = collect_grace_histories()

    # for T, csvs in histories.items():
    #     plot_train_val_curves(
    #         csv_by_T={T: p for p in csvs.values()},
    #         title=f"GRACE Train/Val Curves (T={T})",
    #         save_path=f"grace_train_val_T{T}.png"
    #     )
    plot_grace_train_curves()
