#!/usr/bin/env python3
"""
hyperparam_tune_gde.py

Hyperparameter tuning for GRACE + downstream GCN regression.

Usage:
python hyperparam_tune_gde.py --csv visit_matrix_large_depr_z_1.csv --weights weights.csv
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import dense_to_sparse, dropout_edge

from model import Encoder, Model, drop_feature


# --------------------------------------------------
# Utils
# --------------------------------------------------

def load_weight_matrix(path, device):
    W = pd.read_csv(path, header=None).values
    W = torch.tensor(W, dtype=torch.float, device=device)
    edge_index, edge_weight = dense_to_sparse(W)
    return edge_index, edge_weight


def build_snapshots(X, history: int):
    """Build temporal windows of size `history`."""
    xs, ys = [], []
    T = X.shape[1]
    for t in range(T - history):
        xs.append(X[:, t:t + history])
        ys.append(X[:, t + history])
    return xs, ys


# --------------------------------------------------
# Regression Head
# --------------------------------------------------

class Regressor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1)
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


# --------------------------------------------------
# Downstream training
# --------------------------------------------------

def train_downstream(model: Model, xs_train, ys_train, xs_val, ys_val, hidden_dim: int,
                     lr: float, edge_index, edge_weight, device, reg_epochs: int = 250):
    """
    Train downstream 2-layer GCN + MLP regression head.
    Early stopping on val MAE.
    """
    # Freeze encoder
    for p in model.parameters():
        p.requires_grad = False

    # Initialize downstream regressor
    regressor = Regressor(hidden_dim).to(device)
    optimizer = optim.Adam(regressor.parameters(), lr=lr)
    mse = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None
    history_log = []

    for epoch in range(reg_epochs):
        total_loss = 0.0

        # Training
        regressor.train()
        for x, y in zip(xs_train, ys_train):
            with torch.no_grad():
                z = model(x, edge_index)  # shape: num_nodes x hidden_dim
            pred = regressor(z)
            loss = mse(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        regressor.eval()
        val_mae = 0.0
        with torch.no_grad():
            for x, y in zip(xs_val, ys_val):
                z = model(x, edge_index)
                pred = regressor(z)
                val_mae += torch.mean(torch.abs(pred - y)).item()
        val_mae /= len(xs_val)

        # Track best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = regressor.state_dict()

        # Log for plotting
        history_log.append({
            "epoch": epoch + 1,
            "train_mse": total_loss / len(xs_train),
            "val_mae": val_mae,
            "best_val_mae_so_far": best_val_mae
        })

    # Load best regressor
    regressor.load_state_dict(best_state)

    # Save CSV for plotting
    df_hist = pd.DataFrame(history_log)
    return regressor, best_val_mae, df_hist


# --------------------------------------------------
# Main hyperparameter loop
# --------------------------------------------------

def main(args):
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Load node features
    # -----------------------------
    df = pd.read_csv(args.csv, index_col=0)
    X_full = torch.tensor(df.values, dtype=torch.float, device=device)
    num_nodes, num_days = X_full.shape
    print(f"Loaded data: {num_nodes} nodes, {num_days} days")

    # -----------------------------
    # Load weighted graph
    # -----------------------------
    edge_index, edge_weight = load_weight_matrix(args.weights, device)
    print("Loaded weighted adjacency matrix")

    # -----------------------------
    # Hyperparameter search
    # -----------------------------
    hidden_dim_options = args.hidden_dim_options
    lr_options = args.lr_options
    history_options = args.history_options

    best_val_mae_overall = float("inf")
    best_combination = None
    records = []  # For grouped bar plot

    for hidden_dim in hidden_dim_options:
        for lr in lr_options:
            for history in history_options:
                print(f"\n=== Trying hidden_dim={hidden_dim}, lr={lr}, history={history} ===")

                # Build temporal snapshots
                xs, ys = build_snapshots(X_full, history)
                n_snapshots = len(xs)
                train_split = int(n_snapshots * 0.6)                     # This way fo split maintains the sequential flow of data
                val_split = int(n_snapshots * 0.8)

                xs_train, ys_train = xs[:train_split], ys[:train_split]
                xs_val, ys_val = xs[train_split:val_split], ys[train_split:val_split]
                xs_test, ys_test = xs[val_split:], ys[val_split:]

                # -----------------------------
                # GRACE encoder + model
                # -----------------------------
                encoder = Encoder(
                    in_channels=history,
                    out_channels=hidden_dim,
                    activation=nn.PReLU()
                )
                model = Model(
                    encoder=encoder,
                    num_hidden=hidden_dim,
                    num_proj_hidden=hidden_dim,
                    tau=args.tau
                ).to(device)

                optimizer = optim.Adam(model.parameters(), lr=lr)

                # -----------------------------
                # GRACE pretraining
                # -----------------------------
                model.train()
                for epoch in range(args.grace_epochs):
                    total_loss = 0.0
                    for x in xs_train:
                        x1 = drop_feature(x, args.drop_feat)
                        x2 = drop_feature(x, args.drop_feat)

                        # Drop edges
                        ei1, mask1 = dropout_edge(edge_index, p=args.drop_edge)
                        ei2, mask2 = dropout_edge(edge_index, p=args.drop_edge)
                        ew1 = edge_weight[mask1]
                        ew2 = edge_weight[mask2]

                        # Forward pass
                        z1 = model(x1, ei1)
                        z2 = model(x2, ei2)
                        loss = model.loss(z1, z2)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()

                    if (epoch + 1) % 50 == 0:
                        print(f"[GRACE] Epoch {epoch+1}/{args.grace_epochs} | Loss: {total_loss:.4f}")

                # -----------------------------
                # Train downstream
                # -----------------------------
                downstream_gcn, val_mae, df_history = train_downstream(
                    model, xs_train, ys_train, xs_val, ys_val,
                    hidden_dim, lr, edge_index, edge_weight, device,
                    reg_epochs=args.reg_epochs
                )

                # Save CSV for plotting
                csv_path = f"train_val_history_hidden{hidden_dim}_lr{lr}_T{history}.csv"
                df_history.to_csv(csv_path, index=False)

                records.append({
                    "T": history,
                    "model": f"hidden{hidden_dim}_lr{lr}",
                    "best_val": val_mae
                })

                # Track best combination
                if val_mae < best_val_mae_overall:
                    best_val_mae_overall = val_mae
                    best_combination = (hidden_dim, lr, history)
                    best_model_state = downstream_gcn.state_dict()
                    best_encoder_state = model.state_dict()

    # -----------------------------
    # Evaluate on test set using best checkpoint
    # -----------------------------
    if best_combination is None:
        raise ValueError("No best combination found! Check training loop.")

    hidden_dim, lr, history = best_combination
    print(f"\n=== Best hyperparameters: hidden_dim={hidden_dim}, lr={lr}, history={history} ===")

    # Rebuild model and encoder for best hyperparameters
    encoder = Encoder(
        in_channels=history,
        out_channels=hidden_dim,
        activation=nn.PReLU()
    )
    model = Model(
        encoder=encoder,
        num_hidden=hidden_dim,
        num_proj_hidden=hidden_dim,
        tau=args.tau
    ).to(device)
    model.load_state_dict(best_encoder_state)

    # Build snapshots
    xs, ys = build_snapshots(X_full, history)
    n_snapshots = len(xs)
    train_split = int(n_snapshots * 0.6)
    val_split = int(n_snapshots * 0.8)
    xs_train, ys_train = xs[:train_split], ys[:train_split]
    xs_val, ys_val = xs[train_split:val_split], ys[train_split:val_split]
    xs_test, ys_test = xs[val_split:], ys[val_split:]

    # Downstream regressor on best checkpoint
    downstream_gcn = Regressor(hidden_dim).to(device)
    downstream_gcn.load_state_dict(best_model_state)
    downstream_gcn.eval()

    mse = nn.MSELoss()
    mae_total = 0.0
    rmse_total = 0.0

    with torch.no_grad():
        for x, y in zip(xs_test, ys_test):
            z = model(x, edge_index)
            pred = downstream_gcn(z)
            mae_total += torch.mean(torch.abs(pred - y)).item()
            rmse_total += torch.sqrt(torch.mean((pred - y)**2)).item()

    mae_total /= len(xs_test)
    rmse_total /= len(xs_test)

    print("\n==============================")
    print("FINAL BENCHMARK RESULT")
    print(f"Test MAE : {mae_total:.6f}")
    print(f"Test RMSE: {rmse_total:.6f}")
    print("==============================\n")

    # Save records for grouped bar plots
    import pickle
    with open("records.pkl", "wb") as f:
        pickle.dump(records, f)
    print("Saved records.pkl for plotting.")


# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True, type=str,
                        help="Node feature CSV (states × days)")
    parser.add_argument("--weights", required=True, type=str,
                        help="Weight matrix CSV (states × states)")

    parser.add_argument("--hidden_dim_options", type=int, nargs="+", default=[16, 24])
    parser.add_argument("--lr_options", type=float, nargs="+", default=[0.01, 0.001])
    parser.add_argument("--history_options", type=int, nargs="+", default=[8, 16])

    parser.add_argument("--grace_epochs", type=int, default=200)
    parser.add_argument("--reg_epochs", type=int, default=250)

    parser.add_argument("--drop_feat", type=float, default=0.3)
    parser.add_argument("--drop_edge", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.5)

    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    main(args)
