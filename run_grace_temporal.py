import argparse
import pandas as pd
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


def build_snapshots(X, history):
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
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1)
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main(args):
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # -----------------------------
    # Load node features
    # -----------------------------
    df = pd.read_csv(args.csv, index_col=0)
    X = torch.tensor(df.values, dtype=torch.float, device=device)

    num_nodes, num_days = X.shape
    print(f"Loaded data: {num_nodes} nodes, {num_days} days")

    # -----------------------------
    # Load weighted graph
    # -----------------------------
    edge_index, edge_weight = load_weight_matrix(args.weights, device)
    print("Loaded weighted adjacency matrix")

    # -----------------------------
    # Temporal windows
    # -----------------------------
    xs, ys = build_snapshots(X, args.history)
    split = int(len(xs) * args.train_ratio)

    xs_train, ys_train = xs[:split], ys[:split]
    xs_test, ys_test = xs[split:], ys[split:]

    print(f"Train windows: {len(xs_train)}")
    print(f"Test windows:  {len(xs_test)}")

    # -----------------------------
    # GRACE model
    # -----------------------------
    encoder = Encoder(
        in_channels=args.history,
        out_channels=args.hidden_dim,
        activation=nn.PReLU()
    )

    model = Model(
        encoder=encoder,
        num_hidden=args.hidden_dim,
        num_proj_hidden=args.hidden_dim,
        tau=args.tau
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # -----------------------------
    # GRACE pretraining
    # -----------------------------
    model.train()
    for epoch in range(args.grace_epochs):
        total_loss = 0.0

        for x in xs_train:
            x1 = drop_feature(x, args.drop_feat)
            x2 = drop_feature(x, args.drop_feat)

            ei1, _ = dropout_edge(edge_index, p=args.drop_edge)
            ei2, _ = dropout_edge(edge_index, p=args.drop_edge)

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
    # Freeze encoder
    # -----------------------------
    for p in model.parameters():
        p.requires_grad = False

    # -----------------------------
    # Regression
    # -----------------------------
    regressor = Regressor(args.hidden_dim).to(device)
    reg_opt = optim.Adam(regressor.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    for epoch in range(args.reg_epochs):
        total_loss = 0.0

        for x, y in zip(xs_train, ys_train):
            with torch.no_grad():
                z = model(x, edge_index)

            pred = regressor(z)
            loss = mse(pred, y)

            reg_opt.zero_grad()
            loss.backward()
            reg_opt.step()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"[REG] Epoch {epoch+1}/{args.reg_epochs} | Loss: {total_loss:.4f}")

    # -----------------------------
    # Evaluation
    # -----------------------------
    regressor.eval()
    test_loss = 0.0

    with torch.no_grad():
        for x, y in zip(xs_test, ys_test):
            z = model(x, edge_index)
            pred = regressor(z)
            test_loss += mse(pred, y).item()

    test_loss /= len(xs_test)

    print("\n==============================")
    print("FINAL BENCHMARK RESULT")
    print(f"Test MSE: {test_loss:.6f}")
    print("==============================\n")


# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True, type=str,
                        help="Node feature CSV (states × days)")
    parser.add_argument("--weights", required=True, type=str,
                        help="Weight matrix CSV (states × states)")

    parser.add_argument("--history", type=int, default=7)
    parser.add_argument("--train_ratio", type=float, default=0.7)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--grace_epochs", type=int, default=200)
    parser.add_argument("--reg_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--drop_feat", type=float, default=0.3)
    parser.add_argument("--drop_edge", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.5)

    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    main(args)




























########################################################################################
# import argparse
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from torch_geometric.utils import dense_to_sparse, dropout_edge
# from model import Encoder, Model, drop_feature
# import torch.nn.functional as F


# # --------------------------------------------------
# # Utils
# # --------------------------------------------------

# def build_fully_connected_edge_index(num_nodes, device):
#     adj = torch.ones((num_nodes, num_nodes), device=device)
#     edge_index, _ = dense_to_sparse(adj)
#     return edge_index


# def build_snapshots(X, history):
#     xs, ys = [], []
#     T = X.shape[1]
#     for t in range(T - history):
#         xs.append(X[:, t:t + history])
#         ys.append(X[:, t + history])
#     return xs, ys


# # --------------------------------------------------
# # Regression Head
# # --------------------------------------------------

# class Regressor(nn.Module):
#     def __init__(self, in_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, in_dim),
#             nn.ReLU(),
#             nn.Linear(in_dim, 1)
#         )

#     def forward(self, z):
#         return self.net(z).squeeze(-1)


# # --------------------------------------------------
# # Main
# # --------------------------------------------------

# def main(args):
#     device = torch.device(args.device)

#     # -----------------------------
#     # Load data
#     # -----------------------------
#     df = pd.read_csv(args.csv, index_col=0)
#     X = torch.tensor(df.values, dtype=torch.float, device=device)

#     num_nodes, num_days = X.shape
#     print(f"Dataset loaded: {num_nodes} nodes, {num_days} days")

#     # -----------------------------
#     # Temporal snapshots
#     # -----------------------------
#     xs, ys = build_snapshots(X, args.history)
#     total = len(xs)
#     train_size = int(args.train_ratio * total)

#     xs_train, ys_train = xs[:train_size], ys[:train_size]
#     xs_test, ys_test = xs[train_size:], ys[train_size:]

#     print(f"History: {args.history} → predict next day")
#     print(f"Train snapshots: {len(xs_train)}")
#     print(f"Test snapshots: {len(xs_test)}")

#     # -----------------------------
#     # Graph
#     # -----------------------------
#     edge_index = build_fully_connected_edge_index(num_nodes, device)

#     # -----------------------------
#     # GRACE model
#     # -----------------------------
#     encoder = Encoder(
#         in_channels=args.history,
#         out_channels=args.hidden_dim,
#         activation=F.relu
#     )

#     model = Model(
#         encoder=encoder,
#         num_hidden=args.hidden_dim,
#         num_proj_hidden=args.hidden_dim,
#         tau=args.tau
#     ).to(device)

#     optimizer = optim.Adam(model.parameters(), lr=args.lr)

#     # -----------------------------
#     # GRACE pretraining
#     # -----------------------------
#     model.train()
#     for epoch in range(args.grace_epochs):
#         total_loss = 0.0

#         for x in xs_train:
#             # feature + edge augmentations
#             x1 = drop_feature(x, args.drop_feat)
#             x2 = drop_feature(x, args.drop_feat)

#             edge_index1, _ = dropout_edge(edge_index, p=args.drop_edge)
#             edge_index2, _ = dropout_edge(edge_index, p=args.drop_edge)

#             z1 = model.encoder(x1, edge_index1)
#             z2 = model.encoder(x2, edge_index2)

#             loss = model.loss(z1, z2)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         if (epoch + 1) % 50 == 0:
#             print(f"[GRACE] Epoch {epoch+1}/{args.grace_epochs}, Loss: {total_loss:.4f}")

#     # -----------------------------
#     # Freeze encoder
#     # -----------------------------
#     for p in model.parameters():
#         p.requires_grad = False

#     # -----------------------------
#     # Regression
#     # -----------------------------
#     regressor = Regressor(args.hidden_dim).to(device)
#     reg_optimizer = optim.Adam(regressor.parameters(), lr=args.lr)
#     mse = nn.MSELoss()

#     for epoch in range(args.reg_epochs):
#         total_loss = 0.0

#         for x, y in zip(xs_train, ys_train):
#             with torch.no_grad():
#                 z = model.encoder(x, edge_index)

#             pred = regressor(z)
#             loss = mse(pred, y)

#             reg_optimizer.zero_grad()
#             loss.backward()
#             reg_optimizer.step()

#             total_loss += loss.item()

#         if (epoch + 1) % 20 == 0:
#             print(f"[REG] Epoch {epoch+1}/{args.reg_epochs}, Loss: {total_loss:.4f}")

#     # -----------------------------
#     # Evaluation
#     # -----------------------------
#     regressor.eval()
#     test_loss = 0.0

#     with torch.no_grad():
#         for x, y in zip(xs_test, ys_test):
#             z = model.encoder(x, edge_index)
#             pred = regressor(z)
#             test_loss += mse(pred, y).item()

#     test_loss /= len(xs_test)

#     print("\n==============================")
#     print("FINAL BENCHMARK RESULT")
#     print(f"Test MSE: {test_loss:.6f}")
#     print("==============================\n")


# # --------------------------------------------------
# # CLI
# # --------------------------------------------------

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--csv", type=str, required=True)
#     parser.add_argument("--history", type=int, default=7)
#     parser.add_argument("--train_ratio", type=float, default=0.7)

#     parser.add_argument("--hidden_dim", type=int, default=256)
#     parser.add_argument("--grace_epochs", type=int, default=200)
#     parser.add_argument("--reg_epochs", type=int, default=100)
#     parser.add_argument("--lr", type=float, default=1e-3)

#     parser.add_argument("--drop_feat", type=float, default=0.3)
#     parser.add_argument("--drop_edge", type=float, default=0.2)
#     parser.add_argument("--tau", type=float, default=0.5)

#     parser.add_argument("--device", type=str, default="cpu")

#     args = parser.parse_args()
#     main(args)
