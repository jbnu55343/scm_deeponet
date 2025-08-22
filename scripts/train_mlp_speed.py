# scripts/train_mlp_speed.py
import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import mean_absolute_error, r2_score

def load_npz(path):
    d = np.load(path, allow_pickle=True, mmap_mode='r')  # 关键：allow_pickle=True
    X = d["X"]     # (N, F) float32
    Y = d["Y"]     # (N, 1) float32
    feats = d["features"].tolist()

    def _as_py(obj):
        # 尽量把 numpy object/标量 转成 Python 原生对象
        try:
            if isinstance(obj, np.ndarray):
                if obj.shape == ():             # 标量
                    return obj.item()
                if obj.size == 1:               # 单元素数组
                    return obj.reshape(()).item()
                return obj.tolist()             # 多元素就转 list
            return obj
        except Exception:
            return obj

    # meta / split 尝试解析，失败就给合理默认值
    meta = {}
    if "meta" in d.files:
        try:
            meta = _as_py(d["meta"])
            if isinstance(meta, list) and len(meta) == 1 and isinstance(meta[0], dict):
                meta = meta[0]
            if not isinstance(meta, dict):
                meta = {}
        except Exception:
            meta = {}

    split = None
    if "split" in d.files:
        try:
            split = _as_py(d["split"])
        except Exception:
            split = None

    return X, Y, feats, meta, split

def build_mask_for_traveltime(X, features, use_tt=True):
    if (not use_tt) or ("traveltime" not in features):
        return np.ones(X.shape[0], dtype=bool)
    idx = features.index("traveltime")
    tt  = X[:, idx]
    # 过滤 SUMO 哨兵/异常：<=0 或 >=1e5
    m = (tt > 0.0) & (tt < 1e5)
    return m

class NpzDataset(Dataset):
    def __init__(self, X, Y, idx, x_mean=None, x_std=None, y_mean=None, y_std=None):
        self.X = X; self.Y = Y; self.idx = np.asarray(idx, dtype=np.int64)
        self.x_mean = x_mean; self.x_std = x_std
        self.y_mean = y_mean; self.y_std = y_std

    def __len__(self): return self.idx.shape[0]

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = self.X[j].astype(np.float32, copy=False)
        y = self.Y[j].astype(np.float32, copy=False)  # shape (1,)

        # 归一化（推理和验证也要用同一均值方差）
        if self.x_mean is not None:
            x = (x - self.x_mean) / (self.x_std + 1e-8)
        if self.y_mean is not None:
            y = (y - self.y_mean) / (self.y_std + 1e-8)

        return torch.from_numpy(x), torch.from_numpy(y)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(256,256), dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, F)
        return self.net(x)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/dataset_sumo_5km.npz")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=str, default="256,256")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--test_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train", type=int, default=2_000_000, help="限流：最多用多少训练样本（加速）")
    ap.add_argument("--max_val",   type=int, default=200_000)
    ap.add_argument("--max_test",  type=int, default=200_000)
    ap.add_argument("--filter_traveltime", action="store_true",
                    help="过滤 traveltime<=0 或 >=1e5 的样本")
    ap.add_argument("--save", default="models/mlp_speed.pt")
    return ap.parse_args()

@torch.no_grad()
def eval_epoch(model, loader, device, y_mean, y_std):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        pb = model(xb)
        # 反标准化
        pb = pb * (y_std + 1e-8) + y_mean
        yb = yb * (y_std + 1e-8) + y_mean
        ys.append(yb.cpu().numpy())
        ps.append(pb.cpu().numpy())
    y = np.concatenate(ys, 0).reshape(-1)
    p = np.concatenate(ps, 0).reshape(-1)
    mae = mean_absolute_error(y, p)
    rmse = math.sqrt(((y - p) ** 2).mean())
    r2 = r2_score(y, p)
    return mae, rmse, r2

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, Y, feats, meta, split = load_npz(args.npz)
    print(f"[LOAD] X={X.shape} Y={Y.shape} features={feats}")

    # 可选：过滤 traveltime 异常样本
    mask = build_mask_for_traveltime(X, feats, use_tt=args.filter_traveltime)
    if args.filter_traveltime:
        keep = np.where(mask)[0]
        print(f"[FILTER] keep {keep.size}/{mask.size} samples ({keep.size/mask.size:.1%})")
    else:
        keep = np.arange(X.shape[0], dtype=np.int64)

    # 简单随机划分（你也可以用 d['split'] 自带的索引；但注意与 keep 的交集）
    N = keep.size
    idx = keep.copy()
    np.random.shuffle(idx)
    n_test = int(N * args.test_ratio)
    n_val  = int(N * args.val_ratio)
    test_idx = idx[:n_test]
    val_idx  = idx[n_test:n_test+n_val]
    train_idx= idx[n_test+n_val:]

    # 限流（为了更快得到结果）
    if args.max_train and train_idx.size > args.max_train:
        train_idx = np.random.choice(train_idx, args.max_train, replace=False)
    if args.max_val and val_idx.size > args.max_val:
        val_idx = np.random.choice(val_idx, args.max_val, replace=False)
    if args.max_test and test_idx.size > args.max_test:
        test_idx = np.random.choice(test_idx, args.max_test, replace=False)

    # 计算 train 的均值方差（避免信息泄漏）
    x_train = X[train_idx]
    y_train = Y[train_idx]
    x_mean, x_std = x_train.mean(0), x_train.std(0)
    y_mean, y_std = y_train.mean(0), y_train.std(0)
    print("[STATS] x_mean[:3]=", x_mean[:3], "x_std[:3]=", x_std[:3], "y_mean=", y_mean, "y_std=", y_std)

    # Dataset/DataLoader
    ds_train = NpzDataset(X, Y, train_idx, x_mean, x_std, y_mean, y_std)
    ds_val   = NpzDataset(X, Y, val_idx,   x_mean, x_std, y_mean, y_std)
    ds_test  = NpzDataset(X, Y, test_idx,  x_mean, x_std, y_mean, y_std)

    loader_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    loader_val   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    loader_test  = DataLoader(ds_test,  batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # 模型
    hidden = tuple(int(x) for x in args.hidden.split(",") if x)
    model = MLP(in_dim=X.shape[1], hidden=hidden, dropout=args.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_rmse, best_state = 1e9, None
    patience, wait = 5, 0

    for epoch in range(1, args.epochs+1):
        model.train()
        t0 = time.time()
        losses = []
        for xb, yb in loader_train:
            xb = xb.to(device); yb = yb.to(device)
            pb = model(xb)
            loss = loss_fn(pb, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        # 验证（反标准化后计算）
        val_mae, val_rmse, val_r2 = eval_epoch(model, loader_val, device, y_mean, y_std)
        dt = time.time() - t0
        print(f"[E{epoch:02d}] train_mse={np.mean(losses):.4f}  "
              f"val: MAE={val_mae:.3f} RMSE={val_rmse:.3f} R2={val_r2:.4f}  ({dt:.1f}s)")

        if val_rmse < best_rmse - 1e-4:
            best_rmse, wait = val_rmse, 0
            best_state = { "model": model.state_dict(),
                           "x_mean": x_mean, "x_std": x_std,
                           "y_mean": y_mean, "y_std": y_std,
                           "features": feats, "args": vars(args) }
        else:
            wait += 1
            if wait >= patience:
                print("[EARLY-STOP]")
                break

    # 测试
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    test_mae, test_rmse, test_r2 = eval_epoch(model, loader_test, device, y_mean, y_std)
    print(f"[TEST] MAE={test_mae:.3f} RMSE={test_rmse:.3f} R2={test_r2:.4f}")

    # 保存
    Path(os.path.dirname(args.save) or ".").mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.save)
    print(f"[SAVE] {args.save}")

if __name__ == "__main__":
    main()
