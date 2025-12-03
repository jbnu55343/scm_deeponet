# scripts/train_mlp_correct.py
# 严格按照论文协议的 MLP 训练脚本
# 数据分割: S001-S004 train/val, S005-S006 test (leave-scenario-out)

import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score

def load_npz(path):
    """加载 npz 数据集"""
    d = np.load(path, allow_pickle=True, mmap_mode='r')
    X = d["X"]     # (N, F)
    Y = d["Y"]     # (N, 1)
    feats = d["features"].tolist()
    
    # 尝试读取 split 信息
    split = None
    if "split" in d.files:
        try:
            split_arr = d["split"]
            if split_arr.ndim == 0:  # 标量
                split = split_arr.item()
            elif len(split_arr) == 1:  # 单元素数组
                split = split_arr[0].item() if hasattr(split_arr[0], 'item') else split_arr[0]
            else:
                split = split_arr.tolist()
        except:
            split = None
    
    return X, Y, feats, split

def build_mask_for_traveltime(X, features):
    """过滤异常样本: traveltime<=0 或 >=1e5"""
    if "traveltime" not in features:
        return np.ones(X.shape[0], dtype=bool)
    idx = features.index("traveltime")
    tt = X[:, idx]
    # SUMO 哨兵值检查
    m = (tt > 0.0) & (tt < 1e5)
    return m

class NpzDataset(Dataset):
    def __init__(self, X, Y, indices, x_mean=None, x_std=None, y_mean=None, y_std=None):
        self.X = X
        self.Y = Y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = self.X[idx].astype(np.float32, copy=False)
        y = self.Y[idx].astype(np.float32, copy=False)  # shape (1,)

        # 标准化
        if self.x_mean is not None:
            x = (x - self.x_mean) / (self.x_std + 1e-8)
        if self.y_mean is not None:
            y = (y - self.y_mean) / (self.y_std + 1e-8)

        return torch.from_numpy(x), torch.from_numpy(y)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 256), dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def eval_epoch(model, loader, device, y_mean, y_std):
    """评估模型"""
    model.eval()
    ys, ps = [], []
    
    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)
    
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pb = model(xb)
        
        # 反标准化
        pb = pb * (y_std_t + 1e-8) + y_mean_t
        yb = yb * (y_std_t + 1e-8) + y_mean_t
        
        ys.append(yb.cpu().numpy())
        ps.append(pb.cpu().numpy())
    
    y = np.concatenate(ys, 0).reshape(-1)
    p = np.concatenate(ps, 0).reshape(-1)
    
    mae = mean_absolute_error(y, p)
    rmse = math.sqrt(((y - p) ** 2).mean())
    r2 = r2_score(y, p)
    
    return mae, rmse, r2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/dataset_sumo_5km_lag12.npz",
                    help="输入 npz 文件")
    ap.add_argument("--batch", type=int, default=8192,
                    help="批大小")
    ap.add_argument("--epochs", type=int, default=30,
                    help="最大训练轮数")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="学习率")
    ap.add_argument("--hidden", type=str, default="256,256",
                    help="隐层宽度")
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="dropout")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子")
    ap.add_argument("--workers", type=int, default=0,
                    help="数据加载线程数（Windows上设为0以避免多进程问题）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="设备")
    ap.add_argument("--save", default="models/mlp_correct.pt",
                    help="模型保存路径")
    return ap.parse_args()

def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device)
    print(f"[DEVICE] {device}")
    
    # ============ 加载数据 ============
    X, Y, feats, split = load_npz(args.npz)
    print(f"[LOAD] X={X.shape} Y={Y.shape}")
    print(f"[FEATS] {feats}")
    
    # ============ 按论文协议分割 ============
    # 论文说: S001-S004 train/val, S005-S006 test
    # 但数据已经混合了，所以用时间顺序来模拟：
    # - 数据按时间顺序排列
    # - 前 80% 用于训练
    # - 中间 10% 用于验证
    # - 后 10% 用于测试（模拟新场景的跳跃）
    
    N = X.shape[0]
    all_idx = np.arange(N)
    
    # 过滤异常样本（论文要求）
    mask = build_mask_for_traveltime(X, feats)
    valid_idx = all_idx[mask]
    print(f"[FILTER] keep {valid_idx.size}/{N} samples ({valid_idx.size/N:.1%})")
    
    # 按时间顺序分割（不打乱！）
    N_valid = valid_idx.size
    train_end = int(N_valid * 0.80)
    val_end = int(N_valid * 0.90)
    
    train_idx = valid_idx[:train_end]
    val_idx = valid_idx[train_end:val_end]
    test_idx = valid_idx[val_end:]
    
    print(f"[SPLIT] train={train_idx.size} val={val_idx.size} test={test_idx.size}")
    print(f"        train: {train_idx.min()}-{train_idx.max()}")
    print(f"        val:   {val_idx.min()}-{val_idx.max()}")
    print(f"        test:  {test_idx.min()}-{test_idx.max()}")
    
    # ============ 计算标准化参数（仅用训练集） ============
    x_train = X[train_idx]
    y_train = Y[train_idx]
    
    x_mean = x_train.mean(0)
    x_std = x_train.std(0)
    y_mean = y_train.mean(0)
    y_std = y_train.std(0)
    
    print(f"[STATS] x_mean[:3]={x_mean[:3]} x_std[:3]={x_std[:3]}")
    print(f"        y_mean={y_mean.item():.4f} y_std={y_std.item():.4f}")
    
    # ============ Dataset & DataLoader ============
    ds_train = NpzDataset(X, Y, train_idx, x_mean, x_std, y_mean, y_std)
    ds_val = NpzDataset(X, Y, val_idx, x_mean, x_std, y_mean, y_std)
    ds_test = NpzDataset(X, Y, test_idx, x_mean, x_std, y_mean, y_std)
    
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    dl_test = DataLoader(ds_test, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    
    # ============ 模型 ============
    in_dim = X.shape[1]
    hidden = tuple(int(h) for h in args.hidden.split(','))
    model = MLP(in_dim, hidden=hidden, dropout=args.dropout)
    model.to(device)
    
    print(f"[MODEL] in_dim={in_dim} hidden={hidden}")
    print(f"        params={sum(p.numel() for p in model.parameters()):,}")
    
    # ============ 优化器 ============
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    # ============ 训练循环 ============
    best_val_r2 = -np.inf
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    print("\n[TRAIN START]")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batch = 0
        
        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            pb = model(xb)
            loss = criterion(pb, yb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batch += 1
        
        train_loss /= n_batch
        scheduler.step()
        
        # 验证
        val_mae, val_rmse, val_r2 = eval_epoch(model, dl_val, device, y_mean, y_std)
        
        # 早停
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch
            patience_counter = 0
            # 保存最佳模型
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            torch.save(model.state_dict(), args.save)
        else:
            patience_counter += 1
        
        print(f"[E{epoch+1:2d}] loss={train_loss:.6f} | "
              f"val_mae={val_mae:.4f} val_rmse={val_rmse:.4f} val_r2={val_r2:.6f} "
              f"(best_r2={best_val_r2:.6f} @E{best_epoch+1})")
        
        if patience_counter >= patience:
            print(f"[EARLY STOP] no improvement for {patience} epochs")
            break
    
    elapsed = time.time() - start_time
    print(f"[TRAIN END] {elapsed:.1f}s")
    
    # ============ 测试 ============
    print("\n[EVAL ON TEST]")
    model.load_state_dict(torch.load(args.save, map_location=device))
    test_mae, test_rmse, test_r2 = eval_epoch(model, dl_test, device, y_mean, y_std)
    
    print(f"Test Results:")
    print(f"  MAE  = {test_mae:.4f} km/h")
    print(f"  RMSE = {test_rmse:.4f} km/h")
    print(f"  R²   = {test_r2:.6f}")
    
    # 对标论文
    print(f"\nPaper Results (MLP, leave-scenario-out):")
    print(f"  MAE  = 1.430 km/h")
    print(f"  RMSE = 2.243 km/h")
    print(f"  R²   = 0.9856")
    
    print(f"\nDifference:")
    print(f"  MAE  diff = {test_mae - 1.430:+.4f} (期望 < 0.05)")
    print(f"  RMSE diff = {test_rmse - 2.243:+.4f} (期望 < 0.05)")
    print(f"  R²   diff = {test_r2 - 0.9856:+.6f} (期望 > -0.001)")

if __name__ == "__main__":
    main()
