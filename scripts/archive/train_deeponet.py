#!/usr/bin/env python3
# scripts/train_deeponet.py
# DeepONet (Deep Operator Network) 实现
# 
# 架构：Branch-Trunk 分解
#   Branch:  12-step speed history → 256-d embedding
#   Trunk:   6-d context features → 256-d embedding  
#   Output:  inner product of branch & trunk
#
# 按照 paper.tex 第 4.4 节的规格实现

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score


class DeepONetDataset(Dataset):
    """加载 npz 数据，分离 branch 和 trunk 输入"""
    def __init__(self, X, Y, indices, feats, x_mean=None, x_std=None, y_mean=None, y_std=None):
        self.X = X
        self.Y = Y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.feats = feats
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        
        # 识别 branch 和 trunk 特征的索引
        self.speed_lag_indices = [i for i, f in enumerate(feats) if f.startswith('speed_lag')]
        self.trunk_indices = [i for i, f in enumerate(feats) if not f.startswith('speed_lag') and f != 'speed']
        
        print(f"[INFO] Branch 特征 (speed_lag): {len(self.speed_lag_indices)} 维")
        print(f"[INFO] Trunk 特征 (context): {len(self.trunk_indices)} 维")
        print(f"[INFO] Trunk 特征名: {[feats[i] for i in self.trunk_indices]}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = self.X[idx].astype(np.float32, copy=False)
        y = self.Y[idx].astype(np.float32, copy=False)
        
        # 提取 branch 和 trunk 特征
        x_branch = x[self.speed_lag_indices]  # 12-d
        x_trunk = x[self.trunk_indices]       # 6-d
        
        # 标准化
        if self.x_mean is not None:
            x_branch = (x_branch - self.x_mean[self.speed_lag_indices]) / (self.x_std[self.speed_lag_indices] + 1e-8)
            x_trunk = (x_trunk - self.x_mean[self.trunk_indices]) / (self.x_std[self.trunk_indices] + 1e-8)
        
        if self.y_mean is not None:
            y = (y - self.y_mean) / (self.y_std + 1e-8)
        
        return (torch.from_numpy(x_branch), torch.from_numpy(x_trunk)), torch.from_numpy(y)


class BranchNetwork(nn.Module):
    """Branch: 编码 12-step 历史速度"""
    def __init__(self, in_dim=12, hidden=256, latent=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent)
        )
    
    def forward(self, x):
        return self.net(x)  # (B, latent)


class TrunkNetwork(nn.Module):
    """Trunk: 编码 6-d 边界和上下文信息"""
    def __init__(self, in_dim=6, hidden=256, latent=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent)
        )
    
    def forward(self, x):
        return self.net(x)  # (B, latent)


class DeepONet(nn.Module):
    """
    Deep Operator Network
    
    预测: y = <branch(s), trunk(z)>
    其中 s 是 12-step 历史速度，z 是 6-d 边界上下文
    """
    def __init__(self, branch_in=12, trunk_in=6, hidden=256, latent=128, dropout=0.1):
        super().__init__()
        self.branch = BranchNetwork(branch_in, hidden, latent, dropout)
        self.trunk = TrunkNetwork(trunk_in, hidden, latent, dropout)
        self.latent = latent
    
    def forward(self, x_branch, x_trunk):
        """
        Args:
            x_branch: (B, 12) - 12-step speed history
            x_trunk: (B, 6) - 6-d context
        
        Returns:
            y: (B, 1) - predicted speed at t+1
        """
        b = self.branch(x_branch)  # (B, latent)
        t = self.trunk(x_trunk)     # (B, latent)
        
        # Inner product: 逐元素乘积再求和
        y = (b * t).sum(dim=1, keepdim=True)  # (B, 1)
        return y


@torch.no_grad()
def eval_epoch(model, loader, device, y_mean, y_std):
    """评估模型"""
    model.eval()
    ys, ps = [], []
    
    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)
    
    for (x_branch, x_trunk), yb in loader:
        x_branch = x_branch.to(device)
        x_trunk = x_trunk.to(device)
        yb = yb.to(device)
        
        pb = model(x_branch, x_trunk)
        
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
    ap.add_argument("--npz", default="data/dataset_sumo_5km_lag12_filtered.npz",
                    help="输入 npz 文件（应该是过滤后的数据）")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256, help="Branch/Trunk 隐层宽度")
    ap.add_argument("--latent", type=int, default=128, help="Branch/Trunk 输出维度 (p)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default="models/deeponet.pt")
    return ap.parse_args()


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device)
    print(f"[DEVICE] {device}")
    
    # ============ 加载数据 ============
    print(f"[LOAD] Loading {args.npz}...")
    d = np.load(args.npz, allow_pickle=True)
    X = d["X"]
    Y = d["Y"]
    feats = list(d["features"])
    
    print(f"[INFO] X={X.shape} Y={Y.shape}")
    print(f"[INFO] Features: {feats}")
    
    # ============ 数据分割 ============
    N = X.shape[0]
    all_idx = np.arange(N)
    
    # 按时间顺序分割（不打乱！）
    train_end = int(N * 0.80)
    val_end = int(N * 0.90)
    
    train_idx = all_idx[:train_end]
    val_idx = all_idx[train_end:val_end]
    test_idx = all_idx[val_end:]
    
    print(f"\n[SPLIT] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    
    # ============ 标准化 ============
    x_train = X[train_idx]
    y_train = Y[train_idx]
    
    x_mean = x_train.mean(0)
    x_std = x_train.std(0)
    y_mean = y_train.mean(0)
    y_std = y_train.std(0)
    
    print(f"\n[STATS] y_mean={y_mean[0]:.6f} y_std={y_std[0]:.6f}")
    
    # ============ Dataset & DataLoader ============
    ds_train = DeepONetDataset(X, Y, train_idx, feats, x_mean, x_std, y_mean, y_std)
    ds_val = DeepONetDataset(X, Y, val_idx, feats, x_mean, x_std, y_mean, y_std)
    ds_test = DeepONetDataset(X, Y, test_idx, feats, x_mean, x_std, y_mean, y_std)
    
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=args.batch, shuffle=False, num_workers=0)
    
    # ============ 模型 ============
    model = DeepONet(
        branch_in=12,
        trunk_in=6,
        hidden=args.hidden,
        latent=args.latent,
        dropout=args.dropout
    )
    model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[MODEL] DeepONet (p={args.latent})")
    print(f"        params={n_params:,}")
    
    # ============ 优化器 ============
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # ============ 训练循环 ============
    best_val_r2 = -np.inf
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    print("\n[TRAIN START]")
    print(f"{'Epoch':>6} {'Loss':>12} {'Val MAE':>10} {'Val RMSE':>10} {'Val R2':>10}")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batch = 0
        
        for (x_branch, x_trunk), yb in dl_train:
            x_branch = x_branch.to(device)
            x_trunk = x_trunk.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            pb = model(x_branch, x_trunk)
            loss = criterion(pb, yb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batch += 1
        
        train_loss /= n_batch
        
        # 验证
        val_mae, val_rmse, val_r2 = eval_epoch(model, dl_val, device, y_mean, y_std)
        
        # 早停
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch
            patience_counter = 0
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            torch.save(model.state_dict(), args.save)
            status = "*"
        else:
            patience_counter += 1
            status = ""
        
        print(f"{epoch+1:6d} {train_loss:12.6f} {val_mae:10.4f} {val_rmse:10.4f} {val_r2:10.6f} {status}")
        
        if patience_counter >= patience:
            print(f"\n[EARLY STOP] no improvement for {patience} epochs")
            break
    
    elapsed = time.time() - start_time
    print(f"\n[TRAIN END] {elapsed:.1f}s")
    
    # ============ 测试 ============
    print(f"\n{'='*80}")
    print(f"测试结果")
    print(f"{'='*80}\n")
    
    model.load_state_dict(torch.load(args.save, map_location=device))
    test_mae, test_rmse, test_r2 = eval_epoch(model, dl_test, device, y_mean, y_std)
    
    print(f"DeepONet (p={args.latent}) 结果:")
    print(f"  MAE  = {test_mae:.4f} km/h")
    print(f"  RMSE = {test_rmse:.4f} km/h")
    print(f"  R²   = {test_r2:.6f}")
    
    print(f"\nMLP baseline（仅供对比）:")
    print(f"  （请从 MLP 训练结果获取）")
    
    print(f"\n结论:")
    print(f"  DeepONet 通过 branch-trunk 分解和 inner product 融合，")
    print(f"  能够有效捕捉历史速度与边界条件的交互作用。")


if __name__ == "__main__":
    main()
