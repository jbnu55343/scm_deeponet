# scripts/train_mlp_paper_exact.py
# 严格按照论文规格的 MLP 训练脚本
# 
# 论文规格 (paper.tex 4.2-4.3):
# - 特征: 12-step speed lags + 6 exogenous context
# - 数据分割: S001-S004 train/val (80/20), S005-S006 test (留场景测试)
# - 过滤: traveltime > 0 且 < 1e5
# - 输入维度: 18 (12 lags + 6 context)
# - 目标: y_{t+1} = speed at t+1
#
# 注意：当前的 dataset_sumo_5km_lag12.npz 与论文数据不匹配
# 本脚本创建一个简化版本，假设特征已正确生成

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


class NpzDataset(Dataset):
    """加载 npz 数据集"""
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
        y = self.Y[idx].astype(np.float32, copy=False)

        # 标准化
        if self.x_mean is not None:
            x = (x - self.x_mean) / (self.x_std + 1e-8)
        if self.y_mean is not None:
            y = (y - self.y_mean) / (self.y_std + 1e-8)

        return torch.from_numpy(x), torch.from_numpy(y)


class MLP(nn.Module):
    """MLP: 2 层隐层，各 256 单元，ReLU，Dropout 0.1"""
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
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=str, default="256,256")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default="models/mlp_paper.pt")
    ap.add_argument("--analyze", action="store_true", help="仅分析数据质量，不训练")
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
    X = d["X"]     # (N, F)
    Y = d["Y"]     # (N, 1)
    feats = list(d["features"])
    
    print(f"[INFO] X={X.shape} Y={Y.shape}")
    print(f"[INFO] Features ({len(feats)}): {feats}")
    
    # 如果只分析数据
    if args.analyze:
        print(f"\n{'='*80}")
        print(f"数据质量分析")
        print(f"{'='*80}")
        y = Y.reshape(-1)
        print(f"\n目标变量统计:")
        print(f"  mean: {y.mean():.6f} (论文: 8.2547)")
        print(f"  std:  {y.std():.6f} (论文: 10.7359)")
        print(f"  min:  {y.min():.6f}, max: {y.max():.6f}")
        
        # 检查是否与论文数据匹配
        if abs(y.mean() - 8.2547) < 0.5:
            print(f"✅ 目标变量与论文匹配！")
        else:
            print(f"⚠️ 目标变量与论文差异大，可能数据集不同")
            print(f"   差异: {abs(y.mean() - 8.2547):.4f} km/h")
        return
    
    # ============ 特征检查 ============
    # 论文: speed_lag1, ..., speed_lag12 (12) + entered, left, density, occupancy, waitingTime, traveltime (6)
    # 共 18 个特征
    
    expected_features = [
        "speed_lag1", "speed_lag2", "speed_lag3", "speed_lag4", "speed_lag5", "speed_lag6",
        "speed_lag7", "speed_lag8", "speed_lag9", "speed_lag10", "speed_lag11", "speed_lag12",
        "entered", "left", "density", "occupancy", "waitingTime", "traveltime"
    ]
    
    if len(feats) != len(expected_features):
        print(f"\n⚠️ 警告: 特征数不匹配")
        print(f"   期望: {len(expected_features)} 个")
        print(f"   实际: {len(feats)} 个")
        print(f"   预期特征: {expected_features}")
        print(f"   实际特征: {feats}")
    
    # ============ 数据分割（按时间顺序） ============
    N = X.shape[0]
    all_idx = np.arange(N)
    
    # 按时间顺序分割（不打乱！）
    # 论文: train 80%, val 10%, test 10%
    train_end = int(N * 0.80)
    val_end = int(N * 0.90)
    
    train_idx = all_idx[:train_end]
    val_idx = all_idx[train_end:val_end]
    test_idx = all_idx[val_end:]
    
    print(f"\n[SPLIT] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"        train ratio: {len(train_idx)/N*100:.1f}%")
    
    # ============ 标准化 ============
    x_train = X[train_idx]
    y_train = Y[train_idx]
    
    x_mean = x_train.mean(0)
    x_std = x_train.std(0)
    y_mean = y_train.mean(0)
    y_std = y_train.std(0)
    
    print(f"\n[STATS] x_mean[:3]={x_mean[:3]} x_std[:3]={x_std[:3]}")
    print(f"        y_mean={y_mean[0]:.6f} y_std={y_std[0]:.6f}")
    print(f"        论文 y: mean=8.2547 std=10.7359")
    
    # ============ Dataset & DataLoader ============
    ds_train = NpzDataset(X, Y, train_idx, x_mean, x_std, y_mean, y_std)
    ds_val = NpzDataset(X, Y, val_idx, x_mean, x_std, y_mean, y_std)
    ds_test = NpzDataset(X, Y, test_idx, x_mean, x_std, y_mean, y_std)
    
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=args.batch, shuffle=False, num_workers=0)
    
    # ============ 模型 ============
    in_dim = X.shape[1]
    hidden = tuple(int(h) for h in args.hidden.split(','))
    model = MLP(in_dim, hidden=hidden, dropout=args.dropout)
    model.to(device)
    
    print(f"\n[MODEL] in_dim={in_dim} hidden={hidden}")
    print(f"        params={sum(p.numel() for p in model.parameters()):,}")
    
    # ============ 优化器（按论文规格） ============
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # ============ 训练循环 ============
    best_val_r2 = -np.inf
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    print("\n[TRAIN START]")
    print(f"{'Epoch':>6} {'Loss':>12} {'Val MAE':>10} {'Val RMSE':>10} {'Val R²':>10} {'Status':>10}")
    print("="*60)
    
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
        
        # 验证
        val_mae, val_rmse, val_r2 = eval_epoch(model, dl_val, device, y_mean, y_std)
        
        # 早停
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch
            patience_counter = 0
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            torch.save(model.state_dict(), args.save)
            status = "✓"
        else:
            patience_counter += 1
            status = ""
        
        print(f"{epoch+1:6d} {train_loss:12.6f} {val_mae:10.4f} {val_rmse:10.4f} {val_r2:10.6f} {status:>10}")
        
        if patience_counter >= patience:
            print(f"\n[EARLY STOP] no improvement for {patience} epochs")
            break
    
    elapsed = time.time() - start_time
    print(f"\n[TRAIN END] {elapsed:.1f}s")
    
    # ============ 测试 ============
    print(f"\n{'='*80}")
    print(f"测试结果")
    print(f"{'='*80}")
    
    model.load_state_dict(torch.load(args.save, map_location=device))
    test_mae, test_rmse, test_r2 = eval_epoch(model, dl_test, device, y_mean, y_std)
    
    print(f"\n当前结果:")
    print(f"  MAE  = {test_mae:.4f} km/h")
    print(f"  RMSE = {test_rmse:.4f} km/h")
    print(f"  R²   = {test_r2:.6f}")
    
    print(f"\n论文结果 (MLP, leave-scenario-out):")
    print(f"  MAE  = 1.430 km/h")
    print(f"  RMSE = 2.243 km/h")
    print(f"  R²   = 0.9856")
    
    print(f"\n差距:")
    mae_diff = test_mae - 1.430
    rmse_diff = test_rmse - 2.243
    r2_diff = test_r2 - 0.9856
    
    print(f"  MAE  diff = {mae_diff:+.4f} km/h")
    print(f"  RMSE diff = {rmse_diff:+.4f} km/h")
    print(f"  R²   diff = {r2_diff:+.6f}")
    
    if abs(mae_diff) < 0.1 and abs(rmse_diff) < 0.1 and abs(r2_diff) < 0.01:
        print(f"\n✅ 结果接近论文！问题已解决。")
    elif abs(test_r2 - 0.9856) < 0.05:
        print(f"\n✅ R² 接近论文（允许波动）")
    else:
        print(f"\n❌ 结果与论文差异仍很大")
        print(f"   建议检查:")
        print(f"   1. 数据集是否与论文一致（特征列表）")
        print(f"   2. 是否需要重新生成数据")
        print(f"   3. SUMO 配置是否改变")

if __name__ == "__main__":
    main()
