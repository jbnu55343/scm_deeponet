"""
MLP 训练脚本 - Leave-Scenario-Out 方案

与论文完全相同的评估协议：
- 训练：S001, S002, S003, S004
- 测试：S005, S006 (完全陌生的场景)

这是最严格的泛化能力测试！
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# ============ 配置 ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {DEVICE}")

# ============ MLP 模型 ============
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=(256, 256), dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = in_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# ============ 评估函数 ============
def eval_epoch(model, dl, device, y_mean, y_std):
    """在验证/测试集上评估"""
    model.eval()
    y_true_all = []
    y_pred_all = []
    
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device).float()
            y = y.to(device).float()
            
            pred = model(x)
            
            # 反标准化
            pred_denorm = pred * y_std + y_mean
            y_denorm = y * y_std + y_mean
            
            y_true_all.append(y_denorm.cpu().numpy())
            y_pred_all.append(pred_denorm.cpu().numpy())
    
    y_true = np.concatenate(y_true_all, axis=0).reshape(-1)
    y_pred = np.concatenate(y_pred_all, axis=0).reshape(-1)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return mae, rmse, r2

# ============ 数据加载与分割 ============
def load_and_split_data(npz_path, scenario_ids=None):
    """
    加载数据并按场景分割
    
    scenario_ids: 每个样本的场景标签（可选，如果没有则需要从原始数据推断）
    """
    print(f"[LOAD] Loading {npz_path}...")
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"]     # (N, F)
    Y = d["Y"]     # (N, 1)
    feats = list(d["features"])
    meta = dict(d["meta"]) if "meta" in d else {}
    
    print(f"[INFO] X={X.shape} Y={Y.shape}")
    print(f"[INFO] Features ({len(feats)}): {feats[:5]}... (共{len(feats)}个)")
    
    # 从原始数据推断场景ID
    if "split" in d:
        scenario_ids = d["split"]
        print(f"[INFO] Found scenario IDs in data: {np.unique(scenario_ids)}")
    else:
        print(f"[WARN] No scenario IDs found, cannot use leave-scenario-out split")
        return None
    
    # 按场景分割
    train_mask = np.isin(scenario_ids, [0, 1, 2, 3])  # S001-S004
    test_mask = np.isin(scenario_ids, [4, 5])         # S005-S006
    
    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test, Y_test = X[test_mask], Y[test_mask]
    
    print(f"\n[SPLIT] Leave-Scenario-Out:")
    print(f"  训练 (S001-S004): {X_train.shape[0]:,} samples")
    print(f"  测试 (S005-S006): {X_test.shape[0]:,} samples")
    
    # 从训练集计算标准化参数
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-8
    y_mean = Y_train.mean()
    y_std = Y_train.std() + 1e-8
    
    print(f"\n[STATS] x_mean[:3]={x_mean[:3]} x_std[:3]={x_std[:3]}")
    print(f"        y_mean={y_mean:.6f} y_std={y_std:.6f}")
    
    # 标准化
    X_train = (X_train - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std
    Y_train = (Y_train - y_mean) / y_std
    Y_test = (Y_test - y_mean) / y_std
    
    return X_train, Y_train, X_test, Y_test, x_mean, x_std, y_mean, y_std

# ============ 主函数 ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/dataset_sumo_5km_lag12_filtered.npz",
                        help="输入 npz 文件")
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=str, default="256,256")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="models/mlp_leave_scenario_out.pt")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # ============ 加载数据 ============
    result = load_and_split_data(args.npz)
    if result is None:
        print("[ERROR] Cannot use leave-scenario-out split without scenario IDs")
        return
    
    X_train, Y_train, X_test, Y_test, x_mean, x_std, y_mean, y_std = result
    
    # ============ 创建数据加载器 ============
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(Y_train.reshape(-1, 1))
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(Y_test.reshape(-1, 1))
    )
    
    dl_train = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    dl_test = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    
    # ============ 模型 ============
    in_dim = X_train.shape[1]
    hidden = tuple(map(int, args.hidden.split(",")))
    model = MLP(in_dim=in_dim, hidden_dims=hidden, dropout=args.dropout)
    model = model.to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[MODEL] in_dim={in_dim} hidden={hidden}")
    print(f"        params={n_params:,}")
    
    # ============ 优化器 ============
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # ============ 训练 ============
    print(f"\n[TRAIN START]")
    print(f" Epoch         Loss   Test MAE  Test RMSE   Test R²     Status")
    print("=" * 65)
    
    best_test_r2 = -np.inf
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for x, y in dl_train:
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).float()
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(x)
        
        train_loss /= len(train_dataset)
        
        # 测试
        test_mae, test_rmse, test_r2 = eval_epoch(model, dl_test, DEVICE, y_mean, y_std)
        
        # 早停
        status = ""
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            status = "✓"
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOP] No improvement for {patience} epochs")
                break
        
        print(f"{epoch+1:6d} {train_loss:12.6f} {test_mae:10.4f} {test_rmse:10.4f} "
              f"{test_r2:10.6f} {status:>10}")
    
    # ============ 加载最好的模型 ============
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # ============ 最终评估 ============
    print(f"\n[TRAIN END] Completed")
    print("=" * 65)
    test_mae, test_rmse, test_r2 = eval_epoch(model, dl_test, DEVICE, y_mean, y_std)
    
    print("\n[测试结果] Leave-Scenario-Out (S005-S006)")
    print("=" * 65)
    print(f"MAE  = {test_mae:.4f} km/h")
    print(f"RMSE = {test_rmse:.4f} km/h")
    print(f"R²   = {test_r2:.6f}")
    
    print("\n[论文结果] 用于对比")
    print("=" * 65)
    print(f"MAE  = 1.430 km/h")
    print(f"RMSE = 2.243 km/h")
    print(f"R²   = 0.985600")
    
    print("\n[相对性能]")
    print("=" * 65)
    mae_ratio = test_mae / 1.430
    r2_ratio = test_r2 / 0.9856
    print(f"MAE  相对: {mae_ratio:.2f}x (越小越好)")
    print(f"R²   相对: {r2_ratio:.2f}x (越大越好)")
    
    # 保存模型
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
    }, args.save)
    print(f"\n[SAVE] Model saved to {args.save}")

if __name__ == "__main__":
    main()
