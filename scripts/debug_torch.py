import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

try:
    X = torch.randn(100, 2484).cuda()
    Y = torch.randn(100, 1).cuda()
    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=16)

    model = nn.Sequential(nn.Linear(2484, 128), nn.ReLU(), nn.Linear(128, 1)).cuda()
    opt = torch.optim.Adam(model.parameters())
    crit = nn.MSELoss()

    print("Start training loop...")
    for i, (x, y) in enumerate(dl):
        pred = model(x)
        loss = crit(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"Batch {i} done")
    print("End training loop")
except Exception as e:
    print(f"Error: {e}")
