import torch
import time

print("Checking GPU...")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print("Matrix multiplication successful.")
else:
    print("No GPU found.")
