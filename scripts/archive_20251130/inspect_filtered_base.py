import numpy as np

f = 'data/dataset_sumo_5km_lag12_filtered.npz'
try:
    data = np.load(f, allow_pickle=True)
    print(f"=== {f} ===")
    print("Keys:", data.files)
    if 'split' in data:
        split = data['split'].item()
        print("Split keys:", split.keys())
        for k in split:
            print(f"  {k}: {len(split[k])}")
except Exception as e:
    print(f"Error: {e}")
