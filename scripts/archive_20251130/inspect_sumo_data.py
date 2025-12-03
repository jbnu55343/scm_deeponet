import numpy as np

files = [
    'data/dataset_sumo_5km_lag12_no_spatial.npz',
    'data/dataset_sumo_5km_lag12_filtered_with_spatial.npz'
]

for f in files:
    try:
        data = np.load(f, allow_pickle=True)
        print(f"=== {f} ===")
        print("Keys:", data.files)
        if 'X' in data: print("X shape:", data['X'].shape)
        if 'Y' in data: print("Y shape:", data['Y'].shape)
        if 'split' in data: 
            split = data['split'].item()
            print("Split keys:", split.keys())
            for k in split:
                print(f"  {k}: {len(split[k])}")
    except Exception as e:
        print(f"Error loading {f}: {e}")
