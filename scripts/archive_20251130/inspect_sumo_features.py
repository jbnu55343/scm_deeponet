import numpy as np

f = 'data/dataset_sumo_5km_lag12_filtered_with_spatial.npz'
data = np.load(f, allow_pickle=True)
print(f"Features in {f}:")
print(data['features'])

f2 = 'data/dataset_sumo_5km_lag12_no_spatial.npz'
data2 = np.load(f2, allow_pickle=True)
print(f"Features in {f2}:")
print(data2['features'])
