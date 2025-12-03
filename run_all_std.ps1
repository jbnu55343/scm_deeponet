$env:PYTHONPATH = "D:\pro_and_data\SCM_DeepONet_code"
$python = "D:\DL\envs\pytorch_gpu\python.exe"
$npz = "data/dataset_sumo_5km_lag12_nonzero.npz"
$split = "data/sumo_split_nonzero.json"

Write-Host "Training MLP..."
& $python -m scripts.train_mlp_sumo_std --npz $npz --split $split --save models/mlp_sumo_nonzero.pth

Write-Host "Training DeepONet..."
& $python -m scripts.train_deeponet_sumo_std --npz $npz --split $split --save models/deeponet_sumo_nonzero.pth

Write-Host "Training Transformer..."
& $python -m scripts.train_transformer_sumo_std --npz $npz --split $split --save models/transformer_sumo_nonzero.pth

Write-Host "Training LSTM..."
& $python -m scripts.train_lstm_sumo_std --npz $npz --split $split --save models/lstm_sumo_nonzero.pth

Write-Host "Training GNN..."
& $python -m scripts.train_gnn_sumo_std --npz $npz --split $split --save models/gnn_sumo_nonzero.pth

Write-Host "All training completed."
