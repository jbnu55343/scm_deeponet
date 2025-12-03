import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class StdLogger:
    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.start_time = time.time()
        print(f"[{self.model_name}] Starting training on {self.dataset_name}...")

    def log_config(self, args):
        print(f"[{self.model_name}] Config: {vars(args)}")

    def log_epoch(self, epoch, total_epochs, train_loss, val_loss, val_r2):
        print(f"[Epoch {epoch}/{total_epochs}] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val R2: {val_r2:.4f}")

    def log_result(self, y_true, y_pred, train_time=None):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        if train_time is None:
            train_time = time.time() - self.start_time

        print("-" * 30)
        print(f"[Result] Model: {self.model_name}")
        print(f"[Result] Dataset: {self.dataset_name}")
        print(f"[Result] Test MSE: {mse:.6f}")
        print(f"[Result] Test MAE: {mae:.6f}")
        print(f"[Result] Test R2: {r2:.4f}")
        print(f"[Result] Training Time: {train_time:.2f}s")
        print("-" * 30)
        return mse, mae, r2

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
