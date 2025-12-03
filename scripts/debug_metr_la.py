import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import sys

def load_npz(path):
    print(f"Loading {path}...")
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    Y = d["Y"]
    return X, Y

try:
    X, Y = load_npz('data/metr_la_lag12_temporal.npz')
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"X sample: {X[0]}")
    print(f"Y sample: {Y[0]}")
    
    # Split
    N = len(X)
    train_idx = int(0.7 * N)
    X_train, Y_train = X[:train_idx], Y[:train_idx]
    X_test, Y_test = X[train_idx:], Y[train_idx:]
    
    # Persistence
    # Assuming col 0 is lag1
    p_pred = X_test[:, 0]
    r2 = r2_score(Y_test, p_pred)
    print(f"Persistence R2 (col 0): {r2}")
    
    # Ridge
    model = Ridge()
    model.fit(X_train, Y_train)
    r_pred = model.predict(X_test)
    print(f"Ridge R2: {r2_score(Y_test, r_pred)}")
    
except Exception as e:
    print(f"Error: {e}")
