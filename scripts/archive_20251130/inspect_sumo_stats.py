import numpy as np
import os

def main():
    npz_path = 'data/dataset_sumo_5km_lag12_filtered_with_spatial.npz'
    output_path = 'scripts/sumo_stats_report.txt'
    
    try:
        print(f"Loading {npz_path}...")
        d = np.load(npz_path, allow_pickle=True)
        Y = d['Y']
        X = d['X']
        
        with open(output_path, 'w') as f:
            f.write(f"Dataset: {npz_path}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Shape X: {X.shape}\n")
            f.write(f"Shape Y: {Y.shape}\n")
            f.write("-" * 50 + "\n")
            
            # Y Statistics
            f.write("\n=== Target (Y) Statistics ===\n")
            f.write(f"Mean: {Y.mean():.4f}\n")
            f.write(f"Std:  {Y.std():.4f}\n")
            f.write(f"Min:  {Y.min():.4f}\n")
            f.write(f"Max:  {Y.max():.4f}\n")
            
            # Zero analysis
            zeros = (Y < 0.1).sum()
            # Use size for total elements if Y is multidimensional, or len if 1D. Assuming 1D or flattened interest.
            f.write(f"Zeros (< 0.1): {zeros} ({zeros/Y.size:.2%})\n")
            
            # Histogram of Y
            hist, bins = np.histogram(Y, bins=50)
            f.write("\n=== Y Histogram ===\n")
            for i in range(len(hist)):
                if hist[i] > 0:
                    f.write(f"{bins[i]:.1f} - {bins[i+1]:.1f}: {hist[i]}\n")

            # Feature Statistics (Sample)
            f.write("\n=== Feature Statistics (First 5 cols) ===\n")
            # 0: speed(t), 1: entered, 2: left, 3: density, 4: occupancy
            feature_names = ['speed(t)', 'entered', 'left', 'density', 'occupancy']
            for i in range(min(5, X.shape[1])):
                col = X[:, i]
                f.write(f"Feature {i} ({feature_names[i] if i < len(feature_names) else ''}): Mean={col.mean():.4f}, Std={col.std():.4f}, Min={col.min():.4f}, Max={col.max():.4f}\n")
                zeros_feat = (col < 0.1).sum()
                f.write(f"  Zeros: {zeros_feat} ({zeros_feat/len(col):.2%})\n")
        
        print(f"Successfully wrote stats to {output_path}")

    except Exception as e:
        with open(output_path, 'w') as f:
            f.write(f"Error running script: {str(e)}")
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
