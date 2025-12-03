import os
import json
import glob
import pandas as pd

def main():
    log_dir = 'results/metr_la'
    log_files = glob.glob(os.path.join(log_dir, '*_final.log'))
    
    results = []
    
    for log_file in log_files:
        model_name = os.path.basename(log_file).replace('_final.log', '')
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                json_line = None
                for line in lines:
                    if 'FINAL_RESULT_JSON:' in line:
                        json_line = line.strip().split('FINAL_RESULT_JSON:')[1]
                        break
                
                if json_line:
                    res = json.loads(json_line)
                    # Ensure model name is consistent or use the one from filename
                    if 'model' not in res:
                        res['model'] = model_name
                    results.append(res)
                else:
                    print(f"Warning: No JSON result found in {log_file}")
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            
    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)
    # Reorder columns
    cols = ['model', 'test_mae', 'test_rmse', 'test_r2', 'time_sec', 'params']
    df = df[cols]
    
    print("\n=== FINAL RESULTS COMPARISON ===")
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv(os.path.join(log_dir, 'comparison_summary.csv'), index=False)
    print(f"\nSaved summary to {os.path.join(log_dir, 'comparison_summary.csv')}")

if __name__ == '__main__':
    main()
