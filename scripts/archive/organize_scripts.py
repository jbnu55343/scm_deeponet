import os
import shutil

def main():
    scripts_dir = 'scripts'
    archive_dir = os.path.join(scripts_dir, 'archive')
    
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
        print(f"Created {archive_dir}")
        
    # Files to KEEP (Core)
    keep_files = {
        'train_mlp_sumo_std.py',
        'train_deeponet_sumo_std.py',
        'train_transformer_sumo_std.py',
        'create_standard_split.py',
        'preprocess_metr_la.py',
        'train_deeponet_metr_la.py',
        'train_transformer_metr_la.py',
        'collect_results.py',
        'inspect_sumo_data.py',
        'inspect_sumo_features.py',
        'inspect_filtered_base.py',
        # Files to be renamed (keep them for now, rename later)
        'train_mlp_speed.py',
        'train_gnn_full.py'
    }
    
    # List all files in scripts
    all_files = [f for f in os.listdir(scripts_dir) if os.path.isfile(os.path.join(scripts_dir, f))]
    
    for f in all_files:
        if f not in keep_files:
            src = os.path.join(scripts_dir, f)
            dst = os.path.join(archive_dir, f)
            try:
                shutil.move(src, dst)
                print(f"Archived: {f}")
            except Exception as e:
                print(f"Error archiving {f}: {e}")
                
    # Rename files
    renames = {
        'train_mlp_speed.py': 'train_mlp_metr_la.py',
        'train_gnn_full.py': 'train_gnn_metr_la.py'
    }
    
    for old_name, new_name in renames.items():
        old_path = os.path.join(scripts_dir, old_name)
        new_path = os.path.join(scripts_dir, new_name)
        if os.path.exists(old_path):
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_name} -> {new_name}")
            except Exception as e:
                print(f"Error renaming {old_name}: {e}")
        else:
            print(f"Skipped rename: {old_name} not found")

if __name__ == '__main__':
    main()
