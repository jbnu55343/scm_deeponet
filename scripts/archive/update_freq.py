from pathlib import Path

scenarios_dir = Path('scenarios')
for sdir in sorted(scenarios_dir.glob('S*')):
    config_file = sdir / 'edgedump.add.xml'
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换 freq
        new_content = content.replace('freq="60"', 'freq="10"')
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {config_file.name} (freq changed to 10)")
