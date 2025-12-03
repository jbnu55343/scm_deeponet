import xml.etree.ElementTree as ET
from pathlib import Path

scenarios_dir = Path('scenarios')
for sdir in sorted(scenarios_dir.glob('S*/edgedump.add.xml')):
    # 读取文件
    with open(sdir, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换 excludeEmpty="true" 为 excludeEmpty="false"
    if 'excludeEmpty="true"' in content:
        new_content = content.replace('excludeEmpty="true"', 'excludeEmpty="false"')
        with open(sdir, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: {sdir}")
    else:
        print(f"Already updated or not found: {sdir}")
