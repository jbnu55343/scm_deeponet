import xml.etree.ElementTree as ET
from pathlib import Path

root_dir = Path(r"D:\pro_and_data\SCM_DeepONet_code")
edgedata_path = root_dir / 'scenarios/S001/edgedata.xml'

print(f"Reading {edgedata_path}")
tree = ET.parse(str(edgedata_path))
root = tree.getroot()

# 只获取第一个 interval 并检查
interval = root.find('interval')
if interval is not None:
    edges = interval.findall('edge')
    print(f"Edges in first interval: {len(edges)}")
    
    # 找有流量的边
    for e in edges[:100]:
        speed = float(e.attrib.get('speed', '0'))
        if speed > 0:
            print(f"Found edge with speed > 0: {e.attrib.get('id')}: speed={speed}")
            break
    else:
        print("No edges with speed > 0 in first 100 edges")
        print(f"Sample: {edges[0].attrib}")
