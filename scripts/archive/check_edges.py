import xml.etree.ElementTree as ET
from pathlib import Path

root_dir = Path(__file__).parent
edgedata_path = root_dir / 'scenarios/S001/edgedata.xml'

tree = ET.parse(str(edgedata_path))
root = tree.getroot()
intervals = list(root.iter('interval'))

print(f"Total intervals: {len(intervals)}")
print(f"Sample intervals: {intervals[0].attrib}, {intervals[240].attrib}")

# 找第一个有非零 sampledSeconds 的 edge
for t, interval in enumerate(intervals):
    edges = list(interval.findall('edge'))
    non_zero_edges = [e for e in edges if float(e.attrib.get('sampledSeconds', '0')) > 0]
    
    if non_zero_edges:
        print(f"\nFirst interval with data: interval {t}")
        print(f"Non-zero edges: {len(non_zero_edges)}")
        edge = non_zero_edges[0]
        print(f"Sample edge: {edge.attrib}")
        break
else:
    print("\n❌ No intervals with sampledSeconds > 0 found!")
    print(f"All sampled seconds are likely 0")
    print(f"Checking some edge attributes from middle interval:")
    mid_edges = list(intervals[len(intervals)//2].findall('edge'))[:3]
    for e in mid_edges:
        print(f"  {dict(list(e.attrib.items())[:8])}")
