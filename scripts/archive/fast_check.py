import xml.etree.ElementTree as ET
import sys

# 只扫描前几个 interval 来快速检查
xml_file = r"D:\pro_and_data\SCM_DeepONet_code\scenarios\S001\edgedata.xml"
target_features = {'speed', 'entered', 'left', 'density', 'occupancy', 'waitingTime', 'traveltime'}

tree = ET.iterparse(xml_file, events=('end',))

interval_count = 0
edge_count = 0
non_zero_count = 0

for event, elem in tree:
    if elem.tag == 'interval':
        interval_count += 1
        
        # 只检查前 10 个 interval
        if interval_count > 10:
            break
        
        for edge in elem.findall('edge'):
            edge_count += 1
            for feat in target_features:
                val = float(edge.attrib.get(feat, '0'))
                if val > 0:
                    non_zero_count += 1
                    print(f"Found non-zero: interval={interval_count}, edge={edge.attrib.get('id')}, {feat}={val}")
                    break  # 只打印每条边的第一个非零特征
        
        # 清空 element 以节省内存
        elem.clear()

print(f"\nProcessed: {interval_count} intervals, {edge_count} edges, {non_zero_count} with non-zero values")
