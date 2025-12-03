import xml.etree.ElementTree as ET

xml_file = r'D:\pro_and_data\SCM_DeepONet_code\scenarios\S001\edgedata.xml'

parser = ET.iterparse(xml_file, events=('start', 'end'))
count = 0

for event, elem in parser:
    if event == 'end' and elem.tag == 'edge':
        count += 1
        if count == 1:
            print(f"First edge attributes: {list(elem.attrib.keys())}")
            break
    
    # 清理以节省内存
    elem.clear()

print(f"Checked {count} edges")
