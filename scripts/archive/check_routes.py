import xml.etree.ElementTree as ET

tree = ET.parse(r'D:\pro_and_data\SCM_DeepONet_code\scenarios\S001\routes.rou.alt.xml')
root = tree.getroot()

# 计数所有元素
for tag in {'vehicle', 'trip', 'route'}:
    elements = root.findall('.//' + tag)
    print(f"{tag}: {len(elements)}")
    if elements:
        print(f"  First {tag}: {dict(list(elements[0].attrib.items())[:3])}")
