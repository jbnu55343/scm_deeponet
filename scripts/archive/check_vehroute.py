import xml.etree.ElementTree as ET

# 检查 vehroute.xml 结构
tree = ET.parse('scenarios/S001/vehroute.xml')
root = tree.getroot()

vehicles = list(root.findall('vehicle'))
print(f"Total vehicles in vehroute.xml: {len(vehicles)}")

if vehicles:
    v = vehicles[0]
    routes = list(v.findall('route'))
    print(f"First vehicle routes: {len(routes)}")
    if routes:
        print(f"Route edges: {routes[0].attrib.get('edges', 'No edges')[:100]}...")
