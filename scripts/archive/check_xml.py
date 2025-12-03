import xml.etree.ElementTree as ET

tree = ET.parse('scenarios/S001/edgedata.xml')
root = tree.getroot()
intervals = list(root.iter('interval'))

print(f"Total intervals: {len(intervals)}")
if intervals:
    edges = list(intervals[0].findall('edge'))
    print(f"Edges in first interval: {len(edges)}")
    if edges:
        print(f"First edge attrib: {edges[0].attrib}")
    else:
        print("No edges found in first interval!")
        print(f"First interval tag: {intervals[0].tag}")
        print(f"First interval attrib: {intervals[0].attrib}")
        print(f"First interval children count: {len(list(intervals[0]))}")
