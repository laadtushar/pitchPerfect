
import xml.etree.ElementTree as ET
import re

def parse_bounds(bounds_str):
    # bounds="[x1,y1][x2,y2]"
    match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds_str)
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return center_x, center_y
    return None

def find_element_by_desc(xml_file, content_desc_str):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Helper to recursively find
        # We can also use xpath if supported, or generic iteration
        for node in root.iter():
            desc = node.attrib.get('content-desc', '')
            if content_desc_str.lower() in desc.lower():
                bounds = node.attrib.get('bounds', '')
                if bounds:
                    return desc, parse_bounds(bounds)
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
    return None, None

def validate():
    print("--- Validating Like Button ---")
    desc, coords = find_element_by_desc('window_dump.xml', 'Like photo')
    if coords:
        print(f"SUCCESS: Found '{desc}' at {coords}")
        print(f"Expected: (918, 1398)")
    else:
        print("FAILURE: Could not find 'Like photo'")

    print("\n--- Validating Send Like Button ---")
    # In the dialog dump, we saw 'Send priority like'
    desc, coords = find_element_by_desc('window_dump_dialog.xml', 'Send priority like')
    if coords:
        print(f"SUCCESS: Found '{desc}' at {coords}")
    else:
        print("FAILURE: Could not find 'Send priority like'")
        
    print("\n--- Validating Send Button (Generic) ---")
    desc, coords = find_element_by_desc('window_dump_dialog.xml', 'Send')
    if coords:
         print(f"SUCCESS: Found '{desc}' at {coords}")

if __name__ == "__main__":
    validate()
