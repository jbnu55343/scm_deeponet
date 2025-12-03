import zipfile
import xml.etree.ElementTree as ET
import os

def extract_text_from_docx(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            
            # Define the namespace map
            namespaces = {
                'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
            }
            
            text_parts = []
            # Find all paragraph elements
            for p in tree.findall('.//w:p', namespaces):
                paragraph_text = []
                # Find all text runs within the paragraph
                for r in p.findall('.//w:r', namespaces):
                    # Find text elements
                    for t in r.findall('.//w:t', namespaces):
                        if t.text:
                            paragraph_text.append(t.text)
                
                if paragraph_text:
                    text_parts.append(''.join(paragraph_text))
            
            return '\n'.join(text_parts)
    except Exception as e:
        return f"Error reading {docx_path}: {str(e)}"

base_dir = 'data-3951152'
review_files = ['rev1.docx', 'rev2.docx', 'rev3.docx', 'rev4.docx', 'rev5.docx']

print("--- EXTRACTED REVIEWS ---")
for rev_file in review_files:
    file_path = os.path.join(base_dir, rev_file)
    if os.path.exists(file_path):
        print(f"\n=== {rev_file} ===")
        content = extract_text_from_docx(file_path)
        print(content)
    else:
        print(f"\n=== {rev_file} NOT FOUND ===")
