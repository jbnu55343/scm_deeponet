import docx
import os

def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        return f"Error reading {file_path}: {str(e)}"

files = ['rev3.docx', 'rev4.docx', 'rev5.docx']
base_dir = r'd:\pro_and_data\SCM_DeepONet_code\data-3951152'

output_file = r'd:\pro_and_data\SCM_DeepONet_code\docs\extracted_reviews.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    for file_name in files:
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            print(f"Reading {file_name}...")
            content = read_docx(file_path)
            f.write(f"--- Start of {file_name} ---\n")
            f.write(content)
            f.write(f"\n--- End of {file_name} ---\n\n")
        else:
            print(f"File not found: {file_path}")
            f.write(f"File not found: {file_name}\n\n")

print(f"Extraction complete. Saved to {output_file}")
