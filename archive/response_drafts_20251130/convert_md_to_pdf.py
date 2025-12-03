import os
import markdown
from xhtml2pdf import pisa

def convert_md_to_pdf(source_md_file, output_pdf_file):
    # 1. Read Markdown content
    with open(source_md_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Convert Markdown to HTML
    # using extensions for tables, etc. if needed
    html_text = markdown.markdown(text, extensions=['tables'])

    # 3. Add some basic styling for better PDF look
    html_content = f"""
    <html>
    <head>
    <style>
        body {{ font-family: Arial, sans-serif; font-size: 12pt; line-height: 1.5; }}
        h1 {{ font-size: 18pt; color: #2E4053; }}
        h2 {{ font-size: 16pt; color: #2E4053; margin-top: 20px; }}
        h3 {{ font-size: 14pt; color: #2874A6; margin-top: 15px; }}
        code {{ background-color: #F2F3F4; padding: 2px; font-family: Courier New, monospace; }}
        pre {{ background-color: #F2F3F4; padding: 10px; white-space: pre-wrap; }}
        blockquote {{ border-left: 4px solid #ccc; padding-left: 10px; color: #666; }}
    </style>
    </head>
    <body>
    {html_text}
    </body>
    </html>
    """

    # 4. Write HTML to PDF
    with open(output_pdf_file, "wb") as result_file:
        pisa_status = pisa.CreatePDF(
            html_content,                # the HTML to convert
            dest=result_file             # file handle to recieve result
        )

    if pisa_status.err:
        print(f"Error converting {source_md_file}")
    else:
        print(f"Successfully created {output_pdf_file}")

# List of files to convert
files = [
    "Response_to_Reviewer_1.md",
    "Response_to_Reviewer_2.md",
    "Response_to_Reviewer_3.md",
    "Response_to_Reviewer_4.md",
    "Response_to_Reviewer_5.md"
]

current_dir = os.path.dirname(os.path.abspath(__file__))

for filename in files:
    md_path = os.path.join(current_dir, filename)
    pdf_path = os.path.join(current_dir, filename.replace(".md", ".pdf"))
    
    if os.path.exists(md_path):
        convert_md_to_pdf(md_path, pdf_path)
    else:
        print(f"File not found: {md_path}")
