#!/usr/bin/env python3
"""
PDF Generator for Virtual Interpreter Case Study
Converts markdown to a simple, Google Docs-compatible PDF
"""

import markdown
from weasyprint import HTML, CSS
import os
from pathlib import Path

def create_pdf_from_markdown(markdown_file, output_pdf):
    """Convert markdown file to PDF with simple, compatible formatting"""
    
    # Read markdown content
    with open(markdown_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'codehilite'])
    
    # Create simple HTML document with minimal styling
    html_document = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Virtual Interpreter Case Study</title>
        <style>
            @page {{
                margin: 1in;
                size: A4;
            }}
            
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.5;
                color: #000;
                max-width: 100%;
                margin: 0;
                padding: 20px;
            }}
            
            h1 {{
                color: #000;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 20px;
                text-align: left;
            }}
            
            h2 {{
                color: #000;
                font-size: 20px;
                font-weight: bold;
                margin-top: 25px;
                margin-bottom: 10px;
            }}
            
            h3 {{
                color: #000;
                font-size: 16px;
                font-weight: bold;
                margin-top: 20px;
                margin-bottom: 8px;
            }}
            
            h4 {{
                color: #000;
                font-size: 14px;
                font-weight: bold;
                margin-top: 15px;
                margin-bottom: 6px;
            }}
            
            p {{
                margin-bottom: 10px;
                text-align: left;
            }}
            
            ul, ol {{
                margin-bottom: 10px;
                padding-left: 20px;
            }}
            
            li {{
                margin-bottom: 3px;
            }}
            
            strong {{
                font-weight: bold;
            }}
            
            code {{
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                padding: 1px 3px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }}
            
            pre {{
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                padding: 10px;
                margin: 10px 0;
                overflow-x: auto;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                line-height: 1.3;
            }}
            
            pre code {{
                background: none;
                border: none;
                padding: 0;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Generate PDF
    HTML(string=html_document).write_pdf(output_pdf)
    print(f"✅ PDF generated successfully: {output_pdf}")

def main():
    """Main function to generate PDF"""
    
    # File paths
    markdown_file = "CASE_STUDY.md"
    output_pdf = "Virtual_Interpreter_Case_Study.pdf"
    
    # Check if markdown file exists
    if not os.path.exists(markdown_file):
        print(f"❌ Error: {markdown_file} not found!")
        return
    
    try:
        # Generate PDF
        create_pdf_from_markdown(markdown_file, output_pdf)
        
        # Check if PDF was created
        if os.path.exists(output_pdf):
            file_size = os.path.getsize(output_pdf) / 1024  # KB
            print(f"📄 PDF file size: {file_size:.1f} KB")
            print(f"📁 Location: {os.path.abspath(output_pdf)}")
            print("✅ PDF is Google Docs compatible with simple formatting")
        else:
            print("❌ Error: PDF file was not created")
            
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        print("\n💡 Make sure you have the required dependencies installed:")
        print("   pip install markdown weasyprint")

if __name__ == "__main__":
    main()
