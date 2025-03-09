import requests
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO
import json

load_dotenv()
VisionAgent_api = os.getenv("VisionAgent_api")

# Read the PDF
input_pdf_path = "./datasets/DeepSeek-R1.pdf"
reader = PdfReader(input_pdf_path)

url = "https://api.va.landing.ai/v1/tools/agentic-document-analysis"
headers = {
    "Authorization": f"Basic {VisionAgent_api}",
    "include_marginalia": "false",
    "include_metadata_in_markdown": "false"
}

# Initialize an empty JSON content
json_content = {}

# Process pages 3 to 16 in pairs
for page in range(2, 16, 2): 
    writer = PdfWriter()
    
    # Add the current pair of pages to the writer
    for p in range(page, min(page + 2, len(reader.pages))):
        writer.add_page(reader.pages[p])
    
    # Create a temporary PDF in memory
    output_pdf = BytesIO()
    writer.write(output_pdf)
    output_pdf.seek(0)

    # Send the PDF to the API
    files = {
        "pdf": output_pdf
    }
    response = requests.post(url, files=files, headers=headers)
    
    # Get the response data
    response_data = response.json()
    
    # Extract text from chunks to create the markdown content
    if "data" in response_data and "chunks" in response_data["data"]:
        content = ""
        for chunk in response_data["data"]["chunks"]:
            if "text" in chunk:
                content += chunk["text"] + "\n\n"
        
        content = content.strip()
    
    # Save the markdown content to a single .md file after each API call
    with open("./datasets/DeepSeek-R1.md", "a") as md_file:  # Append mode to save all content
        md_file.write(f"# Pages {page + 1} and {page + 2}\n\n")
        md_file.write(content + "\n\n")
    
    # Log the progress
    print(f"Processed pages {page + 1} and {page + 2}, saved to DeepSeek-R1.md")
