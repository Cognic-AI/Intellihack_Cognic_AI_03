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
    
    # Update the JSON content with the new response
    json_content[f"pages_{page + 1}_{page + 2}"] = response.json()
    
    # Save JSON content after each API call to the datasets folder
    with open("./datasets/DeepSeek-R1.json", "w") as json_file:
        json_file.write(json.dumps(json_content, indent=4))
    
    # Log the progress
    print(f"Processed pages {page + 1} and {page + 2}, saved to DeepSeek-R1.json")
