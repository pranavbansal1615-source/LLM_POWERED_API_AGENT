import fitz 
import pytesseract
from PIL import Image
import io
from langchain_core.documents import Document
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from pathlib import Path

def process_pdf_hybrid(pdf_path: str, text_threshold: int = 50):
    doc = fitz.open(pdf_path)
    docs = []

    for i, page in enumerate(doc):
        text = page.get_text()

        if len(text.strip()) < text_threshold:
            # Fallback to OCR
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            text = pytesseract.image_to_string(img)

        docs.append(
            Document(
                page_content=text.strip(),
                metadata={
                    "source_file": pdf_path.split("\\")[-1],
                    "page_number": i + 1,
                    "file_type": "pdf"
                }
            )
        )

    return docs
