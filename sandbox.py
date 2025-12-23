import sys
import json
import io
import fitz
import pytesseract
from PIL import Image
from langchain_core.documents import Document

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def process_pdf_hybrid(pdf_path: str, text_threshold: int = 100):
    doc = fitz.open(pdf_path)
    docs = []

    for i, page in enumerate(doc):
        text = page.get_text()

        if len(text.strip()) < text_threshold:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes())).convert("RGB")
            text = pytesseract.image_to_string(img, lang = "eng")

        cleaned_text = text.strip()
        if len(cleaned_text) >= 20:
            docs.append({
                "page_content": text.strip(),
                "metadata": {
                    "page_number": i + 1
                }
            })

    return docs


if __name__ == "__main__":
    pdf_path = sys.argv[1]
    result = process_pdf_hybrid(pdf_path)
    print(json.dumps(result))
