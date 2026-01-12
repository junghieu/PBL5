import easyocr
import os

# Initialize EasyOCR reader with Vietnamese language
# This will be initialized once when the module is imported
reader = None


def get_ocr_reader():
    """Get or initialize the OCR reader."""
    global reader
    if reader is None:
        reader = easyocr.Reader(['vi'], gpu=False)
    return reader


def extract_text_from_image(image_path: str) -> str:
    """
    Extract Vietnamese text from an image using EasyOCR.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Extracted text as a string
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    ocr_reader = get_ocr_reader()
    
    # Perform OCR
    results = ocr_reader.readtext(image_path)
    
    # Extract text from results
    # Each result is a tuple: (bbox, text, confidence)
    extracted_text = "\n".join([result[1] for result in results])
    
    return extracted_text
