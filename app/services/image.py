import cv2
import numpy as np
from PIL import Image


def preprocess_image(image_path: str) -> str:
    """
    Preprocess image for better OCR results using OpenCV.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Path to the preprocessed image
    """
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding for better text detection
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)
    
    # Save preprocessed image
    base, ext = os.path.splitext(image_path)
    preprocessed_path = f"{base}_preprocessed{ext}"
    cv2.imwrite(preprocessed_path, denoised)
    
    return preprocessed_path


def validate_image(image_path: str) -> bool:
    """
    Validate if the file is a valid image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception:
        return False
