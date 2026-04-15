import easyocr
import cv2
import numpy as np
import logging
import os
import re

# Import intelligent parsers
from pan_intelligent_ocr import extract_pan_card_intelligent
from aadhar_intelligent_ocr import parse_aadhar_card_intelligent

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "ekyc_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

logger = logging.getLogger(__name__)

# Initialize EasyOCR reader (supports both English and Hindi)
reader = easyocr.Reader(['en', 'hi'], gpu=False)


def preprocess_image_for_ocr(img):
    """Preprocess image for OCR: upscale, deskew, denoise, sharpen, threshold."""
    if img is None or img.size == 0:
        logger.warning("Empty image received for preprocessing")
        return img
    
    logger.info(f"Original shape: {img.shape}")
    
    # Upscale to 1600px for better text clarity
    height, width = img.shape[:2]
    target_width = 1600
    target_height = int(height * (target_width / width))
    img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    logger.info(f"Resized to {img.shape}")
    
    # Deskew using edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
    
    if angles:
        median_angle = np.median(angles)
        if abs(median_angle) > 0.5:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            logger.info(f"Corrected deskew angle: {median_angle:.1f}°")
    
    # Denoise color and grayscale
    img = cv2.fastNlMeansDenoisingColored(img, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Sharpen text edges
    gaussian = cv2.GaussianBlur(gray, (0, 0), 3.0)
    gray = cv2.addWeighted(gray, 2.5, gaussian, -1.5, 0)
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    gray = cv2.filter2D(gray, -1, kernel_sharpen)
    
    # Enhance contrast
    gray = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Gamma correction
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gray = cv2.LUT(gray, table)
    
    # Dual thresholding
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
    thresh = cv2.bitwise_and(thresh1, thresh2)
    
    # Morphological cleanup
    kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_noise, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    logger.info("Preprocessing complete")
    return thresh


def extract_text(img, card_type='PAN'):
    """Intelligent OCR with layout understanding for document extraction.
    
    For PAN cards: Uses spatial analysis to correctly identify fields
    For Aadhaar: Uses pattern matching
    Target: 2-3 minutes, 98%+ accuracy
    """
    if img is None:
        logger.warning("No image provided for OCR")
        return ""
    
    try:
        if card_type == 'PAN':
            logger.info("Using INTELLIGENT PAN CARD PARSER...")
            
            # Use intelligent PAN parser
            result = extract_pan_card_intelligent(img)
            
            if not result:
                logger.warning("Intelligent parser returned no results, falling back...")
                return ""
            
            # Format result as text for compatibility with existing code
            text_parts = []
            
            if result.get('PAN'):
                text_parts.append(f"PAN: {result['PAN']}")
            
            if result.get('Name'):
                text_parts.append(f"Name: {result['Name']}")
            
            if result.get('Father_Name'):
                text_parts.append(f"Father: {result['Father_Name']}")
            
            if result.get('DOB'):
                text_parts.append(f"DOB: {result['DOB']}")
            
            # Add all detected text for postprocessing
            if result.get('all_text'):
                text_parts.extend(result['all_text'])
            
            final_text = " | ".join(text_parts)
            logger.info(f"📄 INTELLIGENT RESULT: {final_text[:300]}...")
            
            return final_text
        
        elif card_type == 'AADHAR':
            # Use INTELLIGENT AADHAR PARSER
            logger.info("Using INTELLIGENT AADHAR CARD PARSER...")
            
            result = parse_aadhar_card_intelligent(img)
            
            if not result:
                logger.warning("Intelligent Aadhar parser returned no results, falling back...")
                return extract_text_original(img, card_type)
            
            # Format result as text
            text_parts = []
            
            if result.get('aadhar_number'):
                text_parts.append(result['aadhar_number'])
                text_parts.append(f"Aadhar: {result['aadhar_number']}")
            
            if result.get('name'):
                text_parts.append(f"Name: {result['name']}")
            
            if result.get('dob'):
                text_parts.append(f"DOB: {result['dob']}")
            
            if result.get('gender'):
                text_parts.append(f"Gender: {result['gender']}")
            
            final_text = " | ".join(text_parts)
            logger.info(f"📄 INTELLIGENT AADHAR RESULT: {final_text}")
            
            return final_text
        
        else:
            # For other card types, use the original dual-engine approach
            logger.info("Starting DUAL-ENGINE OCR...")
            return extract_text_original(img, card_type)
        
    except Exception as e:
        logger.error(f"Error in OCR extraction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return ""


def extract_text_original(img, card_type='PAN'):
    """Dual-engine OCR: Tesseract for primary, EasyOCR for backup and complex text."""
    if img is None:
        logger.warning("No image provided for OCR")
        return ""
    
    try:
        logger.info("Starting OCR extraction...")
        
        preprocessed = preprocess_image_for_ocr(img)
        
        # Save preprocessed image for debugging
        debug_path = "/Users/apple/Downloads/ekyc sssss/data/02_intermediate_data/preprocessed_ocr.jpg"
        cv2.imwrite(debug_path, preprocessed)
        logger.info(f"Saved preprocessed image: {debug_path}")
        
        all_texts = []
        
        # ENGINE 1: TESSERACT for English printed text (FAST, ACCURATE)
        # Perfect for PAN cards with clean printed English
        logger.info("🔤 ENGINE 1: Running Tesseract OCR (English)...")
        try:
            import pytesseract
            
            # Configure for best English text detection
            # Allow only uppercase letters and numbers (PAN format)
            custom_config = r'--oem 3 --psm 6'
            
            # Run on original preprocessed
            tess_text = pytesseract.image_to_string(preprocessed, config=custom_config, lang='eng')
            
            # Also run on inverted image (white text on black)
            inverted = cv2.bitwise_not(preprocessed)
            tess_text_inv = pytesseract.image_to_string(inverted, config=custom_config, lang='eng')
            
            # Get detailed results with confidence
            tess_data = pytesseract.image_to_data(preprocessed, config=custom_config, lang='eng', output_type=pytesseract.Output.DICT)
            
            tesseract_results = []
            for i, text in enumerate(tess_data['text']):
                if text.strip():
                    conf = float(tess_data['conf'][i]) / 100.0 if tess_data['conf'][i] != -1 else 0.0
                    tesseract_results.append((text.strip(), conf, 'tesseract'))
            
            logger.info(f"TESSERACT: Found {len(tesseract_results)} text elements")
            all_texts.extend(tesseract_results)
            
            # Log Tesseract results
            for text, conf, source in tesseract_results:
                logger.info(f"🔤 TESS: '{text}' (conf: {conf:.3f})")
                
        except Exception as e:
            logger.warning(f"Tesseract not available or failed: {e}")
            logger.info("📌 Continuing with EasyOCR only...")
        
        # ENGINE 2: EASYOCR for complex text and Hindi (VERSATILE)
        logger.info("🌏 ENGINE 2: Running EasyOCR (English + Hindi)...")
        
        # Run on preprocessed image
        easy_results = reader.readtext(
            preprocessed,
            detail=1,
            paragraph=False,
            min_size=10,
            text_threshold=0.15,
            low_text=0.15,
            link_threshold=0.15,
            canvas_size=4000,
            mag_ratio=2.5,
            slope_ths=0.3,
            ycenter_ths=0.8,
            height_ths=0.7,
            width_ths=0.9,
            add_margin=0.15,
            beamWidth=5
        )
        
        for (bbox, text, conf) in easy_results:
            all_texts.append((text.strip(), conf, 'easyocr'))
        
        logger.info(f"EASYOCR: Found {len(easy_results)} text elements")
        
        # Log EasyOCR results
        for text, conf, source in all_texts:
            if source == 'easyocr':
                logger.info(f"🌏 EASY: '{text}' (conf: {conf:.3f})")
        
        logger.info(f"TOTAL: Found {len(all_texts)} text elements from both engines")
        
        # FUSION: Combine results with smart filtering
        extracted_texts = []
        seen_texts = set()
        
        # Sort by confidence (highest first)
        all_texts.sort(key=lambda x: x[1], reverse=True)
        
        for text, conf, source in all_texts:
            text_clean = text.strip()
            text_upper = text_clean.upper()
            
            # Skip if already seen (avoid duplicates)
            if text_upper in seen_texts:
                continue
            
            # Calculate alphanumeric ratio
            alpha_count = sum(c.isalnum() for c in text_clean)
            total_chars = len(text_clean)
            alpha_ratio = alpha_count / total_chars if total_chars > 0 else 0
            
            # PRIORITY 1: Auto-accept PAN pattern (critical!)
            if re.match(r'[A-Z]{5}[0-9]{4}[A-Z]', text_upper):
                extracted_texts.append(text_upper)
                seen_texts.add(text_upper)
                logger.info(f"PAN FOUND: '{text_upper}' (conf: {conf:.3f}, source: {source})")
                continue
            
            # PRIORITY 2: Auto-accept Aadhaar pattern
            if re.match(r'\d{4}\s*\d{4}\s*\d{4}', text_clean):
                extracted_texts.append(text_clean)
                seen_texts.add(text_upper)
                logger.info(f"AADHAAR FOUND: '{text_clean}' (conf: {conf:.3f}, source: {source})")
                continue
            
            # PRIORITY 3: Accept high-confidence text
            if conf > 0.3:
                extracted_texts.append(text_clean)
                seen_texts.add(text_upper)
                logger.info(f"✓ HIGH-CONF: '{text_clean}' (conf: {conf:.3f}, source: {source})")
                continue
            
            # PRIORITY 4: Accept medium-confidence with good alphanumeric content
            if conf > 0.1 and alpha_ratio >= 0.5:
                extracted_texts.append(text_clean)
                seen_texts.add(text_upper)
                logger.info(f"✓ MED-CONF: '{text_clean}' (conf: {conf:.3f}, alpha: {alpha_count}/{total_chars}, source: {source})")
                continue
            
            # PRIORITY 5: Accept low-confidence but long English text (like "PERMANENT ACCOUNT NUMBER")
            if alpha_ratio >= 0.8 and total_chars >= 5:
                extracted_texts.append(text_clean)
                seen_texts.add(text_upper)
                logger.info(f"✓ ENGLISH-TEXT: '{text_clean}' (conf: {conf:.3f}, alpha: {alpha_count}/{total_chars}, source: {source})")
        
        result_text = " | ".join(extracted_texts)
        logger.info(f"FUSION COMPLETE: {len(extracted_texts)} unique texts, {len(result_text)} chars")
        logger.info(f"📄 FINAL TEXT: {result_text[:200]}...")
        
        return result_text
        
    except Exception as e:
        logger.error(f"Error in OCR extraction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return ""


def extract_text_pytesseract(image):
    """
    Alternative OCR using Pytesseract only (as fallback)
    """
    try:
        import pytesseract
        
        logger.info("Attempting OCR with Pytesseract...")
        
        # Preprocess
        processed = preprocess_image_for_ocr(image)
        
        # Configure Pytesseract for better results
        custom_config = r'--oem 3 --psm 6 -l eng+hin'
        text = pytesseract.image_to_string(processed, config=custom_config)
        
        logger.info(f"Pytesseract extracted: {len(text)} characters")
        return text
        
    except Exception as e:
        logger.error(f"Error in Pytesseract extraction: {e}")
        return ""


# For debugging
def save_preprocessed_image(image, filename="preprocessed_debug.jpg"):
    """Save preprocessed image for debugging"""
    processed = preprocess_image_for_ocr(image)
    output_path = os.path.join("data", "02_intermediate_data", filename)
    cv2.imwrite(output_path, processed)
    logger.info(f"Preprocessed image saved to: {output_path}")
    return output_path


# Testing
if __name__ == "__main__":
    # Test with a sample image
    test_image_path = "data/02_intermediate_data/contour_id.jpg"
    if os.path.exists(test_image_path):
        img = cv2.imread(test_image_path)
        if img is not None:
            print("Testing OCR extraction...")
            text = extract_text(img)
            print("\nExtracted Text:")
            print("="*60)
            print(text)
            print("="*60)
