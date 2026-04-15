"""Intelligent PAN card OCR with layout understanding and field extraction."""
import cv2
import numpy as np
import easyocr
import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Initialize OCR readers
reader_en = easyocr.Reader(['en'], gpu=False)

# Try Tesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract not available, using EasyOCR only")


def preprocess_for_pan(img):
    """Preprocess PAN card image for OCR extraction."""
    if img is None or img.size == 0:
        return None
    
    logger.info(f"Original shape: {img.shape}")
    
    # Upscale to 1800px for clarity
    height, width = img.shape[:2]
    target_width = 1800
    target_height = int(height * (target_width / width))
    img_resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    if len(img_resized.shape) == 3:
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_resized.copy()
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Sharpen text
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.5)
    sharpened = cv2.addWeighted(denoised, 2.5, gaussian, -1.5, 0)
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1,  10, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(sharpened, -1, kernel_sharpen)
    
    # Histogram and local contrast
    equalized = cv2.equalizeHist(sharpened)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(equalized)
    
    # Threshold
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 5
    )
    
    # Morphology cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    final = cv2.medianBlur(cleaned, 3)
    
    logger.info(f"PAN preprocessing complete: {final.shape}")
    return final


def extract_text_with_positions(img, engine='both'):
    """
    Extract text with bounding box positions
    Returns: [(text, confidence, (x, y, w, h)), ...]
    """
    results = []
    
    # Tesseract extraction
    if engine in ['tesseract', 'both'] and TESSERACT_AVAILABLE:
        try:
            logger.info("🔤 Running Tesseract with layout analysis...")
            
            # Get detailed data with positions - OPTIMIZED CONFIG FOR PAN CARDS
            # Use PSM 6 (uniform block of text) and explicitly whitelist PAN characters
            custom_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-. '
            
            tess_data = pytesseract.image_to_data(
                img, 
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            n_boxes = len(tess_data['text'])
            for i in range(n_boxes):
                text = tess_data['text'][i].strip()
                conf = float(tess_data['conf'][i])
                
                if text and conf > 0:
                    x = tess_data['left'][i]
                    y = tess_data['top'][i]
                    w = tess_data['width'][i]
                    h = tess_data['height'][i]
                    
                    results.append({
                        'text': text,
                        'conf': conf / 100.0,  # Convert to 0-1
                        'bbox': (x, y, w, h),
                        'source': 'tesseract'
                    })
            
            logger.info(f"Tesseract: {len(results)} text elements")
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
    
    # EasyOCR extraction
    if engine in ['easyocr', 'both']:
        try:
            logger.info("🌏 Running EasyOCR with layout analysis...")
            
            easy_results = reader_en.readtext(
                img,
                detail=1,
                paragraph=False,
                width_ths=0.5,
                height_ths=0.5
            )
            
            for (bbox, text, conf) in easy_results:
                # Extract bounding box coordinates
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x = int(min(x_coords))
                y = int(min(y_coords))
                w = int(max(x_coords) - x)
                h = int(max(y_coords) - y)
                
                results.append({
                    'text': text.strip(),
                    'conf': conf,
                    'bbox': (x, y, w, h),
                    'source': 'easyocr'
                })
            
            logger.info(f"EasyOCR: {len(easy_results)} text elements")
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
    
    return results


def parse_pan_card_intelligent(text_elements: List[Dict], img_height: int, img_width: int) -> Dict:
    """
    INTELLIGENT PAN CARD PARSER
    Uses spatial understanding and pattern matching to extract fields correctly
    
    Standard PAN Card Layout:
    - Top-right: Large PAN number (XXXXX9999X format)
    - Center-left: Name (below "स्थायी खाता संख्या" / "Permanent Account Number")
    - Below name: Father's Name (after "पिता का नाम" / "Father's Name")
    - Bottom: DOB (DD/MM/YYYY format)
    """
    logger.info("Starting INTELLIGENT PAN parsing...")
    
    extracted = {
        'PAN': None,
        'Name': None,
        'Father_Name': None,
        'DOB': None,
        'all_text': []
    }
    
    # Divide image into regions for spatial analysis
    # PAN cards have predictable layout
    top_region = img_height * 0.3  # Top 30% usually has PAN number
    middle_region = img_height * 0.6  # Middle 30-60% has name
    right_region = img_width * 0.5  # Right side for PAN number
    
    pan_candidates = []
    name_candidates = []
    father_candidates = []
    dob_candidates = []
    
    for elem in text_elements:
        text = elem['text'].strip()
        conf = elem['conf']
        x, y, w, h = elem['bbox']
        
        # Store all text
        extracted['all_text'].append(text)
        
        # PATTERN 1: PAN Number (XXXXX9999X)
        # Usually in top-right, larger font
        pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', text.upper())
        if pan_match:
            pan_num = pan_match.group()
            # PAN numbers are usually in top-right region
            position_score = 0
            if y < top_region:
                position_score += 50
            if x > right_region:
                position_score += 30
            if h > 30:  # Larger text
                position_score += 20
            
            pan_candidates.append({
                'pan': pan_num,
                'conf': conf,
                'score': conf * 100 + position_score,
                'pos': (x, y)
            })
            logger.info(f"PAN CANDIDATE: {pan_num} (conf: {conf:.2f}, score: {conf*100 + position_score:.1f}, pos: {x},{y})")
        
        # PATTERN 2: DOB (DD/MM/YYYY or DD-MM-YYYY)
        dob_match = re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', text)
        if dob_match:
            dob = dob_match.group().replace('-', '/')
            dob_candidates.append({
                'dob': dob,
                'conf': conf,
                'score': conf * 100,
                'pos': (x, y)
            })
            logger.info(f"📅 DOB CANDIDATE: {dob} (conf: {conf:.2f})")
        
        # PATTERN 3: Name detection
        # Names are usually after "Permanent Account Number" text
        # In center-left region, medium font
        if len(text) > 3 and conf > 0.3:
            # Check if it's in name region
            if top_region < y < middle_region and x < right_region:
                # Check if it's mostly alphabetic
                alpha_ratio = sum(c.isalpha() for c in text) / len(text)
                if alpha_ratio > 0.7:
                    # Avoid Hindi text (look for English)
                    if all(ord(c) < 128 or c.isspace() for c in text):
                        name_candidates.append({
                            'name': text,
                            'conf': conf,
                            'score': conf * 100 + (alpha_ratio * 30),
                            'pos': (x, y)
                        })
                        logger.info(f"NAME CANDIDATE: {text} (conf: {conf:.2f}, y: {y})")
        
        # PATTERN 4: Father's Name
        # Usually below name, similar characteristics
        if len(text) > 3 and conf > 0.2:
            if middle_region < y < img_height * 0.8 and x < right_region:
                alpha_ratio = sum(c.isalpha() for c in text) / len(text)
                if alpha_ratio > 0.7:
                    if all(ord(c) < 128 or c.isspace() for c in text):
                        father_candidates.append({
                            'name': text,
                            'conf': conf,
                            'score': conf * 100,
                            'pos': (x, y)
                        })
                        logger.info(f"FATHER CANDIDATE: {text} (conf: {conf:.2f}, y: {y})")
    
    # SELECT BEST CANDIDATES
    
    # Best PAN (highest score)
    if pan_candidates:
        pan_candidates.sort(key=lambda x: x['score'], reverse=True)
        extracted['PAN'] = pan_candidates[0]['pan']
        logger.info(f"SELECTED PAN: {extracted['PAN']}")
    
    # Best DOB
    if dob_candidates:
        dob_candidates.sort(key=lambda x: x['conf'], reverse=True)
        extracted['DOB'] = dob_candidates[0]['dob']
        logger.info(f"SELECTED DOB: {extracted['DOB']}")
    
    # Best Name (look for full name in middle region)
    if name_candidates:
        # Sort by y-position (top to bottom) and confidence
        name_candidates.sort(key=lambda x: (x['pos'][1], -x['conf']))
        
        # Try to find compound name (multiple words)
        for nc in name_candidates:
            if ' ' in nc['name'] or len(nc['name']) > 10:
                extracted['Name'] = nc['name'].upper()
                logger.info(f"SELECTED NAME: {extracted['Name']}")
                break
        
        # Fallback to first candidate
        if not extracted['Name'] and name_candidates:
            extracted['Name'] = name_candidates[0]['name'].upper()
            logger.info(f"SELECTED NAME (fallback): {extracted['Name']}")
    
    # Best Father's Name (below name region)
    if father_candidates:
        father_candidates.sort(key=lambda x: (x['pos'][1], -x['conf']))
        
        # Look for different name than person's name
        for fc in father_candidates:
            if extracted['Name'] and fc['name'].upper() != extracted['Name']:
                if ' ' in fc['name'] or len(fc['name']) > 8:
                    extracted['Father_Name'] = fc['name'].upper()
                    logger.info(f"SELECTED FATHER: {extracted['Father_Name']}")
                    break
    
    logger.info("=" * 80)
    logger.info(f"INTELLIGENT EXTRACTION COMPLETE:")
    logger.info(f"   PAN: {extracted['PAN']}")
    logger.info(f"   Name: {extracted['Name']}")
    logger.info(f"   Father: {extracted['Father_Name']}")
    logger.info(f"   DOB: {extracted['DOB']}")
    logger.info("=" * 80)
    
    return extracted


def extract_pan_card_intelligent(img):
    """
    Main function: Extract PAN card with intelligence
    Returns structured dictionary with all fields
    """
    if img is None:
        return {}
    
    try:
        # Step 1: Preprocess
        logger.info("🔧 Preprocessing PAN card...")
        preprocessed = preprocess_for_pan(img)
        
        if preprocessed is None:
            logger.error("Preprocessing failed")
            return {}
        
        # Save for debugging
        debug_path = "data/02_intermediate_data/pan_preprocessed_debug.jpg"
        cv2.imwrite(debug_path, preprocessed)
        logger.info(f"Saved: {debug_path}")
        
        # Step 2: Extract text with positions
        logger.info("📖 Extracting text with layout analysis...")
        text_elements = extract_text_with_positions(preprocessed, engine='both')
        
        if not text_elements:
            logger.warning("No text detected!")
            return {}
        
        logger.info(f"Total text elements: {len(text_elements)}")
        
        # Step 3: Intelligent parsing
        img_height, img_width = preprocessed.shape[:2]
        result = parse_pan_card_intelligent(text_elements, img_height, img_width)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in intelligent PAN extraction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}
