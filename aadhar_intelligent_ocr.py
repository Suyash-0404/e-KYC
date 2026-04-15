"""Intelligent Aadhar card OCR parser with field extraction."""

import cv2
import numpy as np
import pytesseract
import easyocr
import re
import logging
from typing import Dict, List, Tuple, Optional

def preprocess_for_aadhar(image: np.ndarray) -> np.ndarray:
    """Preprocess Aadhar card image for OCR extraction."""
    logging.info("Preprocessing Aadhar card...")
    
    # Upscale to 1800px
    target_width = 1800
    aspect_ratio = image.shape[0] / image.shape[1]
    target_height = int(target_width * aspect_ratio)
    upscaled = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale and denoise
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Sharpen text
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 3.0)
    sharpened = cv2.addWeighted(denoised, 2.5, gaussian, -1.5, 0)
    kernel_sharpen = np.array([[-1,-1,-1],[-1,10,-1],[-1,-1,-1]])
    sharpened = cv2.filter2D(sharpened, -1, kernel_sharpen)
    
    # Enhance contrast
    equalized = cv2.equalizeHist(sharpened)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    clahe_applied = clahe.apply(equalized)
    
    # Threshold and clean up
    thresh = cv2.adaptiveThreshold(clahe_applied, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 15, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    final = cv2.medianBlur(morph, 3)
    
    logging.info(f"Preprocessing complete: {final.shape}")
    cv2.imwrite("data/02_intermediate_data/aadhar_preprocessed_debug.jpg", final)
    
    return final


def extract_text_with_positions(image: np.ndarray) -> List[Tuple[str, float, Dict, str]]:
    """Extract text with bounding box positions using OCR engines."""
    logging.info("Extracting text with layout analysis...")
    
    results = []
    
    # Tesseract with layout analysis
    logging.info("Running Tesseract...")
    custom_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/-.:, '
    
    data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
    
    for i, text in enumerate(data['text']):
        if text.strip():
            conf = data['conf'][i] / 100.0
            bbox = {
                'x': data['left'][i],
                'y': data['top'][i],
                'w': data['width'][i],
                'h': data['height'][i]
            }
            results.append((text.strip(), conf, bbox, 'tesseract'))
    
    logging.info(f"Tesseract: {len(results)} text elements")
    
    # 2. EASYOCR for Hindi/English text
    logging.info("🌏 Running EasyOCR with layout analysis...")
    reader = easyocr.Reader(['en', 'hi'], gpu=False)
    easyocr_results = reader.readtext(image)
    
    for bbox_coords, text, conf in easyocr_results:
        x_min = int(min(p[0] for p in bbox_coords))
        y_min = int(min(p[1] for p in bbox_coords))
        x_max = int(max(p[0] for p in bbox_coords))
        y_max = int(max(p[1] for p in bbox_coords))
        
        bbox = {
            'x': x_min,
            'y': y_min,
            'w': x_max - x_min,
            'h': y_max - y_min
        }
        results.append((text.strip(), conf, bbox, 'easyocr'))
    
    logging.info(f"EasyOCR: {len(easyocr_results)} text elements")
    logging.info(f"Total text elements: {len(results)}")
    
    return results


def parse_aadhar_card_intelligent(image: np.ndarray) -> Dict[str, Optional[str]]:
    """
    INTELLIGENT AADHAR CARD PARSER - Understands Aadhar card layout
    
    Standard Aadhar card layout:
    - TOP: "Government of India" header
    - UPPER-CENTER: Name (large, bold, center-aligned)
    - MID-CENTER: Date of Birth / DOB: DD/MM/YYYY
    - MID-CENTER: Gender: Male/Female
    - BOTTOM-CENTER: 12-digit Aadhar number (XXXX XXXX XXXX format)
    - BOTTOM: VID number (if present)
    """
    logging.info("Starting INTELLIGENT AADHAR parsing...")
    
    # Step 1: Preprocess
    preprocessed = preprocess_for_aadhar(image)
    height, width = preprocessed.shape
    
    # Step 2: Extract all text with positions
    text_elements = extract_text_with_positions(preprocessed)
    
    # Initialize result
    result = {
        'aadhar_number': None,
        'name': None,
        'dob': None,
        'gender': None
    }
    
    # Helper patterns
    aadhar_pattern = re.compile(r'\b\d{4}\s*\d{4}\s*\d{4}\b')
    dob_pattern = re.compile(r'\b\d{2}[/\-\.]\d{2}[/\-\.]\d{4}\b')
    
    # Step 3: Find Aadhar number (12 digits, bottom area)
    aadhar_candidates = []
    for text, conf, bbox, source in text_elements:
        # Look in bottom 40% of card
        if bbox['y'] > height * 0.6:
            match = aadhar_pattern.search(text)
            if match:
                aadhar_num = match.group().replace(' ', '')
                if len(aadhar_num) == 12:
                    aadhar_candidates.append((aadhar_num, conf, bbox['y']))
                    logging.info(f"🔢 AADHAR CANDIDATE: {aadhar_num} (conf: {conf:.2f})")
    
    if aadhar_candidates:
        # Pick the one with highest confidence
        result['aadhar_number'] = max(aadhar_candidates, key=lambda x: x[1])[0]
        logging.info(f"SELECTED AADHAR: {result['aadhar_number']}")
    
    # Step 4: Find DOB (DD/MM/YYYY format, middle area)
    dob_candidates = []
    for text, conf, bbox, source in text_elements:
        # Look in middle 30-70% vertical area
        y_percent = (bbox['y'] / height) * 100
        if 30 <= y_percent <= 70:
            match = dob_pattern.search(text)
            if match:
                dob = match.group().replace('-', '/').replace('.', '/')
                dob_candidates.append((dob, conf))
                logging.info(f"📅 DOB CANDIDATE: {dob} (conf: {conf:.2f})")
    
    if dob_candidates:
        result['dob'] = max(dob_candidates, key=lambda x: x[1])[0]
        logging.info(f"SELECTED DOB: {result['dob']}")
    
    # Step 5: Find Gender
    gender_keywords = ['male', 'female', 'पुरुष', 'महिला']
    for text, conf, bbox, source in text_elements:
        text_lower = text.lower()
        for keyword in gender_keywords:
            if keyword in text_lower:
                if 'female' in text_lower or 'महिला' in text_lower:
                    result['gender'] = 'Female'
                else:
                    result['gender'] = 'Male'
                logging.info(f"⚥ GENDER: {result['gender']} (from: {text})")
                break
        if result['gender']:
            break
    
    # Step 6: Find NAME (most challenging!)
    # Name is typically:
    # - In UPPER area (20-50% from top)
    # - Larger font (bigger bbox height)
    # - Center-aligned (x position around center)
    # - NOT containing numbers (except maybe DOB text)
    # - NOT "Government of India"
    # - NOT containing special Aadhar keywords
    
    name_candidates = []
    center_x = width / 2
    
    exclude_keywords = ['government', 'india', 'aadhar', 'uidai', 'male', 'female', 
                        'dob', 'date of birth', 'यूआईडीएआई', 'आधार', 'vid']
    
    for text, conf, bbox, source in text_elements:
        y_percent = (bbox['y'] / height) * 100
        x_center = bbox['x'] + bbox['w'] / 2
        x_offset = abs(x_center - center_x) / width
        
        # Name criteria:
        # - Upper-middle area (20-60% from top)
        # - Relatively large text (height > 25px)
        # - Near center (within 40% of center)
        # - Good confidence
        # - At least 2 words or 8 characters
        # - Not excluded keywords
        
        if (20 <= y_percent <= 60 and 
            bbox['h'] > 25 and 
            x_offset < 0.4 and
            conf > 0.5 and
            (len(text.split()) >= 2 or len(text) >= 8) and
            not any(kw in text.lower() for kw in exclude_keywords) and
            not re.search(r'\d{4}', text)):  # No 4-digit numbers
            
            # Score based on position and size
            position_score = 1.0 - (y_percent - 35) / 100  # Prefer around 35% from top
            size_score = min(bbox['h'] / 50, 1.0)  # Prefer larger text
            center_score = 1.0 - x_offset
            
            total_score = conf * 0.4 + position_score * 0.3 + size_score * 0.2 + center_score * 0.1
            
            name_candidates.append((text, total_score, bbox, conf))
            logging.info(f"NAME CANDIDATE: {text} (conf: {conf:.2f}, y: {y_percent:.0f}%, score: {total_score:.2f})")
    
    if name_candidates:
        # IMPROVED: Get both English and Hindi names if available
        # Sort by score
        sorted_candidates = sorted(name_candidates, key=lambda x: x[1], reverse=True)
        
        # Check if best candidate is Hindi (has Devanagari characters)
        best_name = sorted_candidates[0][0]
        has_devanagari = any('\u0900' <= c <= '\u097F' for c in best_name)
        
        if has_devanagari and len(sorted_candidates) > 1:
            # Look for English name in remaining candidates
            for candidate_text, score, bbox, conf in sorted_candidates[1:]:
                # Check if it's English (mostly ASCII)
                is_english = all(ord(c) < 128 or c.isspace() for c in candidate_text)
                if is_english and len(candidate_text) >= 5:
                    # Found English name - use it!
                    result['name'] = candidate_text.upper()
                    logging.info(f"SELECTED NAME (English): {result['name']}")
                    break
            else:
                # No English name found, use Hindi
                result['name'] = best_name.upper()
                logging.info(f"SELECTED NAME (Hindi): {result['name']}")
        else:
            # Best candidate is already English or only one candidate
            result['name'] = best_name.upper()
            logging.info(f"SELECTED NAME: {result['name']}")
    
    logging.info("=" * 80)
    logging.info("INTELLIGENT AADHAR EXTRACTION COMPLETE:")
    logging.info(f"   Aadhar: {result['aadhar_number']}")
    logging.info(f"   Name: {result['name']}")
    logging.info(f"   DOB: {result['dob']}")
    logging.info(f"   Gender: {result['gender']}")
    logging.info("=" * 80)
    
    return result
