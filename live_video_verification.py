import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import easyocr
import logging
import re
from datetime import datetime
from sql_connection import fetch_records, fetch_records_aadhar
from domain_config import get_domain_config

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_str)

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en', 'hi'], gpu=False, verbose=False)

def get_db_details(original_id, option):
    """Fetch user data from DB with multiple ID format candidates."""
    try:
        logging.info(f"Fetching {option} details for ID: {original_id}")
        import hashlib
        candidates = []
        orig = str(original_id).strip()
        if not orig:
            logging.warning("Empty original_id provided to get_db_details")
            return {}

        candidates.append(orig)
        candidates.append(orig.replace(' ', ''))
        candidates.append(orig.upper())
        candidates.append(orig.lower())
        try:
            candidates.append(hashlib.sha256(orig.encode()).hexdigest())
        except Exception:
            pass

        seen = set()
        cand_list = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                cand_list.append(c)

        df = None
        for cand in cand_list:
            text_info = {'original_id': cand, 'ID': cand}
            if option == "PAN":
                df = fetch_records(text_info)
            else:
                df = fetch_records_aadhar(text_info)

            if df is not None and not df.empty:
                logging.info(f"Candidate '{cand}' returned {len(df)} records")
                user_data = df.iloc[0].to_dict()
                logging.info(f"✓ User found: {user_data.get('name', 'N/A')}")
                return user_data

        if option == 'PAN':
            for a, b in [('W', 'V'), ('0', 'O'), ('1', 'I'), ('5', 'S')]:
                trial = orig.replace(a, b)
                if trial not in seen:
                    text_info = {'original_id': trial, 'ID': trial}
                    df = fetch_records(text_info)
                    if df is not None and not df.empty:
                        user_data = df.iloc[0].to_dict()
                        return user_data

        logging.warning(f"✗ No user found with {option} ID: {original_id}")
        return {}
        
    except Exception as e:
        logging.error(f"Error fetching DB details: {str(e)}")
        return {}

def get_stored_face_image(db_details):
    """Extract stored face image from database record."""
    try:
        if 'face_image' not in db_details:
            logging.error(f"face_image not found in db_details. Available keys: {list(db_details.keys())}")
            return None
            
        if db_details['face_image'] is None:
            logging.error("face_image is None - user needs to re-register")
            return None
        
        face_blob = db_details['face_image']
        
        # Handle both bytes and memoryview types
        if isinstance(face_blob, memoryview):
            face_blob = face_blob.tobytes()
        elif not isinstance(face_blob, bytes):
            logging.error(f"Unexpected face_image type: {type(face_blob)}")
            return None
        
        # Decode JPEG bytes to numpy array
        nparr = np.frombuffer(face_blob, np.uint8)
        face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if face_img is None:
            logging.error("Failed to decode stored face image")
            return None
        
        # Convert BGR to RGB (DeepFace expects RGB)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        logging.info(f"Face image loaded: {face_img.shape}")
        return face_img
        
    except Exception as e:
        logging.error(f"Error extracting face image: {e}")
        return None

def preprocess_document_region(doc_region):
    """Preprocess document region for OCR with upscaling, denoising, and enhancement."""
    try:
        if len(doc_region.shape) == 3:
            gray = cv2.cvtColor(doc_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = doc_region.copy()
        
        h, w = gray.shape
        if h < 600:
            scale = 700 / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        unsharp = cv2.addWeighted(gray, 2.0, gaussian, -1.0, 0)
        
        coords = np.column_stack(np.where(unsharp > 0))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) > 0.5:
                (h, w) = unsharp.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                unsharp = cv2.warpAffine(unsharp, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        denoised = cv2.fastNlMeansDenoising(unsharp, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 3
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        kernel_sharpen = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(binary, -1, kernel_sharpen)
        
        return sharpened
        
    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        return doc_region

def extract_aadhar_info(text):
    """Extract Aadhaar number, name, DOB, and gender from OCR text."""
    info = {
        "id": "",
        "name": "",
        "dob": "",
        "gender": ""
    }
    
    try:
        aadhar_pattern1 = r'\b(\d{4}\s+\d{4}\s+\d{4})\b'
        aadhar_pattern2 = r'\b(\d{12})\b'
        
        match = re.search(aadhar_pattern1, text)
        if match:
            info["id"] = match.group(1)
        else:
            match = re.search(aadhar_pattern2, text)
            if match:
                num = match.group(1)
                info["id"] = f"{num[0:4]} {num[4:8]} {num[8:12]}"
        
        name_patterns = [
            r'([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)'
        ]
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for name in matches:
                    if 'Government' not in name and 'India' not in name:
                        info["name"] = name
                        break
                if info["name"]:
                    break
        
        dob_match = re.search(r'\b(\d{2})/(\d{2})/(\d{4})\b', text)
        if dob_match:
            info["dob"] = f"{dob_match.group(1)}/{dob_match.group(2)}/{dob_match.group(3)}"
        
        gender_match = re.search(r'\b(Male|Female|MALE|FEMALE)\b', text, re.IGNORECASE)
        if gender_match:
            info["gender"] = gender_match.group(1).capitalize()
        
        logging.info(f"Extracted: ID={info['id']}, Name={info['name']}, DOB={info['dob']}, Gender={info['gender']}")
        
    except Exception as e:
        logging.error(f"Info extraction error: {e}")
    
    return info

def extract_pan_info(text):
    """Extract PAN number, name, father name, and DOB from OCR text with variant generation."""
    info = {
        "id": "",
        "name": "",
        "father_name": "",
        "dob": ""
    }
    
    try:
        import re
        text_clean = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        pan_pattern = r'\b([A-Z]{5}[0-9]{4}[A-Z]{1})\b'
        pan_match = re.search(pan_pattern, text_clean.upper())
        
        if pan_match:
            info["id"] = pan_match.group(1)
        else:
            def normalize_pan(s):
                s = s.upper()
                s = re.sub(r'[^A-Z0-9]', '', s)
                if len(s) != 10:
                    return ''
                chars = list(s)
                letter_map = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z', '3': 'E', '4': 'A', '6': 'G', '7': 'T'}
                digit_map = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'E': '3', 'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8'}
                for i in range(10):
                    if i in (0,1,2,3,4,9) and chars[i].isdigit():
                        chars[i] = letter_map.get(chars[i], chars[i])
                    elif i in (5,6,7,8) and chars[i].isalpha():
                        chars[i] = digit_map.get(chars[i], chars[i])
                result = ''.join(chars)
                return result if re.match(pan_pattern, result) else ''
            
            def generate_pan_variants(s):
                """Generate variants for common OCR errors"""
                variants = set()
                s = s.upper().strip()
                s = re.sub(r'[^A-Z0-9]', '', s)
                
                if len(s) < 9 or len(s) > 11:
                    return variants
                
                if len(s) == 10:
                    variants.add(s)
                    # D<->I, 0<->O, 4<->A, 3<->E, 8<->B swaps
                    for i in range(10):
                        swaps = [
                            ('D', 'I'), ('I', 'D'),
                            ('0', 'O'), ('O', '0'),
                            ('4', 'A'), ('A', '4'),
                            ('3', 'E'), ('E', '3'),
                            ('8', 'B'), ('B', '8'),
                            ('1', 'I'), ('I', '1')
                        ]
                        for old, new in swaps:
                            if s[i] == old:
                                variant = s[:i] + new + s[i+1:]
                                variants.add(variant)
                
                return variants
            
            # Try tokens of length 8-12
            tokens = re.findall(r'[A-Z0-9]{8,12}', text_clean.upper())
            pan_candidates = []
            
            for tok in tokens:
                normalized = normalize_pan(tok)
                if normalized:
                    pan_candidates.append(normalized)
                    continue
                
                variants = generate_pan_variants(tok)
                for var in variants:
                    normalized = normalize_pan(var)
                    if normalized:
                        pan_candidates.append(normalized)
                        break
            
            if pan_candidates:
                info["id"] = pan_candidates[0]
                logging.info(f"PAN extracted via variants: {pan_candidates[0]}")
        
        text_for_name = re.sub(r'\b(INCOME|TAX|DEPARTMENT|GOVT|INDIA|PERMANENT|ACCOUNT|NUMBER)\b', '', text_clean, flags=re.IGNORECASE)
        
        name_patterns = [
            r'(?:Name|NAME)\s*[:/]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\b',
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
        ]
        for pattern in name_patterns:
            matches = re.findall(pattern, text_for_name)
            if matches:
                info["name"] = max(matches, key=len)
                break
        
        dob_match = re.search(r'\b(\d{2})/(\d{2})/(\d{4})\b', text_clean)
        if dob_match:
            info["dob"] = f"{dob_match.group(1)}/{dob_match.group(2)}/{dob_match.group(3)}"
        
        logging.info(f"Extracted PAN: ID={info['id']}, Name={info['name']}")
        
    except Exception as e:
        logging.error(f"PAN extraction error: {e}")
    
    return info

def analyze_frame(frame, db_details, option, reader):
    """Face verification using DeepFace VGG-Face model (proven for accuracy)."""
    try:
        stored_face = get_stored_face_image(db_details)
        if stored_face is None:
            return "No stored face! RE-REGISTER in Phase 1", False, 0.0
        
        live_face = None
        try:
            for backend in ['opencv', 'ssd', 'retinaface']:
                try:
                    faces = DeepFace.extract_faces(
                        img_path=frame, 
                        detector_backend=backend,
                        enforce_detection=False,
                        align=True
                    )
                    
                    if faces and len(faces) > 0:
                        live_face = max(faces, key=lambda f: f['facial_area']['w'] * f['facial_area']['h'])['face']
                        logging.info(f"Face detected with {backend}")
                        break
                except:
                    continue
            
            if live_face is None:
                return "No face detected", False, 0.0
            
        except Exception as e:
            return "Face detection failed", False, 0.0
        
        try:
            target_size = (224, 224)
            stored_resized = cv2.resize(stored_face, target_size, interpolation=cv2.INTER_LANCZOS4)
            live_resized = cv2.resize(live_face, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            result = DeepFace.verify(
                live_resized,
                stored_resized,
                model_name='VGG-Face',
                enforce_detection=False,
                distance_metric='cosine'
            )
            
            is_verified = result.get('verified', False)
            distance = result.get('distance', 1.0)
            threshold = result.get('threshold', 0.68)
            
            if distance <= threshold:
                similarity_percentage = 100 - (distance / threshold) * 35
            else:
                similarity_percentage = max(0, 65 - ((distance - threshold) / (1.5 - threshold)) * 65)
            
            logging.info(f"Face comparison - Distance: {distance:.3f}, Threshold: {threshold:.3f}, Similarity: {similarity_percentage:.1f}%")
            
            if is_verified:
                # VERIFIED by DeepFace - show actual similarity
                db_name = db_details.get('name', 'User')
                logging.info(f"MATCH! DeepFace verified=True, Similarity: {similarity_percentage:.1f}%")
                return f"VERIFIED! Welcome {db_name}!", True, similarity_percentage
            else:
                # Not verified - show why
                logging.info(f"NO MATCH! Distance {distance:.3f} > Threshold {threshold:.3f}")
                return f"Face doesn't match ({similarity_percentage:.1f}%)", False, similarity_percentage
                
        except Exception as e:
            logging.error(f"Comparison error: {e}")
            return "Comparison failed", False, 0.0
            
    except Exception as e:
        logging.error(f"analyze_frame error: {e}")
        return "System error", False, 0.0

def analyze_frame_legacy(frame, db_details, option, reader):
    """Fallback method: detects 2 faces in frame for OCR-only verification."""
    try:
        try:
            faces = DeepFace.extract_faces(
                img_path=frame, 
                detector_backend='opencv', 
                enforce_detection=False,
                align=False
            )
        except Exception as e:
            logging.warning(f"Face detection error: {e}")
            faces = []
        
        if len(faces) < 2:
            logging.info("Less than 2 faces detected - proceeding with OCR-only verification")
            
            # Try OCR on entire frame
            try:
                ocr_results = reader.readtext(
                    frame, 
                    detail=1,
                    paragraph=False,
                    batch_size=4,
                    width_ths=0.5,
                    mag_ratio=1.5
                )
                
                if not ocr_results or len(ocr_results) < 2:
                    return "Show your ID card clearly in frame", False
                
                extracted_texts = []
                for (bbox, text, conf) in ocr_results:
                    if conf > 0.15:
                        cleaned = re.sub(r'[^\x00-\x7F]+', ' ', text.strip())
                        if cleaned and len(cleaned) > 1:
                            extracted_texts.append(cleaned)
                
                full_text = " | ".join(extracted_texts)
                
                if len(full_text) < 10:
                    return "Cannot read document. Hold it closer", False
                
                if option == "PAN":
                    ocr_info = extract_pan_info(full_text)
                else:
                    ocr_info = extract_aadhar_info(full_text)
                
                return verify_ocr_match(ocr_info, db_details, option, skip_face=True)
                
            except Exception as e:
                logging.error(f"OCR-only error: {e}")
                return "Show both your FACE and DOCUMENT in frame", False
        
        faces_sorted = sorted(
            faces, 
            key=lambda f: f['facial_area']['w'] * f['facial_area']['h'], 
            reverse=True
        )
        
        user_face_region = faces_sorted[0]['facial_area']
        doc_face_region = faces_sorted[1]['facial_area']
        
        user_face = faces_sorted[0]['face']
        doc_face = faces_sorted[1]['face']
        
        face_verified = False
        try:
            match_result = DeepFace.verify(
                user_face, doc_face, 
                model_name='Facenet',
                enforce_detection=False,
                distance_metric='cosine'
            )
            
            distance = match_result.get('distance', 1.0)
            if match_result['verified'] or distance < 0.50:
                face_verified = True
                logging.info(f"Face match accepted (distance: {distance:.3f})")
            else:
                logging.info(f"Face mismatch but continuing (distance: {distance:.3f})")
                
        except Exception as e:
            logging.warning(f"Face matching error: {e}")
        
        doc_x = max(0, doc_face_region['x'] - 200)
        doc_y = max(0, doc_face_region['y'] - 200)
        doc_w = min(frame.shape[1] - doc_x, doc_face_region['w'] + 400)
        doc_h = min(frame.shape[0] - doc_y, doc_face_region['h'] + 400)
        
        doc_region = frame[doc_y:doc_y+doc_h, doc_x:doc_x+doc_w]
        
        scale_factor = 2.5
        doc_region_upscaled = cv2.resize(
            doc_region, 
            None, 
            fx=scale_factor, 
            fy=scale_factor, 
            interpolation=cv2.INTER_CUBIC
        )
        
        processed_doc = doc_region_upscaled
        if len(processed_doc.shape) == 3:
            processed_doc = cv2.cvtColor(processed_doc, cv2.COLOR_RGB2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        processed_doc = clahe.apply(processed_doc)
        
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        processed_doc = cv2.filter2D(processed_doc, -1, kernel)
        
        try:
            ocr_results = reader.readtext(
                processed_doc, 
                detail=1,
                paragraph=False,
                batch_size=1,
                width_ths=0.7,
                mag_ratio=1.0,
                decoder='greedy'
            )
            
            if not ocr_results:
                return "Cannot read document. Hold it closer", False
            
            extracted_texts = []
            for (bbox, text, conf) in ocr_results:
                if conf > 0.5:
                    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', text.strip())
                    if cleaned and len(cleaned) > 1:
                        extracted_texts.append(cleaned)
            
            full_text = " | ".join(extracted_texts)
            
            logging.info(f"OCR extracted: {full_text[:200]}...")
            
            if len(full_text) < 10:
                return "Document text unclear. Hold document closer", False
            
        except Exception as e:
            logging.error(f"OCR error: {e}")
            return "Reading document... Hold steady", False
        
        if option == "PAN":
            ocr_info = extract_pan_info(full_text)
        else:
            ocr_info = extract_aadhar_info(full_text)
        
        return verify_ocr_match(ocr_info, db_details, option, skip_face=not face_verified)
        
    except Exception as e:
        logging.error(f"Frame analysis error: {e}")
        return "Processing... Hold steady", False

def verify_ocr_match(ocr_info, db_details, option, skip_face=False):
    """Verify OCR results against database. Match strategy: ID > Name + quality."""
    try:
        if not db_details:
            return "No database record found", False
        
        db_name = str(db_details.get("name", "")).lower().strip()
        db_id = str(db_details.get("original_id", "")).strip()
        
        ocr_name = ocr_info.get("name", "").lower().strip()
        ocr_id = ocr_info.get("id", "").replace(" ", "").strip()
        db_id_normalized = db_id.replace(" ", "").strip()
        
        logging.info(f"Extracted: ID={ocr_id}, Name={ocr_name}")
        logging.info(f"DB Match - Name: '{db_name}' vs '{ocr_name}'")
        logging.info(f"DB Match - ID: '{db_id_normalized}' vs '{ocr_id}'")
        
        name_match = False
        name_similarity = 0.0
        if db_name and ocr_name:
            db_clean = ' '.join(db_name.split())
            ocr_clean = ' '.join(ocr_name.split())
            
            if db_clean == ocr_clean:
                name_match = True
                name_similarity = 1.0
                logging.info("Exact name match!")
            else:
                db_words = set(db_clean.split())
                ocr_words = set(ocr_clean.split())
                common_words = db_words.intersection(ocr_words)
                
                if len(common_words) >= 1:
                    name_match = True
                    name_similarity = len(common_words) / max(len(db_words), len(ocr_words))
                    logging.info(f"Name partial match: {len(common_words)} common words (similarity: {name_similarity:.2f})")
                else:
                    from difflib import SequenceMatcher
                    ratio = SequenceMatcher(None, db_clean, ocr_clean).ratio()
                    if ratio > 0.5:
                        name_match = True
                        name_similarity = ratio
                        logging.info(f"Name fuzzy match (similarity: {ratio:.2f})")
        
        id_match = False
        if db_id_normalized and ocr_id:
            if option == "AADHAR":
                if db_id_normalized == ocr_id:
                    id_match = True
                    logging.info("Exact Aadhaar match!")
                elif len(db_id_normalized) >= 8 and len(ocr_id) >= 8:
                    if db_id_normalized[-8:] == ocr_id[-8:]:
                        id_match = True
                        logging.info("Aadhaar last 8 digits match!")
                    elif len(db_id_normalized) >= 6 and len(ocr_id) >= 6:
                        if db_id_normalized[-6:] == ocr_id[-6:]:
                            id_match = True
                            logging.info("Aadhaar last 6 digits match!")
                    else:
                        for i in range(len(ocr_id) - 7):
                            if ocr_id[i:i+8] in db_id_normalized:
                                id_match = True
                                logging.info(f"Aadhaar partial match: {ocr_id[i:i+8]}")
                                break
            else:
                if db_id_normalized.upper() == ocr_id.upper():
                    id_match = True
                    logging.info("Exact PAN match!")
                elif len(db_id_normalized) >= 7 and len(ocr_id) >= 7:
                    if db_id_normalized[:7].upper() == ocr_id[:7].upper():
                        id_match = True
                        logging.info("PAN partial match (first 7 chars)!")
        
        if name_match and id_match:
            return "KYC VERIFIED! Full match confirmed!", True
        elif id_match:
            return "ID verified! Accepting (name partially matched).", True
        elif name_match and name_similarity > 0.7:
            return f"Name matched strongly ({name_similarity*100:.0f}%)! Accepting.", True
        elif name_match:
            return f"Name matched ({name_similarity*100:.0f}%) but ID unclear. Hold document steady", False
        else:
            return "No match.", False
            
    except Exception as e:
        logging.error(f"Verification error: {e}")
        return "Verification in progress...", False

def main():
    st.header("n Phase 2: Live Video Verification")
    
    with st.expander("📱 Camera & Quality Settings", expanded=True):
        st.markdown("""
        **📱 Smartphone as webcam (MUCH BETTER QUALITY!):**
        
        **Option 1: IP Webcam App (Recommended - Wireless)**
        1. Install "IP Webcam" app on Android 
        2. Open app and tap "Start Server"
        3. Note the IP address shown (e.g., http://192.168.1.100:8080)
        4. Enter that URL below
        
        **Option 2: USB Debugging (Advanced - Android)**
        1. Enable Developer Options on Android
        2. Enable USB Debugging
        3. Connect via USB-C cable
        4. Use `scrcpy --v4l2sink=/dev/video2` or ADB webcam tools
        """)
        
        camera_source = st.radio(
            "Select Camera Source:",
            ["Built-in Webcam", "External Webcam", "Smartphone via IP Webcam URL"],
            horizontal=True
        )
        
        if camera_source == "Smartphone via IP Webcam URL":
            ip_url = st.text_input("Enter IP Webcam URL (e.g., http://192.168.1.100:8080/video)", 
                                   placeholder="http://192.168.1.100:8080/video")
        else:
            ip_url = None
        
        enhance_quality = st.checkbox("Enable Sharpening Enhancement", value=True, 
                                     help="Apply real-time sharpening to improve text clarity")
    
    st.info("""
    **Instructions:**
    1. Select your ID card type
    2. Enter your EXACT ID number (as registered in Phase 1)
    3. Click 'Start Verification'
    
    **💡 Camera Tips:**
    - Use good lighting 
    - Stay still for 5-7 seconds during camera stabilization
    - Balanced approach: Fast matches get 3s confirmation, Poor conditions speed up checks
    """)
    
    option = st.selectbox("ID Card Type", ("AADHAR", "PAN"))
    
    if option == "PAN":
        original_id = st.text_input("Enter your PAN number (e.g., ABCDE1234F):").upper().strip()
    else:
        original_id = st.text_input("Enter your Aadhaar number:").strip()
    
    start = st.button("Start Live Verification")
    
    if start:
        if not original_id:
            st.error("Please enter your ID number")
            return
        
        # Load OCR reader
        with st.spinner("Loading OCR engine..."):
            reader = load_ocr_reader()
        
        # Check database
        with st.spinner("Checking database..."):
            db_details = get_db_details(original_id, option)
            
            logging.info(f"DB details fetched for: {original_id}")
            logging.info(f"DB details keys: {list(db_details.keys()) if db_details else 'empty'}")
            if db_details:
                for key, val in db_details.items():
                    if key == 'face_image':
                        logging.info(f"  - {key}: {type(val)} (len={len(val) if val and hasattr(val, '__len__') else 'None'})")
                    else:
                        logging.info(f"  - {key}: {val}")
            logging.info("=" * 80)
        
        if not db_details:
            st.error(f"User not found in database with {option} ID: {original_id}")
            st.info("Please complete Phase 1 registration first.")
            return
        
        st.success(f"User found: {db_details.get('name', 'N/A')}")
        st.info("Opening camera... Hold your ID card next to your face")
        
        # Determine camera source
        if camera_source == "Smartphone via IP Webcam URL":
            if not ip_url:
                st.error("Please enter the IP Webcam URL")
                return
            # For IP webcam, add /video to URL if not present
            if not ip_url.endswith('/video'):
                ip_url = ip_url.rstrip('/') + '/video'
            cap = cv2.VideoCapture(ip_url)
            st.info(f"Connecting to smartphone at {ip_url}")
        elif camera_source == "External Webcam (Index 1)":
            cap = cv2.VideoCapture(1)
        else:
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Cannot access camera. Check permissions.")
            return
        
        resolutions = [
            (3840, 2160),
            (2560, 1440),
            (1920, 1080),
            (1280, 720),
        ]
        
        best_width, best_height = 0, 0
        
        for width, height in resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            if actual_w >= width * 0.9 and actual_h >= height * 0.9:
                best_width, best_height = actual_w, actual_h
                logging.info(f"Camera accepted resolution: {actual_w:.0f}x{actual_h:.0f}")
                break
            else:
                logging.info(f"Camera rejected {width}x{height}, got {actual_w:.0f}x{actual_h:.0f}")
        
        if best_width == 0:
            best_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            best_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
        cap.set(cv2.CAP_PROP_CONTRAST, 128)
        cap.set(cv2.CAP_PROP_SATURATION, 128)
        cap.set(cv2.CAP_PROP_SHARPNESS, 128)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        cap.set(cv2.CAP_PROP_FOCUS, 50)
        
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        st.success(f"🎥 Camera ready: **{actual_width:.0f}x{actual_height:.0f}** @ {actual_fps:.0f}fps")
        
        if actual_width >= 1920:
            st.success("Full HD or higher resolution achieved!")
        
        st.markdown("""
        <style>
        .stImage {
            width: 100% !important;
            max-width: 100% !important;
        }
        .element-container img {
            width: 100% !important;
            max-width: 100% !important;
            height: auto !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("### Live Verification Status")
        countdown_display = st.empty()
        percentage_display = st.empty()
        progress_bar = st.progress(0)
        result_box = st.empty()
        st.markdown("---")
        stframe = st.empty()
        stop_button = st.button("⏹ Stop Verification")
        
        verified = False
        verification_streak = 0
        required_streak = 3
        
        # Get domain-specific threshold from session state
        domain_config = get_domain_config(st.session_state.get('selected_domain', 'restaurant'))
        threshold = domain_config.get('face_threshold', 65.0)
        st.info(f"Domain: {domain_config['name']} | Threshold: {threshold}%")
        
        import time
        
        face_first_detected = None
        initial_detection_wait = 3.5
        
        base_check_interval = 1.5
        poor_lighting_interval = 1.2
        good_matching_interval = 1.8
        
        fast_match_min_delay = 3.0
        normal_match_min_delay = 2.0
        
        last_check_time = time.time() - 10.0
        check_interval = base_check_interval
        verification_confirmed_time = None
        matched_streak_start_time = None
        quick_match_detected = False
        
        st.info("Initializing... Wait 3-4 seconds for camera stabilization before verification starts")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            
            if not ret or frame is None or frame.size == 0:
                logging.warning("Failed to grab frame, retrying...")
                cap.release()
                time.sleep(0.5)
                if camera_source == "Smartphone via IP Webcam URL":
                    cap = cv2.VideoCapture(ip_url)
                elif camera_source == "External Webcam (Index 1)":
                    cap = cv2.VideoCapture(1)
                else:
                    cap = cv2.VideoCapture(0)
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            poor_lighting = avg_brightness < 60
            
            if enhance_quality:
                gaussian = cv2.GaussianBlur(frame_rgb, (0, 0), 2.0)
                frame_rgb = cv2.addWeighted(frame_rgb, 1.5, gaussian, -0.5, 0)
            
            stframe.image(frame_rgb, channels="RGB", use_column_width=True, output_format='PNG')
            
            current_time = time.time()
            
            if face_first_detected is None:
                try:
                    quick_faces = DeepFace.extract_faces(
                        img_path=frame_rgb, 
                        detector_backend='opencv', 
                        enforce_detection=False,
                        align=False
                    )
                    if len(quick_faces) > 0:
                        face_first_detected = current_time
                        logging.info("Face detected - starting stabilization wait")
                except:
                    pass
            
            if poor_lighting:
                check_interval = poor_lighting_interval
            else:
                check_interval = good_matching_interval
            
            time_until_check = check_interval - (current_time - last_check_time)
            
            if face_first_detected is None:
                countdown_display.info(f"Waiting for face detection... Position your face in frame")
            elif current_time - face_first_detected < initial_detection_wait:
                time_remain = initial_detection_wait - (current_time - face_first_detected)
                countdown_display.info(f"Camera stabilizing... **{time_remain:.1f}s** until verification starts")
            elif verification_confirmed_time is None and time_until_check > 0:
                countdown_display.info(f"Next check: **{time_until_check:.1f}s** | Check {verification_streak + 1}/3")
            elif verification_confirmed_time is not None:
                time_since_confirmed = current_time - verification_confirmed_time
                countdown_display.success(f"Verified! Processing... **{time_since_confirmed:.1f}s**")
            else:
                countdown_display.success("Analyzing now...")
            
            should_check = (
                face_first_detected is not None and 
                current_time - face_first_detected >= initial_detection_wait and
                current_time - last_check_time >= check_interval
            )
            
            if should_check:
                last_check_time = current_time
                
                result_text, frame_verified, match_percentage = analyze_frame(frame_rgb, db_details, option, reader)
                
                frame_verified_fixed = (match_percentage >= threshold)
                
                if match_percentage >= threshold:
                    percentage_display.success(f"**Match Score: {match_percentage:.1f}%** (✓ Passes 65% threshold)")
                elif match_percentage >= 45:
                    percentage_display.info(f"**Match Score: {match_percentage:.1f}%** (need 65%+)")
                else:
                    percentage_display.warning(f"**Match Score: {match_percentage:.1f}%** (too low - need 65%+)")
                
                progress_value = max(0, min(100, int(match_percentage)))
                progress_bar.progress(progress_value)
                
                if frame_verified_fixed:
                    if verification_streak == 0:
                        matched_streak_start_time = current_time
                        if match_percentage >= 85:
                            quick_match_detected = True
                            logging.info("RAPID MATCH DETECTED - Very clear match!")
                    
                    verification_streak += 1
                    result_box.success(f"{result_text} | ✓ Check {verification_streak}/{required_streak} Passed!")
                    logging.info(f"Verification streak: {verification_streak}/{required_streak}")
                    
                    if verification_streak >= required_streak:
                        verification_confirmed_time = current_time
                        result_box.success(f"🎉 ALL CHECKS PASSED! {result_text}")
                        percentage_display.success(f"**Final Score: {match_percentage:.1f}%** ALL VERIFIED!")
                        
                        if quick_match_detected:
                            min_delay = fast_match_min_delay
                        else:
                            min_delay = normal_match_min_delay
                        
                        time_elapsed_since_verified = 0
                        while time_elapsed_since_verified < min_delay and cap.isOpened():
                            ret, hold_frame = cap.read()
                            if ret:
                                hold_frame_rgb = cv2.cvtColor(hold_frame, cv2.COLOR_BGR2RGB)
                                if enhance_quality:
                                    gaussian = cv2.GaussianBlur(hold_frame_rgb, (0, 0), 2.0)
                                    hold_frame_rgb = cv2.addWeighted(hold_frame_rgb, 1.5, gaussian, -0.5, 0)
                                stframe.image(hold_frame_rgb, channels="RGB", use_column_width=True, output_format='PNG')
                            
                            time_elapsed_since_verified = current_time - verification_confirmed_time
                            time_remain_delay = min_delay - time_elapsed_since_verified
                            countdown_display.success(f"Verification confirmed! Processing... **{time_remain_delay:.1f}s**")
                            time.sleep(0.1)
                            current_time = time.time()
                        
                        try:
                            from sql_connection import insert_verified_record
                            db_name = db_details.get('name', 'Unknown')
                            db_original_id = db_details.get('original_id', 'Unknown')
                            db_id_type = db_details.get('id_type', 'Unknown')
                            
                            stored_face = get_stored_face_image(db_details)
                            
                            if stored_face is not None:
                                success = insert_verified_record(db_original_id, db_name, db_id_type, stored_face)
                                if success:
                                    logging.info(f"Verified record saved: {db_name} ({db_id_type}: {db_original_id})")
                                    st.info(f"Verification record saved to database!")
                                else:
                                    logging.error(f"Failed to save verified record")
                            else:
                                logging.error(f"No face image available for verified table")
                        except Exception as e:
                            logging.error(f"Error saving to verified table: {e}")
                        
                        verified = True
                        break
                else:
                    if verification_streak >= 1:
                        if poor_lighting and avg_brightness < 20:
                            verification_streak = 0
                            quick_match_detected = False
                            matched_streak_start_time = None
                            result_box.error(f"Screen too dark! Streak reset (Match: {match_percentage:.1f}%)")
                            logging.info(f"PITCH BLACK detected (brightness: {avg_brightness:.1f}) - Streak RESET")
                        else:
                            result_box.warning(f"Mismatch ({match_percentage:.1f}%) | Streak PROTECTED at {verification_streak}/{required_streak} - Hold steady!")
                            logging.info(f"Mismatch but streak {verification_streak}/{required_streak} PROTECTED (match: {match_percentage:.1f}%)")
                    else:
                        result_box.error(f"Mismatch ({match_percentage:.1f}%) | Try again from 0/3")
                        logging.info(f"Failed before reaching 1/3 (match: {match_percentage:.1f}%)")
                        countdown_display.info(f"Hold steady - improving detection. Current progress: 0/3")
        
        cap.release()
        
        if verified:
            st.success("e-KYC Verification is Complete!")
            st.info("Congratulations !! e-KYC Verification is Complete! You can now close the window. Thank You!!")

if __name__ == "__main__":
    main()
else:
    __all__ = ['main']