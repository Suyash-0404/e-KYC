import pandas as pd
from datetime import datetime
import re

def filter_lines(lines):
    start_index = None
    end_index = None

    for i in range(len(lines)):
        line = lines[i]
        if "INCOME TAX DEPARTMENT" in line and start_index is None:
            start_index = i
        if "Signature" in line:
            end_index = i
            break

    filtered_lines = []
    if start_index is not None and end_index is not None:
        for line in lines[start_index:end_index + 1]:
            if len(line.strip()) > 2:
                filtered_lines.append(line.strip())
    
    return filtered_lines

def extract_information(data_string):
    """Extract PAN card information from OCR text."""
    logging.info("Extracting PAN information")
    
    extracted_info = {
        "ID": "",
        "Name": "",
        "Father's Name": "",
        "DOB": "",
        "ID Type": "PAN",
        "original_id": ""
    }

    try:
        # Remove noisy header/footer lines
        import difflib
        header_phrases = [
            "INCOME TAX DEPARTMENT",
            "GOVT OF INDIA",
            "GOVERNMENT OF INDIA",
            "PERMANENT ACCOUNT NUMBER",
            "PERMANENT ACCOUNT NUMBER CARD",
            "INCOME TAX",
            "ACCOUNT NUMBER"
        ]

        cleaned_lines = []
        for line in data_string.split('\n'):
            s = line.strip()
            if not s:
                continue
            is_header = False
            for phrase in header_phrases:
                ratio = difflib.SequenceMatcher(None, s.upper(), phrase).ratio()
                if ratio > 0.6:
                    is_header = True
                    break
            if not is_header:
                cleaned_lines.append(s)

        cleaned_text = '\n'.join(cleaned_lines)
        # use cleaned_text for downstream extraction
        data_string = cleaned_text

        # Improved PAN Number Extraction (line-wise, robust)
        pan_pattern = r'\b([A-Z]{5}[0-9]{4}[A-Z])\b'
        lines = [line.strip() for line in data_string.split('\n') if len(line.strip()) > 2]

        def _normalize_pan_candidate(s: str) -> str:
            """Try to correct common OCR misreads and return a candidate in strict PAN format or empty string."""
            s = s.upper()
            # Remove non-alphanumeric
            s = re.sub(r'[^A-Z0-9]', '', s)
            if len(s) != 10:
                return ''
            # Positions 0-4 and 9 are letters; 5-8 digits
            chars = list(s)
            # Enhanced maps for common OCR errors (including D<->I swap)
            letter_map = {
                '0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z', '3': 'E', '4': 'A', '6': 'G', '7': 'T'
            }
            digit_map = {
                'O': '0', 'D': '0', 'Q': '0', 
                'I': '1', 'L': '1', 
                'Z': '2', 
                'E': '3',
                'A': '4',
                'S': '5', 
                'G': '6',
                'T': '7',
                'B': '8'
            }
            for i in range(10):
                c = chars[i]
                if i in (0,1,2,3,4,9):
                    # should be letter - but handle D/I ambiguity specially
                    if c.isdigit() and c in letter_map:
                        chars[i] = letter_map[c]
                    # Special case: D and I are often confused
                    elif c == 'D':
                        chars[i] = 'D'  # Keep D, but we'll try I variant below
                else:
                    # should be digit (positions 5-8)
                    if c.isalpha() and c in digit_map:
                        chars[i] = digit_map[c]

            candidate = ''.join(chars)
            if re.match(pan_pattern, candidate):
                return candidate
            return ''
        
        def _generate_pan_variants(s: str):
            """Generate multiple variants of a PAN-like string to handle common OCR errors"""
            variants = set()
            s = s.upper().strip()
            s = re.sub(r'[^A-Z0-9]', '', s)
            
            if len(s) < 9 or len(s) > 11:
                return variants
            
            # If length is 11, try removing each char
            if len(s) == 11:
                for i in range(len(s)):
                    variant = s[:i] + s[i+1:]
                    if len(variant) == 10:
                        variants.add(variant)
            
            # If length is 9, it might be missing a char
            if len(s) == 9:
                # Common positions where chars go missing
                for i in range(len(s) + 1):
                    # Try adding common letters/digits
                    for ch in ['I', 'D', 'W', 'R', 'P', '0', '1', '8']:
                        variant = s[:i] + ch + s[i:]
                        if len(variant) == 10:
                            variants.add(variant)
            
            if len(s) == 10:
                variants.add(s)
                # Try D<->I swaps at all positions
                for i in range(10):
                    if s[i] == 'D':
                        variant = s[:i] + 'I' + s[i+1:]
                        variants.add(variant)
                    elif s[i] == 'I':
                        variant = s[:i] + 'D' + s[i+1:]
                        variants.add(variant)
                    # Try 0<->O swaps
                    elif s[i] == '0':
                        variant = s[:i] + 'O' + s[i+1:]
                        variants.add(variant)
                    elif s[i] == 'O':
                        variant = s[:i] + '0' + s[i+1:]
                        variants.add(variant)
                    # Try 4<->A, 3<->E, 8<->B swaps
                    elif s[i] == '4':
                        variant = s[:i] + 'A' + s[i+1:]
                        variants.add(variant)
                    elif s[i] == 'A':
                        variant = s[:i] + '4' + s[i+1:]
                        variants.add(variant)
                    elif s[i] == '3':
                        variant = s[:i] + 'E' + s[i+1:]
                        variants.add(variant)
                    elif s[i] == 'E':
                        variant = s[:i] + '3' + s[i+1:]
                        variants.add(variant)
                    elif s[i] == '8':
                        variant = s[:i] + 'B' + s[i+1:]
                        variants.add(variant)
                    elif s[i] == 'B':
                        variant = s[:i] + '8' + s[i+1:]
                        variants.add(variant)
            
            return variants

        pan_candidates = []
        all_tokens = []
        
        for line in lines:
            # direct strict match first
            match = re.search(pan_pattern, line.upper())
            if match:
                pan_candidates.append(match.group(1))
                continue
            # Extract all possible PAN-like tokens
            tokens = re.findall(r'[A-Z0-9]{8,12}', line.upper())
            all_tokens.extend(tokens)
            
            # Try normalizing each token
            for tok in tokens:
                cand = _normalize_pan_candidate(tok)
                if cand:
                    pan_candidates.append(cand)
        
        # If no direct candidates, try generating variants of all tokens
        if not pan_candidates:
            print(f"No direct PAN match. Trying variants on {len(all_tokens)} tokens...")
            for tok in all_tokens:
                variants = _generate_pan_variants(tok)
                for var in variants:
                    # Normalize and validate each variant
                    normalized = _normalize_pan_candidate(var)
                    if normalized:
                        pan_candidates.append(normalized)
                        print(f"  Variant '{tok}' -> '{normalized}' matched PAN pattern")
        
        if pan_candidates:
            # Deduplicate and pick first
            pan_candidates = list(dict.fromkeys(pan_candidates))
            extracted_info["ID"] = pan_candidates[0]
            extracted_info["original_id"] = pan_candidates[0]
            print(f"PAN Found: {pan_candidates[0]}")
            if len(pan_candidates) > 1:
                print(f"  Other PAN candidates found: {pan_candidates[1:]}")
        else:
            # If strict extraction failed, perform a fuzzy PAN search to correct common OCR misreads
            def _fuzzy_pan_search(text_block: str):
                from itertools import product
                ambig_map = {
                    '0': ['0', 'O'], 'O': ['O', '0'],
                    '1': ['1', 'I', 'L'], 'I': ['I', '1'], 'L': ['L', '1'],
                    '5': ['5', 'S'], 'S': ['S', '5'],
                    '2': ['2', 'Z'], 'Z': ['Z', '2'],
                    '8': ['8', 'B'], 'B': ['B', '8'],
                    '6': ['6', 'G'], 'G': ['G', '6'],
                    '4': ['4', 'A'], 'A': ['A', '4'],
                    '7': ['7', 'T'], 'T': ['T', '7']
                }

                tokens = re.findall(r'[A-Z0-9]{8,12}', text_block.upper())
                tried = set()
                for tok in tokens:
                    if tok in tried:
                        continue
                    tried.add(tok)
                    # Build lists of possible characters for each position
                    choices = []
                    for ch in tok:
                        if ch in ambig_map:
                            choices.append(ambig_map[ch])
                        else:
                            choices.append([ch])

                    # Limit explosion: if total combos > 300, skip heavy tokens
                    total = 1
                    for c in choices:
                        total *= len(c)
                        if total > 300:
                            break
                    if total > 300:
                        continue

                    for prod in product(*choices):
                        cand = ''.join(prod)
                        if re.match(pan_pattern, cand):
                            return cand
                return ''

            # Try concatenated alphanumeric scan (handles spaces/punctuations splitting the PAN)
            concat = re.sub(r'[^A-Z0-9]', '', data_string.upper())
            found_concat = ''
            for i in range(max(0, len(concat) - 9)):
                sub = concat[i:i+10]
                cand = _normalize_pan_candidate(sub)
                if cand:
                    found_concat = cand
                    break
            if found_concat:
                extracted_info["ID"] = found_concat
                extracted_info["original_id"] = found_concat
                print("PAN Found (concat scan):", found_concat)
            else:
                fuzzy = _fuzzy_pan_search(data_string)
                if fuzzy:
                    extracted_info["ID"] = fuzzy
                    extracted_info["original_id"] = fuzzy
                    print("PAN Found (fuzzy):", fuzzy)
                else:
                    # Fallback: traditional method across whole text
                    pan_matches = re.findall(pan_pattern, data_string.upper())
                    if pan_matches:
                        extracted_info["ID"] = pan_matches[0]
                        extracted_info["original_id"] = pan_matches[0]

    # Extract name
        name_patterns = [
            r'(?:Name|Narae|Harne|Narne)\s*[:/]?\s*([A-Z][A-Z\s]{3,50})(?=\s*Father|$|\n)',
            r'\n\s*([A-Z]{3,}\s+[A-Z]{3,}(?:\s+[A-Z]{3,})?)\s*\n',
            r'([A-Z]{3,}\s+[A-Z]{3,}(?:\s+[A-Z]{3,})?)\s+(?=Father(?!\'s))',
        ]
        
        name_match = None
        for pattern in name_patterns:
            match = re.search(pattern, data_string, re.IGNORECASE | re.MULTILINE)
            if match:
                name_candidate = match.group(1).strip()
                name_candidate = ' '.join(name_candidate.split())  # Normalize whitespace
                
                # Validate: must be 2-4 words, each word 2+ chars
                words = name_candidate.split()
                if 2 <= len(words) <= 4 and all(len(w) >= 2 for w in words):
                    name_match = name_candidate
                    print(f"✓ Name Found (pattern match): {name_candidate}")
                    break
        
        if name_match and len(name_match) > 3 and name_match.upper() != "NAME":
            extracted_info["Name"] = name_match.title()
        
        # Fallback: Look for capitalized name-like sequences
        if not extracted_info["Name"]:
            name_pattern2 = r'\b([A-Z]{3,}\s+[A-Z]{3,}(?:\s+[A-Z]{3,})?)\b'
            name_matches = re.findall(name_pattern2, data_string)
            
            # EXPANDED exclusion list (government headers, common OCR noise)
            exclude = [
                'INCOME TAX DEPARTMENT', 'GOVT OF INDIA', 'PERMANENT ACCOUNT', 
                'ACCOUNT NUMBER', 'FATHER NAME', 'DATE OF BIRTH', 'INCOME TAX',
                'TAX DEPARTMENT', 'GOVERNMENT OF INDIA', 'GOVERNMENT INDIA',
                'BHARATIYA BHASHA', 'PERMANENT ACCOUNT NUMBER', 'NUMBER CARD',
                'SIGNATURE', 'PERMANENT', 'DEPARTMENT', 'DIGITALLY SIGNED',
                'INCOME TAX DEPT', 'PAN CARD', 'AADHAAR', 'UNIQUE IDENTIFICATION',
                'GOVT INDIA', 'OF INDIA', 'CARD PERMANENT'
            ]
            
            # Choose best candidate by scoring (prefer 2-3 word names, realistic length)
            best = None
            best_score = 0
            for match in name_matches:
                match_upper = match.upper()
                
                # Skip excluded headers
                if any(ex in match_upper for ex in exclude):
                    continue
                
                # Skip gibberish patterns
                if 'INCOLE' in match_upper or 'TAY' in match_upper or 'DEPARTIENT' in match_upper:
                    continue
                
                # Score based on realistic name structure
                words = match.split()
                if not (2 <= len(words) <= 4):  # Names are typically 2-4 words
                    continue
                
                # Skip if any word is too short (< 2 chars) or too long (> 20 chars)
                if any(len(w) < 2 or len(w) > 20 for w in words):
                    continue
                
                # Calculate score (prefer 2-3 words, moderate length)
                score = 0
                if len(words) in [2, 3]:
                    score += 5
                if 10 <= len(match) <= 40:
                    score += 3
                
                if score > best_score:
                    best_score = score
                    best = match
            
            if best:
                # post-process to fix common OCR misspellings (simple replacements)
                # Expanded corrections map for common OCR misreads seen in PAN/Aadhaar scans
                corrections = {
                    '0F': 'OF',
                    '0FINDIA': 'OF INDIA',
                    'GOWTOFINDIA': 'GOVT OF INDIA',
                    'GOVTOFINDIA': 'GOVT OF INDIA',
                    'GOVTOF INDIA': 'GOVT OF INDIA',
                    'ACCOUNI': 'ACCOUNT',
                    'ACCOUNl': 'ACCOUNT',
                    'DEPARTIENT': 'DEPARTMENT',
                    'DEPARTMONT': 'DEPARTMENT',
                    'INCOLE': 'INCOME',
                    'TAY': 'TAX',
                    'Hiarne': 'Name',
                    'MILINDGOVIND': 'MILIND GOVIND',
                    'IRPD': 'IRPD',
                    'APPLICATLON': 'APPLICATION',
                    'Dlaitally': 'Digitally',
                    'Siarzd': 'Signed',
                    'PhyBitally': 'Physically'
                }
                b = best
                for k, v in corrections.items():
                    b = b.replace(k, v)
                
                # Final safety check: if the corrected name contains header-like words, clear it
                header_words = ['DEPARTMENT', 'INCOME', 'GOVT', 'ACCOUNT', 'PERMANENT', 'TAX', 'GOVERNMENT']
                if any(hw in b.upper() for hw in header_words):
                    extracted_info["Name"] = ''
                    print("Name candidate rejected (contains header words):", b)
                else:
                    extracted_info["Name"] = b.title()
                    print("Name Found (pattern 2):", best)

        # Extract Father Name (PROFESSIONAL ENHANCED)
        father_patterns = [
            r'(?:Father\'?s?\s*Name|Fathers\s*Name)\s*[:/]?\s*([A-Z][A-Z\s]{3,50})(?=\s*Date|DOB|Birth|\n|$)',
            r'Father\'?s?\s*[:/]?\s*([A-Z]{3,}\s+[A-Z]{3,}(?:\s+[A-Z]{3,})?)',
            r'(?:Father|FATHER)\s*[:/]?\s*([A-Z][A-Z\s]{3,50})(?=\s*Date|DOB|Birth|\n|$)',
        ]
        
        father_match = None
        for pattern in father_patterns:
            match = re.search(pattern, data_string, re.IGNORECASE | re.DOTALL)
            if match:
                father_candidate = match.group(1).strip()
                father_candidate = ' '.join(father_candidate.split())
                
                # Validate: realistic name structure
                words = father_candidate.split()
                if 2 <= len(words) <= 4 and all(len(w) >= 2 for w in words):
                    father_match = father_candidate
                    print(f"✓ Father's Name Found: {father_candidate}")
                    break
        
        if father_match and len(father_match) > 3:
            extracted_info["Father's Name"] = father_match.title()

        # Extract Date of Birth
        dob_patterns = [
            r'\b(\d{2})/(\d{2})/(\d{4})\b',
            r'\b(\d{2})-(\d{2})-(\d{4})\b',
            r'\b(\d{2})\.(\d{2})\.(\d{4})\b',
        ]
        
        for pattern in dob_patterns:
            dob_matches = re.findall(pattern, data_string)
            if dob_matches:
                try:
                    day, month, year = dob_matches[0]
                    dob_str = day + "/" + month + "/" + year
                    parsed_date = datetime.strptime(dob_str, "%d/%m/%Y")
                    
                    if 1940 <= parsed_date.year <= 2024:
                        extracted_info["DOB"] = parsed_date
                        print("DOB Found:", dob_str)
                        break
                except ValueError:
                    continue

        print("\n" + "="*60)
        print("EXTRACTION RESULTS:")
        print("PAN:", extracted_info['ID'] or 'NOT FOUND')
        print("Original ID:", extracted_info['original_id'] or 'NOT FOUND')
        print("Name:", extracted_info['Name'] or 'NOT FOUND')
        print("Father:", extracted_info["Father's Name"] or 'NOT FOUND')
        print("DOB:", extracted_info['DOB'] or 'NOT FOUND')
        print("="*60 + "\n")

    except Exception as e:
        print("ERROR in PAN extraction:", str(e))
        import traceback
        traceback.print_exc()
    
    return extracted_info

def extract_information1(data_string):
    """Extract Aadhaar card information"""
    print("="*60)
    print("AADHAR EXTRACTION - Raw OCR Text:")
    print(data_string)
    print("="*60)
    
    extracted_info = {
        "ID": "",
        "Name": "",
        "Gender": "",
        "DOB": "",
        "ID Type": "AADHAR",
        "original_id": ""
    }

    try:
        # FIRST: Check if this is output from INTELLIGENT AADHAR PARSER
        # Format: "4877 2434 8672 | Aadhar: 4877 2434 8672 | Name: Suyash Milind Dustkar | DOB: 04/04/2005 | Gender: Male"
        if "Name:" in data_string and "Aadhar:" in data_string:
            print("Detected INTELLIGENT PARSER output - using structured extraction")
            
            # Extract Aadhar number
            aadhar_match = re.search(r'Aadhar:\s*([\d\s]+?)(?:\s*\||$)', data_string)
            if aadhar_match:
                aadhar_raw = aadhar_match.group(1).strip()
                # Normalize to format: 1234 5678 9012
                digits = re.sub(r'\D', '', aadhar_raw)
                if len(digits) >= 12:
                    formatted = f"{digits[0:4]} {digits[4:8]} {digits[8:12]}"
                    extracted_info["ID"] = formatted
                    extracted_info["original_id"] = formatted
                    print(f"✓ Aadhar: {formatted}")
            
            # Extract Name - IMPROVED to handle Hindi/Devanagari and English names
            # Updated regex to capture ANY Unicode characters (not just A-Za-z)
            name_match = re.search(r'Name:\s*(?:Name:\s*)?([^\|]+?)(?:\s*\||$)', data_string)
            if name_match:
                name_raw = name_match.group(1).strip()
                # Clean up common OCR errors
                name_clean = re.sub(r'\s+', ' ', name_raw)  # Normalize spaces
                # Remove trailing artifacts
                name_clean = re.sub(r'[|.,;]+$', '', name_clean).strip()
                # Remove common noise keywords
                noise_keywords = ['aadhar:', 'dob:', 'gender:', 'male', 'female', 'government', 'india']
                for keyword in noise_keywords:
                    if keyword.lower() in name_clean.lower():
                        name_clean = ''
                        break
                
                # Only title case if it's English text (has ASCII letters)
                if name_clean and any(c.isalpha() and ord(c) < 128 for c in name_clean):
                    name_clean = name_clean.title()
                
                if name_clean and len(name_clean) >= 2:  # At least 2 characters
                    extracted_info["Name"] = name_clean
                    print(f"✓ Name: {name_clean}")
                else:
                    print(f"⚠ Name too short or invalid: '{name_clean}'")
            
            # Extract DOB
            dob_match = re.search(r'DOB:\s*([\d/\-]+)', data_string)
            if dob_match:
                dob_str = dob_match.group(1).strip()
                try:
                    dob_date = datetime.strptime(dob_str, "%d/%m/%Y")
                    extracted_info["DOB"] = dob_date
                    print(f"✓ DOB: {dob_str}")
                except ValueError:
                    print(f"⚠ Could not parse DOB: {dob_str}")
            
            # Extract Gender
            gender_match = re.search(r'Gender:\s*(Male|Female)', data_string, re.IGNORECASE)
            if gender_match:
                extracted_info["Gender"] = gender_match.group(1).title()
                print(f"✓ Gender: {extracted_info['Gender']}")
            
            print("\n" + "="*60)
            print("INTELLIGENT EXTRACTION RESULTS:")
            for key, value in extracted_info.items():
                print(f"{key}: {value}")
            print("="*60 + "\n")
            
            return extracted_info
    
    except Exception as e:
        print(f"⚠ Error in intelligent parsing: {str(e)}, falling back to legacy parsing")
    
    # FALLBACK: Legacy parsing for raw OCR text
    try:
        # Normalize text and handle multiple delimiters
        updated_data_string = data_string.replace(".", "").replace("|", "\n")
        lines = [line.strip() for line in updated_data_string.split('\n') if len(line.strip()) > 2]
        
        print("Processed lines (legacy mode):")
        for i, line in enumerate(lines):
            print(f"Line {i}: {line}")
        # 1. Aadhaar number extraction - improved to handle various formats
        aadhar_line_idx = -1
        aadhar_num = None
        
        # Extract all digit sequences from text
        all_text = data_string.replace(" ", "").replace("\n", "")
        
        # Try multiple patterns
        patterns = [
            r'\b(\d{4}\s+\d{4}\s+\d{4})\b',  # Standard: 1234 5678 9012
            r'\b(\d{4}\s*\d{4}\s*\d{4})\b',  # Flexible spaces
            r'(\d{12})',                      # Continuous 12 digits
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, data_string)
            if matches:
                raw = matches[0].replace(" ", "")
                if len(raw) == 12:
                    aadhar_num = raw
                    break
        
        # Also try to extract from concatenated digits in any line
        if not aadhar_num:
            for line in lines:
                digits_only = re.sub(r'\D', '', line)  # Remove all non-digits
                if len(digits_only) >= 12:
                    # Take first 12 digits
                    aadhar_num = digits_only[:12]
                    break
        
        if aadhar_num:
            formatted_num = f"{aadhar_num[0:4]} {aadhar_num[4:8]} {aadhar_num[8:12]}"
            extracted_info["ID"] = formatted_num
            extracted_info["original_id"] = formatted_num
            
            # Find line containing Aadhaar
            for idx, line in enumerate(lines):
                if aadhar_num in line.replace(" ", ""):
                    aadhar_line_idx = idx
                    break
            
            print(f"Aadhaar extracted: {formatted_num} (from digits: {aadhar_num})")

        print(f"Aadhaar found at line {aadhar_line_idx}")

        # 2. Name extraction (multiple methods)
        name = ""
        
        # Skip words list - expanded
        skip_words = [
            'government', 'govt', 'india', 'aadhaar', 'unique', 'identification',
            'authority', 'male', 'female', 'birth', 'year', 'address', 'father',
            'valid', 'till', 'signature', 'number', 'uid', 'help', 'à¤†à¤§à¤¾à¤°', 'à¤­à¤¾à¤°à¤¤',
            'enrolment', 'verification', 'malel'  # Added malel which appears in OCR
        ]

        # Method 1: First try to find longest alphabetic sequence
        for line in lines:
            line_clean = line.strip()
            # Remove common OCR artifacts
            line_clean = re.sub(r'[|.]', ' ', line_clean)
            line_clean = ' '.join(line_clean.split())
            
            # Skip if contains digits (likely not a name)
            if re.search(r'\d', line_clean):
                continue
            
            # Skip if too short
            if len(line_clean) < 3:
                continue
                
            # Skip unwanted words
            if any(word.lower() in line_clean.lower() for word in skip_words):
                continue
            
            # Check if it's mostly alphabetic
            alpha_chars = sum(c.isalpha() or c.isspace() for c in line_clean)
            if alpha_chars / len(line_clean) > 0.8:  # 80% alphabetic
                words = line_clean.split()
                # Need at least 2 words for a full name
                if len(words) >= 2:
                    # Check each word is mostly alpha
                    if all(sum(c.isalpha() for c in w) >= len(w) * 0.7 for w in words):
                        name = line_clean
                        print(f"Name found (Method 1 - alphabetic scan): {name}")
                        break

        # Method 2: Check lines near Aadhaar number
        if not name and aadhar_line_idx >= 0:
            check_indices = [
                aadhar_line_idx - 1,
                aadhar_line_idx + 1,
                aadhar_line_idx - 2,
                aadhar_line_idx + 2
            ]
            
            for idx in check_indices:
                if 0 <= idx < len(lines):
                    line = lines[idx].strip()
                    # Skip unwanted lines
                    if any(word.lower() in line.lower() for word in skip_words):
                        continue
                    if len(line) < 3 or re.search(r'\d', line):
                        continue
                    # Check for valid name format
                    if re.match(r'^[A-Za-z\s\.]+$', line, re.IGNORECASE):
                        words = line.split()
                        if len(words) >= 2:
                            name = line
                            print(f"Name found (Method 2 - near Aadhaar): {name}")
                            break

        # Method 3: Look for name after markers
        if not name:
            name_markers = ["Name:", "Name", "à¤¨à¤¾à¤®:", "à¤¨à¤¾à¤®", "To:", "S/O:", "D/O:", "W/O:"]
            for line in lines:
                for marker in name_markers:
                    if marker in line:
                        potential_name = line.split(marker)[-1].strip()
                        if len(potential_name) > 2 and re.match(r'^[A-Za-z\s\.]+$', potential_name, re.IGNORECASE):
                            name = potential_name
                            print(f"Name found (Method 3 - after marker): {name}")
                            break
                if name:
                    break

        # Format name if found
        if name:
            name = ' '.join(name.split())  # Normalize spaces
            name = name.title()  # Proper case
            extracted_info["Name"] = name
            print(f"Final name: {name}")

        # 3. Gender extraction
        for line in lines:
            line_lower = line.lower()
            if 'female' in line_lower:
                extracted_info["Gender"] = "Female"
                break
            elif 'male' in line_lower and 'female' not in line_lower:
                extracted_info["Gender"] = "Male"
                break

        # 4. DOB extraction
        dob_patterns = [
            r'\b(\d{2})/(\d{2})/(\d{4})\b',
            r'\b(\d{2})-(\d{2})-(\d{4})\b',
            r'DOB:?\s*(\d{2})[/-](\d{2})[/-](\d{4})',
            r'Date of Birth:?\s*(\d{2})[/-](\d{2})[/-](\d{4})',
            r'à¤œà¤¨à¥à¤® à¤¤à¤¿à¤¥à¤¿:?\s*(\d{2})[/-](\d{2})[/-](\d{4})',
            r'YOB:?\s*(\d{4})',
            r'Year of Birth:?\s*(\d{4})'
        ]
        
        for pattern in dob_patterns:
            matches = re.search(pattern, data_string)
            if matches:
                try:
                    groups = matches.groups()
                    if len(groups) == 3:  # Full date format
                        day, month, year = groups
                        dob_str = f"{day}/{month}/{year}"
                        dob_date = datetime.strptime(dob_str, "%d/%m/%Y")
                        if 1940 <= dob_date.year <= 2024:
                            extracted_info["DOB"] = dob_date
                            print(f"DOB found: {dob_str}")
                            break
                    elif len(groups) == 1:  # Year only
                        year = groups[0]
                        if 1940 <= int(year) <= 2024:
                            extracted_info["DOB"] = year
                            print(f"Year of Birth found: {year}")
                            break
                except (ValueError, AttributeError):
                    continue

        # Print final results
        print("\n" + "="*60)
        print("EXTRACTION RESULTS:")
        for key, value in extracted_info.items():
            print(f"{key}: {value or 'NOT FOUND'}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"Error extracting AADHAR info: {str(e)}")
        import traceback
        traceback.print_exc()

    return extracted_info

