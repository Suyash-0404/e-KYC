import cv2
import pytesseract

# Load the PAN card
img = cv2.imread('data/02_intermediate_data/contour_id.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Try different preprocessing approaches to find PAN number
print("=" * 80)
print("SEARCHING FOR PAN NUMBER IN IMAGE")
print("=" * 80)

# Method 1: Basic preprocessing
resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
text1 = pytesseract.image_to_string(resized, config='--psm 6')
print("\nMethod 1 (2x resize):")
for line in text1.split('\n'):
    if 'IWRPD' in line.upper() or len(line) == 10:
        print(f"  -> {line}")

# Method 2: Threshold
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
text2 = pytesseract.image_to_string(thresh, config='--psm 6')
print("\nMethod 2 (threshold):")
for line in text2.split('\n'):
    if 'IWRPD' in line.upper() or len(line) == 10:
        print(f"  -> {line}")

# Method 3: Look for any 10-character alphanumeric
print("\nAll text elements:")
data = pytesseract.image_to_data(resized, output_type=pytesseract.Output.DICT)
for i, txt in enumerate(data['text']):
    if len(txt) >= 8:
        print(f"  {txt} (conf: {data['conf'][i]})")

# Method 4: Try to find specific pattern
import re
all_text = text1 + " " + text2
pan_matches = re.findall(r'[A-Z]{3,5}[A-Z0-9]{4,6}[A-Z0-9]?', all_text)
if pan_matches:
    print("\nPotential PAN patterns found:")
    for match in pan_matches:
        print(f"  -> {match}")
