#!/usr/bin/env python3
"""Test PAN OCR extraction to debug incorrect PAN detection"""

import cv2
import easyocr
import re
import numpy as np

# Initialize EasyOCR
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

# Read the PAN card image
img_path = "data/02_intermediate_data/contour_id.jpg"
print(f"\nReading image: {img_path}")
img = cv2.imread(img_path)

if img is None:
    print("ERROR: Could not read image!")
    exit(1)

print(f"Image shape: {img.shape}")

# Preprocessing similar to what app uses
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
if h < 800:
    scale = 800 / h
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    print(f"Upscaled to: {gray.shape}")

# CLAHE and adaptive threshold
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Run OCR on original and preprocessed
print("\n" + "=" * 80)
print("OCR ON ORIGINAL IMAGE:")
print("=" * 80)
results_orig = reader.readtext(img, detail=1, batch_size=4, width_ths=0.7, mag_ratio=1.5)
for (bbox, text, conf) in results_orig:
    if conf > 0.2:
        print(f"[{conf:.2f}] {text}")

print("\n" + "=" * 80)
print("OCR ON PREPROCESSED IMAGE:")
print("=" * 80)
results_proc = reader.readtext(binary, detail=1, batch_size=4, width_ths=0.7, mag_ratio=1.5)
for (bbox, text, conf) in results_proc:
    if conf > 0.2:
        print(f"[{conf:.2f}] {text}")

# Combine all text
all_text = " | ".join([text for (_, text, conf) in results_orig if conf > 0.2])
print("\n" + "=" * 80)
print("COMBINED TEXT:")
print(all_text)
print("=" * 80)

# Search for PAN patterns
pan_pattern = r'\b([A-Z]{5}[0-9]{4}[A-Z])\b'
matches = re.findall(pan_pattern, all_text.upper())
print(f"\nSTRICT PAN REGEX MATCHES: {matches}")

# Look for 10-character alphanumeric strings
candidates = re.findall(r'\b([A-Z0-9]{10})\b', all_text.upper())
print(f"10-CHAR CANDIDATES: {candidates}")

# Look for strings that might be PAN with OCR errors
print("\nALL 8-12 CHAR ALPHANUMERIC STRINGS:")
potentials = re.findall(r'[A-Z0-9]{8,12}', all_text.upper())
for p in potentials:
    print(f"  '{p}'")

print("\n" + "=" * 80)
print("ACTUAL PAN SHOULD BE: IWRPD8134D")
print("=" * 80)
