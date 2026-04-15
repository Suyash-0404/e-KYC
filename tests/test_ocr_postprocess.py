#!/usr/bin/env python3
"""Run OCR + postprocess on a few sample ID images to show extraction results.

Usage:
  .venv/bin/python tests/test_ocr_postprocess.py

This uses the project's OCR engine and postprocess functions and prints results.
"""
import os
import sys
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ocr_engine import extract_text_combined
from postprocess import extract_information, extract_information1

candidates = [
    'data/01_raw_data/pan.jpeg',
    'data/01_raw_data/pan_1.jpg',
    'data/01_raw_data/pan_2.jpg',
    'data/01_raw_data/og.adhaar.1.jpg',
]

for p in candidates:
    if not os.path.exists(p):
        print(f"Skipping (not found): {p}")
        continue
    print('\n' + '='*80)
    print('Image:', p)
    img = cv2.imread(p)
    if img is None:
        print('Failed to read image')
        continue
    text = extract_text_combined(img)
    print('OCR text sample:')
    print(text[:800])

    # Try both extractors; PAN extractor expects PAN-style cards, Aadhaar extractor handles Aadhaar
    pan_res = extract_information(text)
    print('\nPAN extraction result:')
    print(pan_res)

    aad_res = extract_information1(text)
    print('\nAadhar extraction result:')
    print(aad_res)
    print('='*80)
