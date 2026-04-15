#!/usr/bin/env python3
"""Programmatic test: insert an aadhar record with missing DOB and run face verification.

This script performs the same work the Streamlit flow does but without the UI:
 - Prepares a `text_info` dict for AADHAR with missing DOB
 - Calls `insert_records_aadhar` to insert into DB
 - Calls `deepface_face_comparison` to verify two images from data/01_raw_data
 - Prints results and logs to help debugging
"""
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sql_connection
from face_verification import deepface_face_comparison


def main():
    # Use example files shipped in repo
    id_img = os.path.join('data', '01_raw_data', 'og.adhaar.1.jpg')
    face_img = os.path.join('data', '01_raw_data', 'og.img.1.jpg')

    print('ID image exists:', os.path.exists(id_img))
    print('Face image exists:', os.path.exists(face_img))

    text_info = {
        'ID': '4877 2434 8672',
        'original_id': '4877 2434 8672',
        'Name': 'Dae',
        'Gender': 'Male',
        'DOB': '',  # missing DOB
        'ID Type': 'AADHAR',
        'Embedding': []
    }

    print('\nAttempting insert_records_aadhar...')
    ok = sql_connection.insert_records_aadhar(text_info)
    print('Insert OK:', ok)

    # Fetch back the record to confirm stored value
    df = sql_connection.fetch_records_aadhar(text_info)
    print('\nFetched rows:')
    try:
        print(df.to_dict(orient='records'))
    except Exception:
        print('No rows returned or pandas not available')

    print('\nRunning face verification (may take some time)...')
    verified, details = deepface_face_comparison(id_img, face_img)
    print('Verified:', verified)
    print('Details:')
    print(json.dumps(details, indent=2))


if __name__ == '__main__':
    main()
