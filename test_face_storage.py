"""
Test script to verify face_image storage works correctly
"""
import cv2
import numpy as np
import sqlite3
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from sql_connection import insert_records, fetch_records

print("=" * 80)
print("🧪 TESTING FACE IMAGE STORAGE")
print("=" * 80)

# Create a test face image (random 224x224 RGB image)
test_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
print(f"✓ Created test face image: {test_face.shape}, dtype: {test_face.dtype}")

# Create test data
test_data = {
    'ID': 'TEST123456',
    'original_id': 'ABCDE1234F',
    'Name': 'Test User',
    "Father's Name": 'Test Father',
    'DOB': '1990-01-01',
    'ID Type': 'PAN',
    'Embedding': [0.1, 0.2, 0.3],  # Dummy embedding
    'face_image': test_face  # CRITICAL: Face image numpy array
}

print(f"\nTest data keys: {list(test_data.keys())}")
print(f"✓ 'face_image' in test_data: {'face_image' in test_data}")
print(f"✓ face_image type: {type(test_data['face_image'])}")
print(f"✓ face_image shape: {test_data['face_image'].shape}")

# Try to insert
print("\nAttempting database insert...")
try:
    success = insert_records(test_data)
    if success:
        print("INSERT SUCCESSFUL!")
    else:
        print("INSERT FAILED - returned False")
except Exception as e:
    print(f"INSERT FAILED with exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try to fetch back
print("\nFetching record back from database...")
try:
    df = fetch_records({'ID': 'TEST123456', 'original_id': 'ABCDE1234F'})
    if df is not None and not df.empty:
        print(f"FETCH SUCCESSFUL! Got {len(df)} rows")
        
        # Check face_image column
        if 'face_image' in df.columns:
            face_blob = df.iloc[0]['face_image']
            print(f"✓ face_image column exists")
            print(f"✓ face_image type: {type(face_blob)}")
            
            if face_blob is not None:
                face_size = len(face_blob) if hasattr(face_blob, '__len__') else 'N/A'
                print(f"FACE IMAGE STORED! Size: {face_size} bytes")
                
                # Try to decode it
                nparr = np.frombuffer(face_blob, np.uint8)
                decoded_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if decoded_img is not None:
                    print(f"Face image decoded successfully: {decoded_img.shape}")
                else:
                    print(f"Failed to decode face_image")
            else:
                print(f"face_image is NULL in database!")
        else:
            print(f"face_image COLUMN NOT IN DATAFRAME!")
            print(f"Available columns: {list(df.columns)}")
    else:
        print(f"FETCH FAILED - no records returned")
except Exception as e:
    print(f"FETCH FAILED with exception: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("🧪 TEST COMPLETE")
print("=" * 80)
