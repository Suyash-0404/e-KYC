#!/usr/bin/env python3
"""Smoke test for face verification using DeepFace.

Usage:
  python smoke_test.py <image1> <image2>

This will run DeepFace.verify for the detector backends: 'opencv' and 'mtcnn'
and print the results (or errors) for each.
"""
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)

try:
    from deepface import DeepFace
except Exception as e:
    logging.error(f"DeepFace import failed: {e}")
    raise

def run_verify(img1, img2, backend):
    try:
        logging.info(f"Running DeepFace.verify for backend='{backend}'")
        res = DeepFace.verify(img1_path=img1, img2_path=img2, detector_backend=backend)
        logging.info(f"Result ({backend}): {res}")
        print(f"--- Backend: {backend} ---")
        print(res)
    except Exception as e:
        logging.error(f"DeepFace.verify failed for backend '{backend}': {e}")
        print(f"--- Backend: {backend} ERROR ---")
        print(str(e))

def main():
    if len(sys.argv) < 3:
        print("Usage: smoke_test.py <image1> <image2>")
        sys.exit(2)

    img1 = sys.argv[1]
    img2 = sys.argv[2]

    # Normalize and check
    img1 = img1.replace('\\', os.sep).replace('/', os.sep)
    img2 = img2.replace('\\', os.sep).replace('/', os.sep)

    print(f"Image 1: {img1} (exists: {os.path.exists(img1)})")
    print(f"Image 2: {img2} (exists: {os.path.exists(img2)})")

    if not os.path.exists(img1) or not os.path.exists(img2):
        print("One or both image paths do not exist. Aborting.")
        sys.exit(3)

    for backend in ("opencv", "mtcnn"):
        run_verify(img1, img2, backend)

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Smoke test for DeepFace.verify using two backends.

Usage:
  python smoke_test.py <img1> <img2>

Prints JSON results for each backend or an error message.
"""
import sys
import os
import json
import traceback

from deepface import DeepFace

def normalize(p):
    return os.path.abspath(p.replace('\\', os.sep).replace('/', os.sep))

def run_verify(img1, img2, backend):
    try:
        res = DeepFace.verify(img1_path=img1, img2_path=img2, detector_backend=backend)
        return {'backend': backend, 'result': res}
    except Exception as e:
        tb = traceback.format_exc()
        return {'backend': backend, 'error': str(e), 'traceback': tb}

def main():
    if len(sys.argv) < 3:
        print("Usage: smoke_test.py <img1> <img2>")
        sys.exit(2)

    img1 = normalize(sys.argv[1])
    img2 = normalize(sys.argv[2])

    print(json.dumps({'img1': img1, 'img2': img2}, indent=2))

    for backend in ['opencv', 'mtcnn']:
        print('\n--- Testing backend:', backend, '---')
        out = run_verify(img1, img2, backend)
        print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Simple smoke-test to run DeepFace.verify on two images.

Usage:
    .venv/bin/python smoke_test.py /path/to/face1.jpg /path/to/face2.jpg

Prints the verification dicts for two detector backends.
"""
import sys
import os
from deepface import DeepFace


def run_verify(img1, img2):
    backends = ['opencv', 'mtcnn']
    results = []
    for backend in backends:
        try:
            print(f"Trying backend: {backend}")
            res = DeepFace.verify(img1_path=img1, img2_path=img2, detector_backend=backend)
            print(f"Result ({backend}): {res}\n")
            results.append((backend, True, res, None))
        except Exception as e:
            print(f"Backend {backend} failed: {e}\n")
            results.append((backend, False, None, str(e)))
    return results


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: smoke_test.py <image1> <image2>")
        sys.exit(1)
    img1 = sys.argv[1]
    img2 = sys.argv[2]

    if not os.path.exists(img1):
        print(f"Image not found: {img1}")
        sys.exit(2)
    if not os.path.exists(img2):
        print(f"Image not found: {img2}")
        sys.exit(2)

    run_verify(img1, img2)
