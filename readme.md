# E-KYC System

This project is a simple electronic KYC web app built with Python. It checks an ID card image and a face image, verifies whether both faces match, reads the text from the ID card, and stores the data in a database if the person is not already registered.

## What the project does

1. The user uploads a PAN card or Aadhaar card image.
2. The user uploads a face photo.
3. The app detects the face on the ID card and compares it with the uploaded face.
4. If the faces match, OCR is used to read the ID card text.
5. The extracted information is cleaned and checked for duplicates.
6. If the record is new, it is saved in MySQL, or SQLite is used as a fallback.
7. The app can then move to live video verification.

## Tech Stack

- Python: main programming language.
- Streamlit: web app interface.
- OpenCV: image processing and face cropping.
- DeepFace: face verification and embeddings.
- EasyOCR: text extraction from ID cards.
- pytesseract: backup OCR engine.
- NumPy: image and numeric array handling.
- pandas: showing database records in table form.
- MySQL: primary database.
- SQLite: fallback local database.
- PyYAML: reading configuration from `config.yaml`.
- hashlib: hashing PAN numbers before storage.

## Main Files

- `app.py`: main Streamlit app. Handles upload, verification, OCR, database insert, and phase switching.
- `preprocess.py`: reads uploaded images, finds the ID card region, and saves intermediate images.
- `face_verification.py`: detects the face on the ID card, compares faces, and creates embeddings.
- `ocr_engine.py`: runs OCR and sends the extracted text to the PAN or Aadhaar parser.
- `pan_intelligent_ocr.py`: smart PAN card parser.
- `aadhar_intelligent_ocr.py`: smart Aadhaar card parser.
- `postprocess.py`: converts OCR text into structured fields like name, ID, DOB, and gender.
- `sql_connection.py`: database connection, insert, fetch, duplicate check, and SQLite fallback.
- `live_video_verification.py`: live camera-based verification phase.
- `utils.py`: helper functions like file checks and YAML loading.
- `create_db.py`, `create_tables.py`, `setup_database.py`: database setup scripts.
- `clear_database.py`: clears stored local records for fresh testing.
- `test_*.py` and `tests/`: debugging and validation scripts.

## Folder Structure

- `assets/`: demo media and visuals.
- `data/01_raw_data/`: sample input images.
- `data/02_intermediate_data/`: cropped and processed images.
- `data/models/`: model and cascade files.
- `logs/`: runtime logs.
- `tests/`: extra validation scripts.

## Requirements

- Python 3.9 or newer.
- MySQL server if you want to use the main database.
- A working camera for live verification.

## Setup

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create `config.yaml` in the project root with your database details:

```yaml
database:
 user: your_username
 password: your_password
 host: localhost
 database: your_database_name
```

5. Make sure MySQL is running if you want to use the main database.

## Run the App

```bash
streamlit run app.py
```

## Database Notes

- PAN numbers are hashed before saving.
- Aadhaar numbers are stored directly because they are already unique IDs.
- If MySQL is unavailable, the app falls back to SQLite in `data/ekyc_local.db`.

## Important Notes

- Do not commit `config.yaml` if it contains real credentials.
- Keep `logs/` and `data/ekyc_local.db` out of GitHub.
- The project is designed for demo and academic use.

## Troubleshooting

- If OCR is weak, try a clearer image with better lighting.
- If face verification fails, use a front-facing photo with good brightness.
- If the database does not connect, check the values in `config.yaml` and confirm MySQL is running.

## Short Presentation Summary

This project automates KYC by combining face verification, OCR, and database checks. It reduces manual work, prevents duplicate registrations, and stores verified user data in a structured way.