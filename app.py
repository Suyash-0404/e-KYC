import os
import logging
import streamlit as st
import yaml
from preprocess import read_image, extract_id_card, save_image
from ocr_engine import extract_text
from postprocess import extract_information, extract_information1
from face_verification import detect_and_extract_face, deepface_face_comparison, get_face_embeddings
from sql_connection import generate_candidates, fuzzy_name_search
from sql_connection import insert_records, fetch_records, check_duplicacy, insert_records_aadhar, fetch_records_aadhar, check_duplicacy_aadhar
import hashlib

from live_video_verification import main as main_live_verification

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "ekyc_logs.log"),
    level=logging.INFO,
    format=logging_str,
    filemode="a"
)

try:
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    db_config = config.get("database", {})
    db_user = db_config.get("user")
    db_password = db_config.get("password", "")
    logging.info("Configuration loaded successfully")
except Exception as e:
    logging.error(f"Error loading configuration: {e}")
    st.error(f"Error loading configuration: {e}")

def hash_id(id_value):
    hash_object = hashlib.sha256(id_value.encode())
    hashed_id = hash_object.hexdigest()
    return hashed_id

def wider_page():
    # Additional responsive layout settings
    max_width_str = "max-width: 95%;"  # Use 95% for ultra-wide
    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container{{ {max_width_str} }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    logging.info("Page layout set to ultra-wide.")

def set_custom_theme():
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f2f6;
                color: #333333;
            }
            .sidebar .sidebar-content {
                background-color: #ffffff;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    logging.info("Custom theme applied.")

def sidebar_section():
    st.sidebar.title("Select ID Card Type")
    option = st.sidebar.selectbox("ID Type", ("PAN", "AADHAR"))
    logging.info(f"Selected: {option}")
    return option

def header_section(option):
    if option == "AADHAR":
        st.title("Registration Using Aadhar Card")
    elif option == "PAN":
        st.title("Registration Using PAN Card")

def main_content(image_file, face_image_file, option):
    if image_file is not None and face_image_file is not None:
        try:
            face_image = read_image(face_image_file, is_uploaded=True)
            logging.info("Face image loaded.")

            if face_image is not None:
                image = read_image(image_file, is_uploaded=True)
                logging.info("ID card image loaded.")

                image_roi, _ = extract_id_card(image)
                logging.info("ID card ROI extracted.")

                face_image_path2 = detect_and_extract_face(img=image_roi)
                face_image_path1 = save_image(face_image, "face_image.jpg", path="data\\02_intermediate_data")
                logging.info("Faces extracted and saved.")

                is_face_verified, verification_details = deepface_face_comparison(image1_path=face_image_path1, image2_path=face_image_path2)
                logging.info(f"Face verification status: {'successful' if is_face_verified else 'failed'}.")

                if is_face_verified:
                    # Pass card_type to extract_text for intelligent parsing
                    extracted_text = extract_text(image_roi, card_type=option)
                    text_info = extract_information(extracted_text) if option == "PAN" else extract_information1(extracted_text)
                    logging.info("Text extracted and information parsed from ID card.")
                    logging.info(f"PARSED INFO - Name: {text_info.get('Name')}, ID: {text_info.get('ID')}")

                    # ADDED: Ensure original_id is stored
                    if 'original_id' not in text_info or not text_info['original_id']:
                        text_info['original_id'] = text_info.get('ID', '')

                    # If OCR failed to find an ID, allow the operator to enter/confirm it before proceeding
                    if not text_info['original_id']:
                        st.warning("No ID detected from OCR. Please enter the ID manually (you can edit the suggested value).")
                        manual_id = st.text_input("Detected ID (edit if incorrect):", value="")
                        if manual_id:
                            text_info['original_id'] = manual_id.strip()
                    
                    original_id = text_info['original_id']
                    
                    # CRITICAL: Only hash PAN IDs, NOT Aadhar IDs!
                    # Aadhar numbers are already unique 12-digit IDs
                    if option == "PAN":
                        text_info['ID'] = hash_id(original_id)
                        logging.info(f"PAN ID hashed: {original_id} -> {text_info['ID']}")
                    else:  # AADHAR
                        text_info['ID'] = original_id  # Use plain Aadhar number
                        logging.info(f"AADHAR ID (unhashed): {original_id}")

                    is_duplicate = check_duplicacy(text_info) if option == "PAN" else check_duplicacy_aadhar(text_info)

                    if is_duplicate:
                        st.warning(f"User already present with ID {original_id}")
                        logging.info(f"Duplicate user found: {original_id}")
                        records = fetch_records(text_info) if option == "PAN" else fetch_records_aadhar(text_info)
                        if not records.empty:
                            st.write("Existing records:")
                            st.dataframe(records)
                            
                            # Check if face_image exists
                            has_face_image = False
                            if 'face_image' in records.columns:
                                face_blob = records.iloc[0]['face_image']
                                has_face_image = face_blob is not None and len(face_blob) > 0
                            
                            if not has_face_image:
                                st.error("Existing record has NO face_image! You must UPDATE it!")
                                if st.button("UPDATE RECORD WITH FACE IMAGE"):
                                    # Extract face image
                                    logging.info("=" * 80)
                                    logging.info("🔄 UPDATE BUTTON CLICKED - Updating record with face_image")
                                    logging.info(f"face_image_path1: {face_image_path1}")
                                    logging.info(f"File exists: {os.path.exists(face_image_path1) if face_image_path1 else False}")
                                    
                                    try:
                                        import cv2
                                        
                                        # Use correct face path for Aadhar
                                        actual_face_path = face_image_path1
                                        logging.info(f"Processing face from: {actual_face_path}")
                                        
                                        if not actual_face_path or not os.path.exists(actual_face_path):
                                            st.error(f"Face image file not found: {actual_face_path}")
                                            logging.error(f"Face image file not found: {actual_face_path}")
                                        else:
                                            face_img = cv2.imread(actual_face_path)
                                            logging.info(f"cv2.imread result: {type(face_img)}, shape: {face_img.shape if face_img is not None else 'None'}")
                                            
                                            if face_img is not None:
                                                face_resized = cv2.resize(face_img, (224, 224))
                                                text_info['face_image'] = face_resized
                                                text_info['Embedding'] = get_face_embeddings(actual_face_path)
                                                
                                                logging.info(f"Face image extracted: {face_resized.shape}")
                                                logging.info(f"text_info keys: {list(text_info.keys())}")
                                                logging.info(f"Name in text_info: {text_info.get('Name')}")
                                                logging.info(f"Calling {'insert_records' if option == 'PAN' else 'insert_records_aadhar'}")
                                                
                                                # Update database (INSERT OR REPLACE will update)
                                                success = insert_records(text_info) if option == "PAN" else insert_records_aadhar(text_info)
                                                
                                                logging.info(f"Insert result: {success}")
                                                
                                                if success:
                                                    st.success("Record UPDATED with face_image!")
                                                    logging.info("UPDATE SUCCESSFUL!")
                                                    st.session_state.phase = "live"
                                                    try:
                                                        st.rerun()
                                                    except Exception as e:
                                                        logging.warning(f"st.rerun() failed: {e}")
                                                else:
                                                    st.error("Update failed - insert_records returned False!")
                                                    logging.error("insert_records/insert_records_aadhar returned False")
                                            else:
                                                st.error(f"Could not load face image from {face_image_path1}")
                                                logging.error(f"cv2.imread returned None for {face_image_path1}")
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                                        logging.error(f"UPDATE ERROR: {e}", exc_info=True)
                                    
                                    logging.info("=" * 80)
                            else:
                                st.success("Existing record has face_image")

                        # Require user confirmation to proceed to live video verification
                        if st.button("Proceed to Live Video Verification", key="proceed_live_button"):
                            st.session_state.phase = "live"
                            try:
                                st.rerun()
                            except Exception:
                                logging.info("st.rerun() not available; continuing without rerun")
                    else:
                        st.success("New user detected. Processing registration...")
                        st.write("Extracted Information:")
                        display_info = text_info.copy()
                        display_info['ID'] = original_id
                        st.json(display_info)
                        
                        # SHOW ALL DATABASE RECORDS IN HORIZONTAL TABLE
                        st.subheader("Current Database Records")
                        try:
                            import pandas as pd
                            from sql_connection import mydb, sqlite_conn, use_sqlite
                            
                            if not use_sqlite and mydb is not None:
                                # MySQL database
                                cursor = mydb.cursor(buffered=True)
                                if option == "PAN":
                                    cursor.execute("SELECT original_id, name, father_name, dob, id_type FROM pan ORDER BY name LIMIT 50")
                                else:
                                    cursor.execute("SELECT original_id, name, gender, dob, id_type FROM aadharcard ORDER BY name LIMIT 50")
                                rows = cursor.fetchall()
                                if rows:
                                    if option == "PAN":
                                        df = pd.DataFrame(rows, columns=['PAN Number', 'Name', 'Father Name', 'DOB', 'ID Type'])
                                    else:
                                        df = pd.DataFrame(rows, columns=['Aadhaar Number', 'Name', 'Gender', 'DOB', 'ID Type'])
                                    st.dataframe(df, use_container_width=True, height=300)
                                    st.info(f"Showing {len(rows)} records from MySQL database (XAMPP)")
                                else:
                                    st.info("ℹ️ No records in database yet. This will be the first entry!")
                            elif use_sqlite and sqlite_conn is not None:
                                # SQLite database
                                cursor = sqlite_conn.cursor()
                                if option == "PAN":
                                    cursor.execute("SELECT original_id, name, father_name, dob, id_type FROM pan ORDER BY name LIMIT 50")
                                else:
                                    cursor.execute("SELECT original_id, name, gender, dob, id_type FROM aadharcard ORDER BY name LIMIT 50")
                                rows = cursor.fetchall()
                                if rows:
                                    if option == "PAN":
                                        df = pd.DataFrame(rows, columns=['PAN Number', 'Name', 'Father Name', 'DOB', 'ID Type'])
                                    else:
                                        df = pd.DataFrame(rows, columns=['Aadhaar Number', 'Name', 'Gender', 'DOB', 'ID Type'])
                                    st.dataframe(df, use_container_width=True, height=300)
                                    st.info(f"Showing {len(rows)} records from SQLite database")
                                else:
                                    st.info("ℹ️ No records in database yet. This will be the first entry!")
                        except Exception as e:
                            logging.error(f"Error displaying database records: {e}")
                            st.warning(f"Could not load database records: {e}")
                        
                        if hasattr(text_info['DOB'], 'strftime'):
                            text_info['DOB'] = text_info['DOB'].strftime('%Y-%m-%d')
                        
                        # Get face embeddings
                        text_info['Embedding'] = get_face_embeddings(face_image_path1)
                        
                        # EXTRACT AND STORE ACTUAL FACE IMAGE (not just embeddings!)
                        logging.info("=" * 80)
                        logging.info("🖼️  EXTRACTING FACE IMAGE FOR STORAGE")
                        logging.info(f"Face image path: {face_image_path1}")
                        logging.info(f"File exists: {os.path.exists(face_image_path1)}")
                        
                        try:
                            import cv2
                            # Load the uploaded face image
                            face_img = cv2.imread(face_image_path1)
                            logging.info(f"cv2.imread result: {type(face_img)}, shape: {face_img.shape if face_img is not None else 'None'}")
                            
                            if face_img is not None:
                                # Resize face to standard size (224x224) for consistent storage
                                face_resized = cv2.resize(face_img, (224, 224))
                                text_info['face_image'] = face_resized
                                logging.info(f"Face image SUCCESSFULLY extracted! Shape: {face_resized.shape}, dtype: {face_resized.dtype}")
                                logging.info(f"face_image added to text_info dict with key 'face_image'")
                            else:
                                text_info['face_image'] = None
                                logging.error(f"Could not load face image file from {face_image_path1}")
                        except Exception as e:
                            text_info['face_image'] = None
                            logging.error(f"Error extracting face image: {e}", exc_info=True)
                        
                        logging.info(f"🔑 text_info keys before insert: {list(text_info.keys())}")
                        logging.info(f"🔑 'face_image' in text_info: {'face_image' in text_info}")
                        logging.info(f"🔑 face_image is None: {text_info.get('face_image') is None}")
                        logging.info("=" * 80)
                        
                        success = insert_records(text_info) if option == "PAN" else insert_records_aadhar(text_info)

                        if success:
                            st.success("Registration completed successfully! Face image stored.")
                            logging.info(f"New user record inserted with FACE IMAGE: {original_id}")
                            
                            # SHOW UPDATED DATABASE TABLE PERMANENTLY
                            st.subheader("Updated Database Records")
                            try:
                                import pandas as pd
                                from sql_connection import mydb, sqlite_conn, use_sqlite
                                
                                if not use_sqlite and mydb is not None:
                                    cursor = mydb.cursor(buffered=True)
                                    if option == "PAN":
                                        cursor.execute("SELECT original_id, name, father_name, dob, id_type FROM pan ORDER BY name LIMIT 50")
                                    else:
                                        cursor.execute("SELECT original_id, name, gender, dob, id_type FROM aadharcard ORDER BY name LIMIT 50")
                                    rows = cursor.fetchall()
                                    if rows:
                                        if option == "PAN":
                                            df = pd.DataFrame(rows, columns=['PAN Number', 'Name', 'Father Name', 'DOB', 'ID Type'])
                                        else:
                                            df = pd.DataFrame(rows, columns=['Aadhaar Number', 'Name', 'Gender', 'DOB', 'ID Type'])
                                        st.dataframe(df, use_container_width=True, height=300)
                                        st.info(f"Showing {len(rows)} records from MySQL database")
                                elif use_sqlite and sqlite_conn is not None:
                                    cursor = sqlite_conn.cursor()
                                    if option == "PAN":
                                        cursor.execute("SELECT original_id, name, father_name, dob, id_type FROM pan ORDER BY name LIMIT 50")
                                    else:
                                        cursor.execute("SELECT original_id, name, gender, dob, id_type FROM aadharcard ORDER BY name LIMIT 50")
                                    rows = cursor.fetchall()
                                    if rows:
                                        if option == "PAN":
                                            df = pd.DataFrame(rows, columns=['PAN Number', 'Name', 'Father Name', 'DOB', 'ID Type'])
                                        else:
                                            df = pd.DataFrame(rows, columns=['Aadhaar Number', 'Name', 'Gender', 'DOB', 'ID Type'])
                                        st.dataframe(df, use_container_width=True, height=300)
                                        st.info(f"Showing {len(rows)} records from SQLite database")
                            except Exception as e:
                                logging.error(f"Error displaying updated database records: {e}")
                            
                            # BUTTON TO PROCEED TO LIVE VERIFICATION
                            st.markdown("---")
                            st.info("🎥 Ready for live video verification!")
                            if st.button("▶️ Proceed to Live Video Verification", type="primary"):
                                st.session_state.phase = "live"
                                try:
                                    st.rerun()
                                except Exception:
                                    logging.info("st.rerun() not available; continuing without rerun")
                        else:
                            st.error("Error during registration. Please try again.")
                            logging.error(f"Failed to insert record for: {original_id}")
                else:
                    st.error("Face verification failed. The face in the uploaded photo doesn't match the face on the ID card.")
                    logging.warning("Face verification failed.")

                    # Debug panel: show saved image paths, existence, and verification details
                    with st.expander("Debug: face verification details", expanded=True):
                        st.write("Saved image paths:")
                        st.write({
                            'uploaded_face_saved_path': face_image_path1,
                            'extracted_face_from_id_path': face_image_path2
                        })
                        # Show existence
                        exists_info = {}
                        try:
                            exists_info['uploaded_face_exists'] = os.path.exists(face_image_path1) if face_image_path1 else False
                        except Exception:
                            exists_info['uploaded_face_exists'] = False
                        try:
                            exists_info['extracted_face_exists'] = os.path.exists(face_image_path2) if face_image_path2 else False
                        except Exception:
                            exists_info['extracted_face_exists'] = False
                        st.write("File existence:")
                        st.write(exists_info)

                        st.write("DeepFace verification attempts:")
                        st.json(verification_details)

                    # Additional debug: show OCR raw text, parsed fields, DB candidates and fuzzy name matches
                    with st.expander("Debug: OCR & DB lookup", expanded=True):
                        st.write("Raw OCR extracted text:")
                        st.text(extracted_text[:1000] if extracted_text else "(none)")
                        st.write("Parsed fields (text_info):")
                        st.json(text_info)

                        try:
                            candidates = generate_candidates(text_info)
                            st.write("ID candidates tried:")
                            st.write(candidates)
                        except Exception as e:
                            st.write(f"Error generating candidates: {e}")

                        try:
                            fuzzy_df = fuzzy_name_search(text_info)
                            if not fuzzy_df.empty:
                                st.write("Fuzzy name-match candidates:")
                                st.dataframe(fuzzy_df)
                            else:
                                st.write("No fuzzy name matches found (threshold 0.75)")
                        except Exception as e:
                            st.write(f"Error during fuzzy name search: {e}")
            else:
                st.error("Could not process the face image. Please upload a clear face image.")
                logging.error("Face image processing failed.")

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            logging.error(f"Error in main_content: {e}")

    elif image_file is None:
        st.warning("Please upload an ID card image.")
        logging.warning("No ID card image uploaded.")
    elif face_image_file is None:
        st.warning("Please upload a face image.")
        logging.warning("No face image uploaded.")

def main():
    # Set page to wide layout for camera display
    st.set_page_config(
        page_title="E-KYC System",
        page_icon="🔐",
        layout="wide",  # Wide layout for camera
        initial_sidebar_state="expanded"
    )
    
    try:
        # Custom CSS for ultra-wide camera display
        st.markdown("""
        <style>
        /* Ultra-wide video/image container */
        .stImage {
            width: 100% !important;
            max-width: 100% !important;
        }
        .element-container img {
            width: 100% !important;
            max-width: 100% !important;
            height: auto !important;
        }
        /* Make main container use full width */
        .main .block-container {
            max-width: 100% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        /* Custom theme */
        body {
            background-color: #f0f2f6;
            color: #333333;
        }
        </style>
        """, unsafe_allow_html=True)
        
        wider_page()
        set_custom_theme()

        if "phase" not in st.session_state:
            st.session_state.phase = "phase1"

        if st.session_state.phase == "phase1":
            option = sidebar_section()
            header_section(option)
            image_file = st.file_uploader("Upload ID Card", type=['jpg', 'jpeg', 'png'])
            face_image_file = st.file_uploader("Upload Face Image", type=['jpg', 'jpeg', 'png'])

            if image_file is not None and face_image_file is not None:
                main_content(image_file, face_image_file, option)
            elif image_file is not None:
                st.info("Please also upload a face image to proceed with verification.")
                
        elif st.session_state.phase == "live":
            st.header("Phase 2: Live Video Verification")
            main_live_verification()

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logging.error(f"Application error: {e}")

if __name__ == "__main__":
    main()