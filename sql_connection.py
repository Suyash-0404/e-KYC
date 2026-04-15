import mysql.connector
from datetime import datetime
import pandas as pd
import difflib
import logging
import os 
import yaml
import sqlite3

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "ekyc_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

def load_db_config():
    try:
        with open("config.yaml", 'r') as file:
            config = yaml.safe_load(file)
            return config.get("database", {})
    except FileNotFoundError:
        logging.error("config.yaml not found")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config.yaml: {e}")
        raise

db_config = load_db_config()
db_user = db_config.get("user")
db_password = db_config.get("password", "")
db_host = db_config.get("host", "localhost")
db_name = db_config.get("database")

if not db_user:
    logging.error("Database user not configured")
    raise ValueError("Database user not found")

if not db_name:
    logging.error("Database name not configured")
    raise ValueError("Database name not found")

use_sqlite = False
mydb = None
mycursor = None
sqlite_conn = None
sqlite_cursor = None

try:
    logging.info("Connecting to MySQL...")
    mydb = mysql.connector.connect(
        host=db_host,
        port=8501,
        user=db_user,
        passwd=db_password,
        database=db_name,
        autocommit=False
    )
    mycursor = mydb.cursor(buffered=True)
    
    # Create tables if needed
    mycursor.execute('''
        CREATE TABLE IF NOT EXISTS pan (
            id VARCHAR(255) PRIMARY KEY,
            original_id VARCHAR(255),
            name VARCHAR(255),
            father_name VARCHAR(255),
            dob VARCHAR(255),
            id_type VARCHAR(255),
            embedding TEXT,
            face_image LONGBLOB
        )
    ''')
    mycursor.execute('''
        CREATE TABLE IF NOT EXISTS aadharcard (
            id VARCHAR(255) PRIMARY KEY,
            original_id VARCHAR(255),
            name VARCHAR(255),
            gender VARCHAR(255),
            dob VARCHAR(255),
            id_type VARCHAR(255),
            embedding TEXT,
            face_image LONGBLOB
        )
    ''')
    mycursor.execute('''
        CREATE TABLE IF NOT EXISTS verified (
            id INT AUTO_INCREMENT PRIMARY KEY,
            original_id VARCHAR(255),
            name VARCHAR(255),
            id_type VARCHAR(50),
            face_image LONGBLOB,
            verification_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_original_id (original_id),
            INDEX idx_name (name),
            INDEX idx_id_type (id_type)
        )
    ''')
    mydb.commit()
    
    logging.info("MySQL connection successful (XAMPP)")
    use_sqlite = False
    
except Exception as e:
    logging.warning(f"MySQL connection failed: {e}")
    logging.info("Falling back to SQLite...")
    use_sqlite = True
    
# Fallback to SQLite if MySQL failed
if use_sqlite:
    logging.info("Using SQLite database")
    sqlite_db_path = os.path.join('data', 'ekyc_local.db')
    os.makedirs('data', exist_ok=True)
    sqlite_conn = sqlite3.connect(sqlite_db_path, check_same_thread=False)
    sqlite_cursor = sqlite_conn.cursor()

    # Create tables in SQLite if they don't exist
    sqlite_cursor.execute('''
        CREATE TABLE IF NOT EXISTS pan (
            id TEXT PRIMARY KEY,
            original_id TEXT,
            name TEXT,
            father_name TEXT,
            dob TEXT,
            id_type TEXT,
            embedding TEXT,
            face_image BLOB
        )
    ''')
    sqlite_cursor.execute('''
        CREATE TABLE IF NOT EXISTS aadharcard (
            id TEXT PRIMARY KEY,
            original_id TEXT,
            name TEXT,
            gender TEXT,
            dob TEXT,
            id_type TEXT,
            embedding TEXT,
            face_image BLOB
        )
    ''')
    sqlite_cursor.execute('''
        CREATE TABLE IF NOT EXISTS verified (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_id TEXT,
            name TEXT,
            id_type TEXT,
            face_image BLOB,
            verification_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    sqlite_conn.commit()

    # Add face_image column to existing tables if not present
    try:
        sqlite_cursor.execute("ALTER TABLE pan ADD COLUMN face_image BLOB")
        sqlite_conn.commit()
        logging.info("Added face_image column to pan table")
    except Exception:
        pass  # Column already exists

    try:
        sqlite_cursor.execute("ALTER TABLE aadharcard ADD COLUMN face_image BLOB")
        sqlite_conn.commit()
        logging.info("Added face_image column to aadharcard table")
    except Exception:
        pass  # Column already exists

def insert_records(text_info):
    try:
        # Normalize DOB: convert empty or invalid DOB to None, otherwise format as YYYY-MM-DD
        def _normalize_dob(dob_val):
            if not dob_val:
                return None
            dob_str = str(dob_val).strip()
            if not dob_str:
                return None
            # Try common date formats
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d %b %Y", "%d %B %Y"):
                try:
                    dt = datetime.strptime(dob_str, fmt)
                    return dt.strftime("%Y-%m-%d")
                except Exception:
                    continue
            # If it's numeric like DDMMYYYY or similar, try to coerce
            digits = ''.join([c for c in dob_str if c.isdigit()])
            if len(digits) == 8:
                try:
                    dt = datetime.strptime(digits, "%d%m%Y")
                    return dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
            # Could not parse -> treat as NULL
            return None

        # Normalize DOB
        dob_val = _normalize_dob(text_info.get('DOB'))
        
        # Convert face image to binary
        logging.info("=" * 80)
        logging.info(f\"{'MySQL' if not use_sqlite else 'SQLite'} INSERT - Processing face_image\")
        logging.info(f"'face_image' in text_info: {'face_image' in text_info}")
        logging.info(f"text_info['face_image'] is not None: {text_info.get('face_image') is not None}")
        
        face_image_binary = None
        if 'face_image' in text_info and text_info['face_image'] is not None:
            import cv2
            face_array = text_info['face_image']
            logging.info(f"Face array type: {type(face_array)}, shape: {face_array.shape if hasattr(face_array, 'shape') else 'N/A'}")
            
            ret, buffer = cv2.imencode('.jpg', face_array)
            logging.info(f"cv2.imencode success: {ret}")
            
            if ret:
                face_image_binary = buffer.tobytes()
                logging.info(f\"Face encoded to JPEG! Size: {len(face_image_binary)} bytes\")
            else:
                logging.error(\"cv2.imencode FAILED!\")
        else:
            logging.error(\"NO FACE IMAGE IN text_info OR IT'S NULL!\")
        
        logging.info(f"face_image_binary is None: {face_image_binary is None}")
        logging.info(f"face_image_binary size: {len(face_image_binary) if face_image_binary else 0} bytes")
        logging.info("=" * 80)
        
        if not use_sqlite:
            # MySQL (XAMPP) - use REPLACE INTO
            sql = "REPLACE INTO pan(id, original_id, name, father_name, dob, id_type, embedding, face_image) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            values = (text_info['ID'],
                      text_info.get('original_id', ""),
                      text_info['Name'],
                      text_info.get("Father's Name", ""),
                      dob_val,
                      text_info['ID Type'],
                      str(text_info['Embedding']),
                      face_image_binary)
            
            mycursor.execute(sql, values)
            mydb.commit()
            logging.info(f"MySQL REPLACE successful! face_image_binary was {'SAVED' if face_image_binary else 'NULL'}")
        else:
            # SQLite fallback
            sql = "INSERT OR REPLACE INTO pan(id, original_id, name, father_name, dob, id_type, embedding, face_image) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            values = (text_info['ID'],
                      text_info.get('original_id', ""),
                      text_info['Name'],
                      text_info.get("Father's Name", ""),
                      dob_val,
                      text_info['ID Type'],
                      str(text_info['Embedding']),
                      face_image_binary)
            
            logging.info(f"Executing INSERT with {len(values)} values")
            logging.info(f"Value 8 (face_image_binary) type: {type(values[7])}, is_None: {values[7] is None}")
            
            sqlite_cursor.execute(sql, values)
            sqlite_conn.commit()
            logging.info(f"SQLite COMMIT successful! face_image_binary was {'SAVED' if face_image_binary else 'NULL'}")

        logging.info("Inserted records successfully into users table WITH FACE IMAGE.")
        return True
    except Exception as e:
        logging.error(f"Error inserting records into users table: {e}")
        return False

def insert_records_aadhar(text_info):
    try:
        # Normalize DOB similarly for aadhar
        def _normalize_dob(dob_val):
            if not dob_val:
                return None
            dob_str = str(dob_val).strip()
            if not dob_str:
                return None
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d %b %Y", "%d %B %Y"):
                try:
                    dt = datetime.strptime(dob_str, fmt)
                    return dt.strftime("%Y-%m-%d")
                except Exception:
                    continue
            digits = ''.join([c for c in dob_str if c.isdigit()])
            if len(digits) == 8:
                try:
                    dt = datetime.strptime(digits, "%d%m%Y")
                    return dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
            return None

        dob_val = _normalize_dob(text_info.get('DOB'))
        
        # Convert face image to binary
        logging.info("=" * 80)
        logging.info("PROCESSING AADHAR - Processing face_image")
        logging.info(f"'face_image' in text_info: {'face_image' in text_info}")
        logging.info(f"text_info['face_image'] is not None: {text_info.get('face_image') is not None}")
        
        face_image_binary = None
        if 'face_image' in text_info and text_info['face_image'] is not None:
            import cv2
            face_array = text_info['face_image']
            logging.info(f"Face array type: {type(face_array)}, shape: {face_array.shape if hasattr(face_array, 'shape') else 'N/A'}")
            
            ret, buffer = cv2.imencode('.jpg', face_array)
            logging.info(f"cv2.imencode success: {ret}")
            
            if ret:
                face_image_binary = buffer.tobytes()
                logging.info(f"Face encoded to JPEG! Size: {len(face_image_binary)} bytes")
            else:
                logging.error("cv2.imencode FAILED!")
        else:
            logging.error("NO FACE IMAGE IN text_info OR IT'S NULL!")
        
        logging.info(f"face_image_binary is None: {face_image_binary is None}")
        logging.info(f"face_image_binary size: {len(face_image_binary) if face_image_binary else 0} bytes")
        logging.info("=" * 80)
        
        # Use MySQL if available, otherwise SQLite
        if not use_sqlite and mydb and mycursor:
            logging.info("Using MySQL for Aadhar insert")
            sql = """INSERT INTO aadharcard(id, original_id, name, gender, dob, id_type, embedding, face_image) 
                      VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                      ON DUPLICATE KEY UPDATE 
                      name=VALUES(name), gender=VALUES(gender), dob=VALUES(dob), 
                      embedding=VALUES(embedding), face_image=VALUES(face_image)"""
            
            values = (text_info['ID'],
                      text_info.get('original_id', ""),
                      text_info['Name'],
                      text_info.get("Gender", ""),
                      dob_val,
                      text_info['ID Type'],
                      str(text_info['Embedding']),
                      face_image_binary)
            
            logging.info(f"Executing MySQL INSERT AADHAR with {len(values)} values")
            logging.info(f"Value 8 (face_image_binary) type: {type(values[7])}, is_None: {values[7] is None}")
            
            mycursor.execute(sql, values)
            mydb.commit()
            
            logging.info(f"MySQL AADHAR COMMIT successful! face_image_binary was {'SAVED' if face_image_binary else 'NULL'}")
            logging.info("Inserted records successfully into aadhar table WITH FACE IMAGE (MySQL).")
            return True
            
        elif sqlite_conn and sqlite_cursor:
            logging.info("Using SQLite for Aadhar insert")
            sql = "INSERT OR REPLACE INTO aadhar(id, original_id, name, gender, dob, id_type, embedding, face_image) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            
            values = (text_info['ID'],
                      text_info.get('original_id', ""),
                      text_info['Name'],
                      text_info.get("Gender", ""),
                      dob_val,
                      text_info['ID Type'],
                      str(text_info['Embedding']),
                      face_image_binary)
            
            logging.info(f"Executing SQLite INSERT AADHAR with {len(values)} values")
            logging.info(f"Value 8 (face_image_binary) type: {type(values[7])}, is_None: {values[7] is None}")
            
            sqlite_cursor.execute(sql, values)
            sqlite_conn.commit()
            
            logging.info(f"SQLite AADHAR COMMIT successful! face_image_binary was {'SAVED' if face_image_binary else 'NULL'}")
            logging.info("Inserted records successfully into aadhar table WITH FACE IMAGE (SQLite).")
            return True
        else:
            logging.error("No database connection available!")
            return False
            
    except Exception as e:
        logging.error(f"Error inserting records into aadhar table: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def fetch_records(text_info):
    try:
        # Try multiple matching strategies to tolerate OCR variations
        candidates = []
        if 'original_id' in text_info and text_info.get('original_id'):
            orig = str(text_info['original_id']).strip()
            candidates.extend([orig, orig.replace(' ', ''), orig.upper(), orig.lower()])
        # always also try the hashed ID
        if 'ID' in text_info and text_info.get('ID'):
            candidates.append(text_info['ID'])

        seen = set()
        for val in candidates:
            if val in seen:
                continue
            seen.add(val)
            
            # Use appropriate cursor based on connection type
            if not use_sqlite and mycursor is not None:
                # MySQL (XAMPP)
                sql = "SELECT * FROM pan WHERE original_id = %s OR id = %s"
                mycursor.execute(sql, (val, val))
                result = mycursor.fetchall()
                if result:
                    df = pd.DataFrame(result, columns=[desc[0] for desc in mycursor.description])
                    logging.info(f"Fetched records successfully from MySQL users table using candidate '{val}'.")
                    return df
            elif sqlite_cursor is not None:
                # SQLite fallback
                sql = "SELECT * FROM pan WHERE original_id = ? OR id = ?"
                sqlite_cursor.execute(sql, (val, val))
                result = sqlite_cursor.fetchall()
                if result:
                    df = pd.DataFrame(result, columns=[desc[0] for desc in sqlite_cursor.description])
                    logging.info(f"Fetched records successfully from SQLite users table using candidate '{val}'.")
                    return df

        logging.info("No records found for any candidate values.")
        # If we couldn't find by ID/original_id, try a name-based fallback
        try:
            if 'Name' in text_info and text_info.get('Name'):
                name_val = str(text_info['Name']).strip()
                if name_val:
                    logging.info(f"Attempting name-based lookup for: {name_val}")
                    if not use_sqlite and mycursor is not None:
                        # MySQL
                        sql = "SELECT * FROM pan WHERE name LIKE %s"
                        mycursor.execute(sql, (f"%{name_val}%",))
                        result = mycursor.fetchall()
                        if result:
                            df = pd.DataFrame(result, columns=[desc[0] for desc in mycursor.description])
                            logging.info(f"Fetched records by name from MySQL users table for '{name_val}'.")
                            return df
                    elif sqlite_cursor is not None:
                        # SQLite
                        sql = "SELECT * FROM pan WHERE name LIKE ?"
                        sqlite_cursor.execute(sql, (f"%{name_val}%",))
                        result = sqlite_cursor.fetchall()
                        if result:
                            df = pd.DataFrame(result, columns=[desc[0] for desc in sqlite_cursor.description])
                            logging.info(f"Fetched records by name from SQLite users table for '{name_val}'.")
                            return df
        except Exception as e:
            logging.error(f"Error during name-based lookup: {e}")

        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error fetching records: {e}")
        return pd.DataFrame()
    
def fetch_records_aadhar(text_info):
    try:
        candidates = []
        if 'original_id' in text_info and text_info.get('original_id'):
            orig = str(text_info['original_id']).strip()
            # For Aadhaar: try with spaces, without spaces, and formatted versions
            orig_no_space = orig.replace(' ', '')
            candidates.extend([orig, orig_no_space, orig.upper(), orig.lower()])
            
            # If it's a 12-digit number, try formatted version (XXXX XXXX XXXX)
            if len(orig_no_space) == 12 and orig_no_space.isdigit():
                formatted = f"{orig_no_space[0:4]} {orig_no_space[4:8]} {orig_no_space[8:12]}"
                candidates.append(formatted)
            
        if 'ID' in text_info and text_info.get('ID'):
            id_val = str(text_info['ID']).strip()
            candidates.append(id_val)
            id_no_space = id_val.replace(' ', '')
            candidates.append(id_no_space)
            
            # If it's a 12-digit number, try formatted version
            if len(id_no_space) == 12 and id_no_space.isdigit():
                formatted = f"{id_no_space[0:4]} {id_no_space[4:8]} {id_no_space[8:12]}"
                candidates.append(formatted)

        seen = set()
        for val in candidates:
            if val in seen:
                continue
            seen.add(val)
            logging.info(f"Trying Aadhaar candidate: '{val}'")
            
            # Use appropriate cursor based on connection type
            if not use_sqlite and mycursor is not None:
                # MySQL (XAMPP)
                sql = "SELECT * FROM aadharcard WHERE original_id = %s OR id = %s"
                mycursor.execute(sql, (val, val))
                result = mycursor.fetchall()
                if result:
                    df = pd.DataFrame(result, columns=[desc[0] for desc in mycursor.description])
                    logging.info(f"✓ Fetched records successfully from MySQL aadhar table using candidate '{val}'.")
                    return df
            elif sqlite_cursor is not None:
                # SQLite fallback
                sql = "SELECT * FROM aadharcard WHERE original_id = ? OR id = ?"
                sqlite_cursor.execute(sql, (val, val))
                result = sqlite_cursor.fetchall()
                if result:
                    df = pd.DataFrame(result, columns=[desc[0] for desc in sqlite_cursor.description])
                    logging.info(f"✓ Fetched records successfully from SQLite aadhar table using candidate '{val}'.")
                    return df

        logging.info("No aadhar records found for any candidate values.")
        # Try name-based fallback for aadhar
        try:
            if 'Name' in text_info and text_info.get('Name'):
                name_val = str(text_info['Name']).strip()
                if name_val:
                    logging.info(f"Attempting name-based lookup for aadhar: {name_val}")
                    if not use_sqlite and mycursor is not None:
                        # MySQL
                        sql = "SELECT * FROM aadharcard WHERE name LIKE %s"
                        mycursor.execute(sql, (f"%{name_val}%",))
                        result = mycursor.fetchall()
                        if result:
                            df = pd.DataFrame(result, columns=[desc[0] for desc in mycursor.description])
                            logging.info(f"Fetched MySQL aadhar records by name for '{name_val}'.")
                            return df
                    elif sqlite_cursor is not None:
                        # SQLite
                        sql = "SELECT * FROM aadharcard WHERE name LIKE ?"
                        sqlite_cursor.execute(sql, (f"%{name_val}%",))
                        result = sqlite_cursor.fetchall()
                        if result:
                            df = pd.DataFrame(result, columns=[desc[0] for desc in sqlite_cursor.description])
                            logging.info(f"Fetched SQLite aadhar records by name for '{name_val}'.")
                            return df
        except Exception as e:
            logging.error(f"Error during aadhar name-based lookup: {e}")

        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error fetching records: {e}")
        return pd.DataFrame()

def check_duplicacy(text_info):
    try:
        df = fetch_records(text_info)
        if df.shape[0] > 0:
            logging.info("Duplicate records found.")
            return True
        else:
            logging.info("No duplicate records found.")
            return False
    except Exception as e:
        logging.error(f"Error checking duplicacy: {e}")
        return False
    
def check_duplicacy_aadhar(text_info):
    try:
        df = fetch_records_aadhar(text_info)
        if df.shape[0] > 0:
            logging.info("Duplicate records found.")
            return True
        else:
            logging.info("No duplicate records found.")
            return False
    except Exception as e:
        logging.error(f"Error checking duplicacy: {e}")
        return False

def generate_candidates(text_info):
    """Return the candidate ID strings that would be tried for matching (for debug UI)."""
    candidates = []
    if 'original_id' in text_info and text_info.get('original_id'):
        orig = str(text_info['original_id']).strip()
        candidates.extend([orig, orig.replace(' ', ''), orig.upper(), orig.lower()])
    if 'ID' in text_info and text_info.get('ID'):
        candidates.append(text_info['ID'])
    # dedupe preserving order
    seen = set()
    out = []
    for c in candidates:
        if c not in seen and c:
            seen.add(c)
            out.append(c)
    return out

def fuzzy_name_search(text_info, threshold=0.75):
    """When exact name/ID lookups fail, try fuzzy name matching against users table.

    Returns a DataFrame with matches above threshold (may be empty).
    """
    try:
        if 'Name' not in text_info or not text_info.get('Name'):
            return pd.DataFrame()
        target = str(text_info['Name']).strip()
        if not target:
            return pd.DataFrame()

        # Fetch candidate names from database
        rows = []
        if not use_sqlite and mycursor is not None:
            # MySQL
            mycursor.execute("SELECT id, original_id, name FROM pan")
            rows = mycursor.fetchall()
        elif sqlite_cursor is not None:
            # SQLite
            sqlite_cursor.execute("SELECT id, original_id, name FROM pan")
            rows = sqlite_cursor.fetchall()
        
        results = []
        for r in rows:
            uid, orig_id, name = r[0], r[1], r[2]
            ratio = difflib.SequenceMatcher(None, target.lower(), str(name).lower()).ratio()
            if ratio >= threshold:
                results.append((uid, orig_id, name, ratio))
        if results:
            df = pd.DataFrame(results, columns=['id', 'original_id', 'name', 'score'])
            return df.sort_values('score', ascending=False)
    except Exception as e:
        logging.error(f"Error during fuzzy name search: {e}")
    return pd.DataFrame()

def insert_verified_record(original_id, name, id_type, face_image):
    """
    Insert a verified record into the verified table after successful KYC completion.
    
    Args:
        original_id (str): Original ID number (PAN/Aadhar)
        name (str): Person's name
        id_type (str): Type of ID (PAN/AADHAR)
        face_image (numpy.ndarray): Face image array
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import cv2
        
        # Encode face image to JPEG binary
        face_image_binary = None
        if face_image is not None:
            ret, buffer = cv2.imencode('.jpg', face_image)
            if ret:
                face_image_binary = buffer.tobytes()
                logging.info(f"Face encoded to JPEG for verified table: {len(face_image_binary)} bytes")
            else:
                logging.error("Failed to encode face image for verified table")
                return False
        else:
            logging.error("No face image provided for verified table")
            return False
        
        if not use_sqlite and mydb is not None:
            # MySQL
            cursor = mydb.cursor(buffered=True)
            sql = """INSERT INTO verified (original_id, name, id_type, face_image) 
                     VALUES (%s, %s, %s, %s)"""
            values = (original_id, name, id_type, face_image_binary)
            cursor.execute(sql, values)
            mydb.commit()
            logging.info(f"Verified record inserted into MySQL: {name} ({id_type}: {original_id})")
            cursor.close()
            return True
        elif use_sqlite and sqlite_conn is not None:
            # SQLite
            cursor = sqlite_conn.cursor()
            sql = """INSERT INTO verified (original_id, name, id_type, face_image) 
                     VALUES (?, ?, ?, ?)"""
            values = (original_id, name, id_type, face_image_binary)
            cursor.execute(sql, values)
            sqlite_conn.commit()
            logging.info(f"Verified record inserted into SQLite: {name} ({id_type}: {original_id})")
            cursor.close()
            return True
        else:
            logging.error("No database connection available for verified table insert")
            return False
            
    except Exception as e:
        logging.error(f"Error inserting verified record: {e}")
        return False