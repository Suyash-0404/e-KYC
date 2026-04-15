import mysql.connector
import yaml
import logging

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    db_config = config.get('database', {})

try:
    # Connect to MySQL
    print(f"Connecting to MySQL as user '{db_config.get('user')}' on database '{db_config.get('database')}'...")
    db = mysql.connector.connect(
        host=db_config.get('host', 'localhost'),
        user=db_config.get('user'),
        passwd=db_config.get('password', ''),
        database=db_config.get('database'),
        autocommit=False
    )
    cursor = db.cursor()
    
    print(f'Connected to MySQL database: {db_config.get("database")}')
    
    # Create users table
    print('\nCreating users table...')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
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
    print('users table created/verified')
    
    # Create aadhar table
    print('\nCreating aadhar table...')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS aadhar (
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
    print('aadhar table created/verified')
    
    db.commit()
    
    # Verify tables exist
    print('\nVerifying tables...')
    cursor.execute('SHOW TABLES')
    tables = cursor.fetchall()
    print(f'\nTables in database "{db_config.get("database")}":')
    for table in tables:
        print(f'  {table[0]}')
    
    # Describe users table
    print('\nStructure of users table:')
    cursor.execute('DESCRIBE users')
    columns = cursor.fetchall()
    for col in columns:
        print(f'  - {col[0]}: {col[1]}')
    
    # Describe aadhar table
    print('\nStructure of aadhar table:')
    cursor.execute('DESCRIBE aadhar')
    columns = cursor.fetchall()
    for col in columns:
        print(f'  - {col[0]}: {col[1]}')
    
    db.close()
    print('\nAll tables created successfully!')
    print('\nYou can now:')
    print('1. Access phpMyAdmin at http://localhost/phpmyadmin')
    print('2. View the "ekyc" database')
    print('3. See both "users" and "aadhar" tables')
    print('4. Restart your app: python -m streamlit run app.py --server.port 8502')
    
except Exception as e:
    print(f'Error: {e}')
    print('\nTroubleshooting:')
    print('1. Make sure XAMPP MySQL is running')
    print('2. Check config.yaml has correct credentials')
    print('3. Verify database "ekyc" exists in phpMyAdmin')
