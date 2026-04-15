#!/usr/bin/env python3
import mysql.connector
import sys

try:
    # Connect to MySQL
    db = mysql.connector.connect(
        host='localhost',
        user='root',
        password=''
    )
    cursor = db.cursor()
    
    print("Connected to MySQL")
    
    # Drop and create database
    cursor.execute('DROP DATABASE IF EXISTS ekyc')
    print("Dropped existing ekyc database")
    
    cursor.execute('CREATE DATABASE ekyc CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci')
    print("Created ekyc database")
    
    cursor.execute('USE ekyc')
    
    # Create pan table
    cursor.execute('''
        CREATE TABLE pan (
            id varchar(128) NOT NULL,
            original_id varchar(255) DEFAULT NULL,
            name varchar(255) DEFAULT NULL,
            father_name varchar(255) DEFAULT NULL,
            dob date DEFAULT NULL,
            id_type varchar(50) DEFAULT NULL,
            embedding longtext,
            face_image longblob,
            PRIMARY KEY (id),
            INDEX idx_original_id (original_id),
            INDEX idx_name (name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    ''')
    print("Created pan table")
    
    # Create aadharcard table
    cursor.execute('''
        CREATE TABLE aadharcard (
            id varchar(128) NOT NULL,
            original_id varchar(255) DEFAULT NULL,
            name varchar(255) DEFAULT NULL,
            gender varchar(20) DEFAULT NULL,
            dob date DEFAULT NULL,
            id_type varchar(50) DEFAULT NULL,
            embedding longtext,
            face_image longblob,
            PRIMARY KEY (id),
            INDEX idx_original_id (original_id),
            INDEX idx_name (name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    ''')
    print("Created aadharcard table")
    
    db.commit()
    
    # Verify
    cursor.execute('SHOW TABLES')
    tables = [t[0] for t in cursor.fetchall()]
    print(f"\nTables in ekyc database: {tables}")
    
    cursor.close()
    db.close()
    
    print("\nDatabase setup complete!")
    sys.exit(0)
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
