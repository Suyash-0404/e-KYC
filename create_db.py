#!/usr/bin/env python3
"""Create the database and required tables if they don't exist.

Reads DB connection info from config.yaml. This script will:
- connect to the MySQL server (without selecting a database),
- create the configured database if it doesn't exist,
- create `users` and `aadhar` tables if they don't exist.

Run with the project's virtualenv python:
"/Users/apple/Downloads/ekyc sssss/.venv/bin/python" create_db.py
"""
import mysql.connector
import yaml
import os
import logging

logging.basicConfig(level=logging.INFO)

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("database", {})

def main():
    cfg = load_config()
    user = cfg.get("user")
    password = cfg.get("password", "")
    host = cfg.get("host", "localhost")
    dbname = cfg.get("database")

    if not (user and dbname):
        logging.error("Database user or name missing from config.yaml")
        return

    # Connect without database to create it if necessary
    conn = None
    try:
        conn = mysql.connector.connect(host=host, user=user, password=password)
        conn.autocommit = True
        cur = conn.cursor()
        logging.info(f"Connected to MySQL server at {host} as {user}")

        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{dbname}` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        logging.info(f"Ensured database `{dbname}` exists")

        # Create tables
        cur.execute(f"USE `{dbname}`;")

        create_users = '''
        CREATE TABLE IF NOT EXISTS users (
            id VARCHAR(128) PRIMARY KEY,
            original_id VARCHAR(255),
            name VARCHAR(255),
            father_name VARCHAR(255),
            dob DATE,
            id_type VARCHAR(50),
            embedding LONGTEXT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        '''

        create_aadhar = '''
        CREATE TABLE IF NOT EXISTS aadhar (
            id VARCHAR(128) PRIMARY KEY,
            original_id VARCHAR(255),
            name VARCHAR(255),
            gender VARCHAR(20),
            dob DATE,
            id_type VARCHAR(50),
            embedding LONGTEXT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        '''

        cur.execute(create_users)
        logging.info("Ensured table `users` exists")
        cur.execute(create_aadhar)
        logging.info("Ensured table `aadhar` exists")

        logging.info("Database and tables are ready.")

    except mysql.connector.Error as e:
        logging.error(f"MySQL error: {e}")
        logging.error("If authentication fails, ensure the MySQL server is running and the credentials in config.yaml are correct.")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    main()
