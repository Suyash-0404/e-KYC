#!/usr/bin/env python3
"""Test script to validate DOB normalization and insertion logic in sql_connection.

This script forces `sql_connection` to use an in-memory SQLite database, creates the
required tables, runs a few inserts with empty and various DOB formats, and prints
the resulting rows so you can confirm DOB is stored as NULL or normalized.
"""
import sqlite3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sql_connection


def setup_inmemory_sqlite():
    # Force module to use sqlite and attach a fresh in-memory DB
    sql_connection.use_sqlite = True
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    cur = conn.cursor()
    sql_connection.sqlite_conn = conn
    sql_connection.sqlite_cursor = cur

    # Create tables as the main module would
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            original_id TEXT,
            name TEXT,
            father_name TEXT,
            dob TEXT,
            id_type TEXT,
            embedding TEXT
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS aadhar (
            id TEXT PRIMARY KEY,
            original_id TEXT,
            name TEXT,
            gender TEXT,
            dob TEXT,
            id_type TEXT,
            embedding TEXT
        )
    ''')
    conn.commit()


def run_tests():
    setup_inmemory_sqlite()

    cases = [
        { 'ID': 'TST1', 'Name': 'Alice', 'Gender': 'Female', 'DOB': '', 'ID Type': 'AADHAR', 'Embedding': [] },
        { 'ID': 'TST2', 'Name': 'Bob', 'Gender': 'Male', 'DOB': '04/04/2005', 'ID Type': 'AADHAR', 'Embedding': [] },
        { 'ID': 'TST3', 'Name': 'Carol', 'Gender': 'Female', 'DOB': '04042005', 'ID Type': 'AADHAR', 'Embedding': [] },
        { 'ID': 'TST4', 'Name': 'Dan', 'Gender': 'Male', 'DOB': '2005-04-04', 'ID Type': 'AADHAR', 'Embedding': [] },
    ]

    for c in cases:
        ok = sql_connection.insert_records_aadhar(c)
        print(f"Inserted {c['ID']}: success={ok}")

    # Query results and print
    cur = sql_connection.sqlite_cursor
    cur.execute('SELECT id, name, dob FROM aadhar ORDER BY id')
    rows = cur.fetchall()
    print('\nStored rows in aadhar table:')
    for r in rows:
        print(r)


if __name__ == '__main__':
    run_tests()
