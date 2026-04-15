import sqlite3
import hashlib

db = sqlite3.connect('data/ekyc_local.db')
c = db.cursor()

# Check what's in the users table
try:
    c.execute('SELECT id, original_id, name, LENGTH(face_image) FROM users')
    rows = c.fetchall()
    print('=== USERS TABLE ===')
    if rows:
        for row in rows:
            print(f'  ID (hash): {row[0][:30]}...')
            print(f'  Original ID: {row[1]}')
            print(f'  Name: {row[2]}')
            print(f'  Face image size: {row[3] if row[3] else 0} bytes')
            print()
    else:
        print('  (EMPTY)')
        
    # Check what hash we're looking for
    search_id = 'IWRPD8134D'
    hashed = hashlib.sha256(search_id.encode()).hexdigest()
    print(f'\nSearching for PAN: {search_id}')
    print(f'Expected hash: {hashed[:30]}...')
    
    c.execute('SELECT COUNT(*) FROM users WHERE id = ? OR original_id = ?', (hashed, search_id))
    count = c.fetchone()[0]
    print(f'Found {count} matching records')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()

db.close()
