"""Database cleanup - removes all records for fresh registration."""
import sqlite3
import os

db_path = "data/ekyc_local.db"

if os.path.exists(db_path):
    print(f"Clearing database: {db_path}")
    db = sqlite3.connect(db_path)
    c = db.cursor()
    
    try:
        c.execute("SELECT COUNT(*) FROM users")
        count = c.fetchone()[0]
        print(f"Users: {count} records")
    except:
        print("Users table: not found")
    
    try:
        c.execute("SELECT COUNT(*) FROM aadhar")
        count = c.fetchone()[0]
        print(f"Aadhar: {count} records")
    except:
        print("Aadhar table: not found")
    
    # Delete all records
    if users_count > 0 or aadhar_count > 0:
        confirm = input("\nDELETE ALL RECORDS? (yes/no): ")
        if confirm.lower() == 'yes':
            try:
                c.execute("DELETE FROM users")
                c.execute("DELETE FROM aadhar")
                db.commit()
                print("\nAll records deleted successfully!")
                print("You can now re-register in Phase 1 with face_image support")
            except Exception as e:
                print(f"\nError deleting records: {e}")
        else:
            print("Cancelled.")
    else:
        print("\nNo records to delete.")
    
    db.close()
else:
    print(f"Database not found at {db_path}")
    print("No cleanup needed - database will be created fresh on first registration")
