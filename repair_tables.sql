-- ===============================================================================
-- REPAIR AND OPTIMIZE TABLES (Fix phpMyAdmin "doesn't exist in engine" error)
-- ===============================================================================
-- Copy and run this in phpMyAdmin SQL tab
-- ===============================================================================

USE ekyc;

-- Repair tables
REPAIR TABLE users;
REPAIR TABLE aadhar;

-- Optimize tables
OPTIMIZE TABLE users;
OPTIMIZE TABLE aadhar;

-- Check table status
SHOW TABLE STATUS WHERE Name IN ('users', 'aadhar');

-- Verify tables work
SELECT 'users' as table_name, COUNT(*) as record_count FROM users
UNION ALL
SELECT 'aadhar' as table_name, COUNT(*) as record_count FROM aadhar;

-- Show structure
DESCRIBE users;
DESCRIBE aadhar;

-- ===============================================================================
-- If error persists, run this to recreate tables:
-- ===============================================================================

-- DROP TABLE IF EXISTS users;
-- DROP TABLE IF EXISTS aadhar;

-- Then run database_setup.sql again

-- ===============================================================================
