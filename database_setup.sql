-- ===============================================================================
-- E-KYC DATABASE SETUP SCRIPT
-- ===============================================================================
-- This script creates the required tables for the E-KYC application
-- Execute this in phpMyAdmin or MySQL command line
-- ===============================================================================

-- Create database (if not exists)
CREATE DATABASE IF NOT EXISTS `ekyc` 
DEFAULT CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

-- Use the ekyc database
USE `ekyc`;

-- ===============================================================================
-- TABLE 1: PAN (For PAN Card registrations)
-- ===============================================================================
CREATE TABLE IF NOT EXISTS `pan` (
  `id` varchar(128) NOT NULL COMMENT 'Hashed PAN ID (SHA-256)',
  `original_id` varchar(255) DEFAULT NULL COMMENT 'Original PAN number',
  `name` varchar(255) DEFAULT NULL COMMENT 'Full name from PAN card',
  `father_name` varchar(255) DEFAULT NULL COMMENT 'Father''s name from PAN card',
  `dob` date DEFAULT NULL COMMENT 'Date of birth',
  `id_type` varchar(50) DEFAULT NULL COMMENT 'Always "PAN"',
  `embedding` longtext COMMENT 'Face embeddings (JSON array)',
  `face_image` longblob COMMENT 'Stored face image (JPEG binary)',
  PRIMARY KEY (`id`),
  INDEX `idx_original_id` (`original_id`),
  INDEX `idx_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='PAN Card registrations';

-- ===============================================================================
-- TABLE 2: AADHARCARD (For Aadhar Card registrations)
-- ===============================================================================
CREATE TABLE IF NOT EXISTS `aadharcard` (
  `id` varchar(128) NOT NULL COMMENT 'Aadhar number (12 digits with spaces)',
  `original_id` varchar(255) DEFAULT NULL COMMENT 'Original Aadhar number',
  `name` varchar(255) DEFAULT NULL COMMENT 'Full name from Aadhar card',
  `gender` varchar(20) DEFAULT NULL COMMENT 'Gender (Male/Female)',
  `dob` date DEFAULT NULL COMMENT 'Date of birth',
  `id_type` varchar(50) DEFAULT NULL COMMENT 'Always "AADHAR"',
  `embedding` longtext COMMENT 'Face embeddings (JSON array)',
  `face_image` longblob COMMENT 'Stored face image (JPEG binary)',
  PRIMARY KEY (`id`),
  INDEX `idx_original_id` (`original_id`),
  INDEX `idx_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Aadhar Card registrations';

-- ===============================================================================
-- VERIFY TABLES CREATED
-- ===============================================================================
SHOW TABLES;

-- Show structure of pan table
DESCRIBE pan;

-- Show structure of aadharcard table
DESCRIBE aadharcard;

-- ===============================================================================
-- SAMPLE QUERIES (for testing in phpMyAdmin)
-- ===============================================================================

-- View all PAN records
-- SELECT original_id, name, father_name, dob, 
--        LENGTH(face_image) as face_size_bytes 
-- FROM pan 
-- ORDER BY name;

-- View all Aadhar records
-- SELECT original_id, name, gender, dob, 
--        LENGTH(face_image) as face_size_bytes 
-- FROM aadharcard 
-- ORDER BY name;

-- Update a specific name (example - change as needed)
-- UPDATE aadharcard 
-- SET name = 'YOUR CORRECT NAME HERE' 
-- WHERE id = '4877 2434 8672';

-- Delete all records (use carefully!)
-- DELETE FROM pan;
-- DELETE FROM aadharcard;

-- ===============================================================================
-- NOTES:
-- ===============================================================================
-- 1. You can EDIT data directly in phpMyAdmin and it will reflect in the app
-- 2. The app reads from database in real-time
-- 3. To fix wrong names: Edit in phpMyAdmin > Browse > Click on row > Edit
-- 4. Face images are stored as BLOB - show as [BLOB] in phpMyAdmin
-- 5. Character set utf8mb4 supports Hindi/Devanagari names
-- ===============================================================================
