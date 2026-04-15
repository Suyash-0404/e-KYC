-- ===============================================================================
-- CLEAN DATABASE SETUP (Fixes tablespace errors)
-- ===============================================================================
-- This completely cleans and recreates the database
-- Run this ENTIRE script in phpMyAdmin SQL tab
-- ===============================================================================

-- Step 1: Drop database completely
DROP DATABASE IF EXISTS `ekyc`;

-- Step 2: Recreate database fresh
CREATE DATABASE `ekyc` 
DEFAULT CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

-- Step 3: Use the new database
USE `ekyc`;

-- Step 4: Create USERS table (for PAN cards)
CREATE TABLE `users` (
  `id` varchar(128) NOT NULL COMMENT 'Hashed PAN ID (SHA-256)',
  `original_id` varchar(255) DEFAULT NULL COMMENT 'Original PAN number',
  `name` varchar(255) DEFAULT NULL COMMENT 'Full name from PAN card',
  `father_name` varchar(255) DEFAULT NULL COMMENT 'Father name from PAN card',
  `dob` date DEFAULT NULL COMMENT 'Date of birth',
  `id_type` varchar(50) DEFAULT NULL COMMENT 'Always "PAN"',
  `embedding` longtext COMMENT 'Face embeddings (JSON array)',
  `face_image` longblob COMMENT 'Stored face image (JPEG binary)',
  PRIMARY KEY (`id`),
  INDEX `idx_original_id` (`original_id`),
  INDEX `idx_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='PAN Card registrations';

-- Step 5: Create AADHAR table
CREATE TABLE `aadhar` (
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

-- Step 6: Verify tables created
SHOW TABLES;

-- Step 7: Show table structures
DESCRIBE users;
DESCRIBE aadhar;

-- ===============================================================================
-- SUCCESS! Tables created fresh without errors
-- ===============================================================================
