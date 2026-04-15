DROP DATABASE IF EXISTS ekyc;
CREATE DATABASE ekyc CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE ekyc;

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

SHOW TABLES;
