-- Script to initialize the medicine database and schema
-- Creates the database, table, and loads initial data

-- Create database if it does not exist
CREATE DATABASE IF NOT EXISTS medicine;
USE medicine;

-- Create the main medication information table
CREATE TABLE IF NOT EXISTS med_info (
    ID INT PRIMARY KEY,
    DRUG_NAME VARCHAR(255), -- Changed from TEXT for more efficient indexing/querying
    DOSAGE_FORM_AND_STRENGTH TEXT,
    INDICATIONS TEXT,
    CONTRAINDICATIONS_OR_PRECAUTIONS TEXT,
    DOSAGE_SCHEDULE TEXT,
    ADVERSE_EFFECTS TEXT,
    DRUG_AND_FOOD_INTERACTIONS TEXT
);

-- Load the CSV file data into the med_info table
-- Note: The file path must be absolute and the file must be on the same machine as the MySQL server.
-- Using LOCAL requires the MySQL client to have local-infile enabled on the client side.
LOAD DATA LOCAL INFILE '/full/path/to/your/csv-files/data.csv' -- You MUST use an absolute path here
INTO TABLE med_info -- Fixed 'your_table_name' to the correct table name 'med_info'
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"' -- Changed to OPTIONALLY for better handling of quoted fields
LINES TERMINATED BY '\n'
IGNORE 1 ROWS; -- Skip the header row