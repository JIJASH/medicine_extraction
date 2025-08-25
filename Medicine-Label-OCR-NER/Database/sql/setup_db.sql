-- Create database if it does not exist
-- Database name 'medicine'
CREATE DATABASE IF NOT EXISTS medicine;
USE medicine;

-- Create table in database
-- Table name 'med-info'
CREATE TABLE IF NOT EXISTS med_info (
    ID INT PRIMARY KEY,
    DRUG_NAME TEXT,
    DOSAGE_FORM_AND_STRENGTH LONGTEXT,
    INDICATIONS LONGTEXT,
    CONTRAINDICATIONS_OR_PRECAUTIONS LONGTEXT,
    DOSAGE_SCHEDULE LONGTEXT,
    ADVERSE_EFFECTS LONGTEXT,
    DRUG_AND_FOOD_INTERACTIONS LONGTEXT
);

-- Load the CSV file data (adjust the file path as needed)
LOAD DATA LOCAL INFILE '../csv-files/data.csv'
INTO TABLE your_table_name
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
