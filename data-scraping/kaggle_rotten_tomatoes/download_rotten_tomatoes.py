import os

# Define dataset details
DATASET_NAME = "andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews"
RAW_DATA_DIR = "raw_data/kaggle_rotten_tomatoes"

# Ensure the directory exists
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Download dataset
os.system(f"kaggle datasets download -d {DATASET_NAME} -p {RAW_DATA_DIR} --unzip")

print(f"Data downloaded to {RAW_DATA_DIR}")
