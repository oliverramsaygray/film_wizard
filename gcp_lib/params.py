import os
from dotenv import load_dotenv

# 🔥 load .env file
if load_dotenv():
    print("✅ .env file successfully loaded.")
else:
    print("⚠️ .env file was not found or could not be loaded.")

# 🔥 Debug: Verify if variables were loaded
GCP_PROJECT = os.getenv("GCP_PROJECT")
BQ_DATASET = os.getenv("BQ_DATASET")

TABLE_DATA_TOMATO_REVIEWS_RAW = os.getenv("TABLE_DATA_TOMATO_REVIEWS_RAW")
TABLE_DATA_TOMATO_REVIEWS_CLEANED = os.getenv("TABLE_DATA_TOMATO_REVIEWS_CLEANED")
TABLE_DATA_TOMATO_MOVIES_RAW = os.getenv("TABLE_DATA_TOMATO_MOVIES_RAW")
TABLE_DATA_TOMATO_MOVIES_CLEANED = os.getenv("TABLE_DATA_TOMATO_MOVIES_CLEANED")

print(f"🔍 Debug: GCP_PROJECT = {GCP_PROJECT}")
print(f"🔍 Debug: BQ_DATASET = {BQ_DATASET}")

missing_vars = [var for var in ["GCP_PROJECT", "BQ_DATASET", "TABLE_DATA_TOMATO_REVIEWS_RAW",
                                "TABLE_DATA_TOMATO_REVIEWS_CLEANED", "TABLE_DATA_TOMATO_MOVIES_RAW",
                                "TABLE_DATA_TOMATO_MOVIES_CLEANED"] if os.getenv(var) is None]

if missing_vars:
    raise ValueError(f"⚠️ Missing variables in .env file: {', '.join(missing_vars)}")
