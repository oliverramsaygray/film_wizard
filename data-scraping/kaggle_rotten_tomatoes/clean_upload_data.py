import os
import pandas as pd
from gcp_lib.clean_data import clean_data_rotten_tomatoes
from gcp_lib.clean_data import upload_csv_to_bigquery

output_csv = "raw_data/kaggle_rotten_tomatoes/rotten_tomatoes_movie_reviews.csv"

upload_csv_to_bigquery(
            csv_path=output_csv,
            table="TABLE_DATA_TOMATO_REVIEWS_RAW",
            truncate=True
        )

clean_data_rotten_tomatoes()
