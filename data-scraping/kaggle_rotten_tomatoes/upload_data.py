import os
import pandas as pd
from gcp_lib.clean_data import upload_df_to_bigquery
from gcp_lib.clean_data import load_data_from_csv_to_dataframe
from gcp_lib.clean_data import load_data_from_bigquery
from gcp_lib.clean_data import clean_movies
from gcp_lib.clean_data import clean_movie_reviews
from gcp_lib.params import *

movies_csv = "raw_data/kaggle_rotten_tomatoes/rotten_tomatoes_movies.csv"
df_movies_raw = load_data_from_csv_to_dataframe(movies_csv)
upload_df_to_bigquery(
            df=df_movies_raw,
            table=TABLE_DATA_TOMATO_MOVIES_RAW,
            truncate=True
            )

movie_reviews_csv = "raw_data/kaggle_rotten_tomatoes/rotten_tomatoes_movie_reviews.csv"
df_movie_reviews_raw = load_data_from_csv_to_dataframe(movie_reviews_csv)
upload_df_to_bigquery(
            df=df_movie_reviews_raw,
            table=TABLE_DATA_TOMATO_REVIEWS_RAW,
            truncate=True
        )
