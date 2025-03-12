from gcp_lib.params import *
from gcp_lib.clean_data import upload_df_to_bigquery
from gcp_lib.clean_data import load_data_from_bigquery
from gcp_lib.clean_data import clean_movies

df_movies_to_clean = load_data_from_bigquery(TABLE_DATA_TOMATO_MOVIES_RAW)
df_movies_to_clean.head(10)

df_movie_reviews_clean = load_data_from_bigquery(TABLE_DATA_TOMATO_REVIEWS_RAW)
df_movie_reviews_clean.head(10)

'''
df_movies_to_clean = clean_movies(df_movies_to_clean)
upload_df_to_bigquery(
            df=df_movies_to_clean,
            table=TABLE_DATA_TOMATO_MOVIES_CLEANED,
            truncate=True
        )
df_movie_reviews_clean = load_data_from_bigquery(TABLE_DATA_TOMATO_REVIEWS_RAW)
df_movie_reviews_clean = clean_movie_reviews(df_movie_reviews_clean)
upload_df_to_bigquery(
            df=df_movies_raw,
            table=TABLE_DATA_TOMATO_REVIEWS_CLEANED,
            truncate=True
        )
'''
