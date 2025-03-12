from google.cloud import bigquery
from gcp_lib.params import GCP_PROJECT, BQ_DATASET, TABLE_DATA_TOMATO_REVIEWS_RAW

client = bigquery.Client()

def train_model():
    """
    Trains a recommendation model using BigQuery ML (Logistic Regression - Free Tier Compatible).
    """
    print("\n🚀 Training BigQuery ML recommendation model (Logistic Regression)...")

    # SQL pro vytvoření modelu
    query_train_model = f"""
    CREATE OR REPLACE MODEL `{GCP_PROJECT}.{BQ_DATASET}.recommender_model`
    OPTIONS(
        MODEL_TYPE='LOGISTIC_REG',  -- ✅ Použití LOGISTIC_REG namísto MATRIX_FACTORIZATION
        INPUT_LABEL_COLS=['liked']
    )
    AS
    SELECT
        SAFE_CAST(reviewId AS STRING) AS user_id,
        SAFE_CAST(id AS STRING) AS movie_id,
        CASE
            WHEN SAFE_CAST(originalScore AS FLOAT64) >= 3 THEN 1  -- 👍 Like (3-5 hvězd)
            ELSE 0  -- 👎 Dislike (0-2 hvězdy)
        END AS liked
    FROM `{GCP_PROJECT}.{BQ_DATASET}.{TABLE_DATA_TOMATO_REVIEWS_RAW}`
    WHERE originalScore IS NOT NULL;
    """

    client.query(query_train_model).result()
    print("✅ Model `recommender_model` successfully trained using Logistic Regression!")

if __name__ == "__main__":
    train_model()
