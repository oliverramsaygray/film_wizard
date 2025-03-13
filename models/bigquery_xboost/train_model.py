from google.cloud import bigquery
from gcp_lib.params import GCP_PROJECT, BQ_DATASET, TABLE_DATA_TOMATO_REVIEWS_RAW

client = bigquery.Client()

def train_xgboost_model():
    """
    Trains an XGBoost recommendation model in BigQuery ML.
    """
    print("\n🚀 Training BigQuery ML XGBoost recommendation model...")

    query_train_xgboost = f"""
    CREATE OR REPLACE MODEL `{GCP_PROJECT}.{BQ_DATASET}.xgboost_recommender_model`
    OPTIONS(
        MODEL_TYPE='BOOSTED_TREE_CLASSIFIER',
        BOOSTER_TYPE='GBTREE',
        MAX_ITERATIONS=50,
        INPUT_LABEL_COLS=['liked']
    )
    AS
    SELECT
        SAFE_CAST(reviewId AS STRING) AS user_id,
        SAFE_CAST(id AS STRING) AS movie_id,

        -- Normalizované skóre (převedeno na škálu 1-5)
        CASE
            WHEN REGEXP_CONTAINS(originalScore, r'^[0-9\.]+/[1-9][0-9]*$') THEN
                SAFE_DIVIDE(
                    SAFE_CAST(SPLIT(originalScore, '/')[SAFE_OFFSET(0)] AS FLOAT64),
                    SAFE_CAST(SPLIT(originalScore, '/')[SAFE_OFFSET(1)] AS FLOAT64)
                ) * 5
            WHEN originalScore = 'A' THEN 5.0
            WHEN originalScore = 'A-' THEN 4.7
            WHEN originalScore = 'B+' THEN 4.3
            WHEN originalScore = 'B' THEN 4.0
            WHEN originalScore = 'B-' THEN 3.7
            WHEN originalScore = 'C+' THEN 3.3
            WHEN originalScore = 'C' THEN 3.0
            WHEN originalScore = 'C-' THEN 2.7
            WHEN originalScore = 'D+' THEN 2.3
            WHEN originalScore = 'D' THEN 2.0
            WHEN originalScore = 'D-' THEN 1.7
            WHEN originalScore = 'F' THEN 1.0
            ELSE NULL
        END AS normalized_score,

        -- Sentiment (1 = pozitivní, 0 = negativní)
        CASE
            WHEN scoreSentiment = 'POSITIVE' THEN 1
            WHEN scoreSentiment = 'NEGATIVE' THEN 0
            ELSE NULL
        END AS sentiment_score,

        -- Rok z creationDate
        SAFE_CAST(EXTRACT(YEAR FROM PARSE_DATE('%Y-%m-%d', creationDate)) AS INT64) AS movie_year,

        -- Like / Dislike podle normalizovaného skóre
        CASE
            WHEN
                CASE
                    WHEN REGEXP_CONTAINS(originalScore, r'^[0-9\.]+/[1-9][0-9]*$') THEN
                        SAFE_DIVIDE(
                            SAFE_CAST(SPLIT(originalScore, '/')[SAFE_OFFSET(0)] AS FLOAT64),
                            SAFE_CAST(SPLIT(originalScore, '/')[SAFE_OFFSET(1)] AS FLOAT64)
                        ) * 5
                    WHEN originalScore = 'A' THEN 5.0
                    WHEN originalScore = 'A-' THEN 4.7
                    WHEN originalScore = 'B+' THEN 4.3
                    WHEN originalScore = 'B' THEN 4.0
                    WHEN originalScore = 'B-' THEN 3.7
                    WHEN originalScore = 'C+' THEN 3.3
                    WHEN originalScore = 'C' THEN 3.0
                    WHEN originalScore = 'C-' THEN 2.7
                    WHEN originalScore = 'D+' THEN 2.3
                    WHEN originalScore = 'D' THEN 2.0
                    WHEN originalScore = 'D-' THEN 1.7
                    WHEN originalScore = 'F' THEN 1.0
                    ELSE NULL
                END >= 3 THEN 1  -- 👍 Like (>= 3)
            ELSE 0  -- 👎 Dislike (< 3)
        END AS liked

    FROM `{GCP_PROJECT}.{BQ_DATASET}.{TABLE_DATA_TOMATO_REVIEWS_RAW}`
    WHERE originalScore IS NOT NULL
    AND scoreSentiment IS NOT NULL;
    """

    client.query(query_train_xgboost).result()
    print("✅ Model `xgboost_recommender_model` successfully trained using XGBoost!")

if __name__ == "__main__":
    train_xgboost_model()
