from google.cloud import bigquery
from flask import Flask, request, jsonify
from gcp_lib.params import GCP_PROJECT, BQ_DATASET, TABLE_DATA_TOMATO_REVIEWS_RAW

app = Flask(__name__)
client = bigquery.Client()

@app.route('/recommend', methods=['GET'])
def recommend():
    """
    Returns movie recommendations for a given user.
    """
    user_id = request.args.get('user_id')

    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400

    try:
        print(f"🔍 Fetching recommendations for user_id={user_id}")

        query = f"""
        WITH user_ratings AS (
            SELECT DISTINCT SAFE_CAST(id AS STRING) AS movie_id
            FROM `{GCP_PROJECT}.{BQ_DATASET}.{TABLE_DATA_TOMATO_REVIEWS_RAW}`
        )
        SELECT movie_id, predicted_liked AS probability_of_liking
        FROM ML.PREDICT(
            MODEL `{GCP_PROJECT}.{BQ_DATASET}.recommender_model`,
            (SELECT movie_id, SAFE_CAST('{user_id}' AS STRING) AS user_id
            FROM user_ratings)
        )
        ORDER BY probability_of_liking DESC
        LIMIT 10;
        """

        print(f"📡 Running BigQuery SQL:\n{query}")

        results = client.query(query).to_dataframe()

        if results.empty:
            print("⚠️ No recommendations found for this user.")
            return jsonify({"message": "No recommendations found for this user."}), 404

        print("✅ Recommendations successfully retrieved!")
        return jsonify({
            "user_id": user_id,
            "recommendations": results.to_dict(orient="records")
        })

    except Exception as e:
        print(f"❌ BigQuery Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommend_xgboost', methods=['GET'])
def recommend_xgboost():
    """
    Returns movie recommendations for a given user using the XGBoost model.
    """
    user_id = request.args.get('user_id')

    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400

    try:
        print(f"🔍 Fetching recommendations for user_id={user_id}")

        query = f"""
        WITH user_ratings AS (
            SELECT DISTINCT
                SAFE_CAST(id AS STRING) AS movie_id,
                -- Normalized score transformation
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
                END AS score,

                -- Sentiment mapping
                CASE
                    WHEN scoreSentiment = 'POSITIVE' THEN 1
                    WHEN scoreSentiment = 'NEGATIVE' THEN 0
                    ELSE NULL
                END AS sentiment_score,

                -- Extracting year from creationDate
                SAFE_CAST(EXTRACT(YEAR FROM PARSE_DATE('%Y-%m-%d', creationDate)) AS INT64) AS movie_year

            FROM `{GCP_PROJECT}.{BQ_DATASET}.{TABLE_DATA_TOMATO_REVIEWS_RAW}`
            WHERE originalScore IS NOT NULL
        )

        SELECT movie_id, predicted_liked_probs[OFFSET(1)] AS probability_of_liking
        FROM ML.PREDICT(
            MODEL `{GCP_PROJECT}.{BQ_DATASET}.xgboost_recommender_model`,
            (
                SELECT
                    movie_id,
                    SAFE_CAST('{user_id}' AS STRING) AS user_id,
                    score,
                    sentiment_score,
                    movie_year
                FROM user_ratings
            )
        )
        ORDER BY probability_of_liking DESC
        LIMIT 10;
        """

        print(f"📡 Running BigQuery SQL:\n{query}")

        results = client.query(query).to_dataframe()

        if results.empty:
            print("⚠️ No recommendations found for this user.")
            return jsonify({"message": "No recommendations found for this user."}), 404

        print("✅ Recommendations successfully retrieved!")
        return jsonify({
            "user_id": user_id,
            "recommendations": results.to_dict(orient="records")
        })

    except Exception as e:
        print(f"❌ BigQuery Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
