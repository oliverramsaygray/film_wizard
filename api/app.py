import os
from google.cloud import bigquery
from flask import Flask, request, jsonify
from gcp_lib.params import GCP_PROJECT, BQ_DATASET, TABLE_DATA_TOMATO_REVIEWS_RAW
import pandas as pd
from models.svd import svd_predict, svd_cluster_predict, cluster_descriptor

app = Flask(__name__)
client = bigquery.Client()


def make_svd_predictions(df):
    """
    Generates a mockup list of 10 random movies based on input DataFrame.
    """
    if df is not None and not df.empty:
        return df.sample(n=min(10, len(df)))

    # TODO:
    # 1.) movies = SVD_model.predict(df)
    # 2.) remove mockup array "movies = [" below, cause is calculated above in SVD_model.predi..

    # movies = svd_predict(df)
    movies = svd_cluster_predict(df)

    # movies = [
    #     {"movie_id": 1, "title": "Inception", "genre": "Sci-Fi", "rating": 8.8},
    #     {"movie_id": 2, "title": "The Dark Knight", "genre": "Action", "rating": 9.0},
    #     {"movie_id": 3, "title": "Interstellar", "genre": "Sci-Fi", "rating": 8.6},
    #     {"movie_id": 4, "title": "Pulp Fiction", "genre": "Crime", "rating": 8.9},
    #     {"movie_id": 5, "title": "Fight Club", "genre": "Drama", "rating": 8.8},
    #     {"movie_id": 6, "title": "Forrest Gump", "genre": "Drama", "rating": 8.8},
    #     {"movie_id": 7, "title": "The Matrix", "genre": "Sci-Fi", "rating": 8.7},
    #     {"movie_id": 8, "title": "The Lord of the Rings", "genre": "Fantasy", "rating": 8.9},
    #     {"movie_id": 9, "title": "The Godfather", "genre": "Crime", "rating": 9.2},
    #     {"movie_id": 10, "title": "Shawshank Redemption", "genre": "Drama", "rating": 9.3}
    # ]

    return movies # pd.DataFrame(movies)

@app.route('/movie_predictions', methods=['POST'])
def movie_predictions():
    try:
        data = request.get_json()
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Invalid input format. Expected a list of dictionaries.'}), 400

        df = pd.DataFrame(data).dropna()

        if df.empty:
            return jsonify({'error': 'Received empty DataFrame after preprocessing.'}), 400

        predictions_df = make_svd_predictions(df)
        return jsonify({'message': 'Predictions generated', 'predictions': predictions_df.to_dict(orient='records')}), 200
    except Exception as e:
        return jsonify({'error': f'Error generating predictions: {str(e)}'}), 500


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

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Recommendation engine is running'}), 200

@app.route('/cluster_info', methods=['POST'])
def cluster_info():
    try:
        data = request.get_json()
        df = pd.DataFrame(data).dropna()
        output = cluster_descriptor(df)
        return jsonify({'message': 'Cluster info generated', 'info': output.to_dict(orient='records')}), 200

    except Exception as e:
        return jsonify({'error': f'Error generating cluster info: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
