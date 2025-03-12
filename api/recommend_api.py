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

if __name__ == '__main__':
    app.run(debug=True)
