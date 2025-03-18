from surprise import SVD, Dataset, Reader
import pandas as pd
from google.cloud import bigquery
from fuzzywuzzy import process, fuzz

# Imports for testing
import time as t

def fuzzy_match(df1: pd.DataFrame, col1: str, df2: pd.DataFrame, col2: str, threshold: int = 80) -> pd.DataFrame:
    '''
    Takes a dataframe of the users rated movies and finds the corresponding
    entries in the grouplens dataset, returning the user's rating of the movie,
    the exact title of the movie, its grouplens movie ID, and grouplens score.
    Returns a dataframe with matching info.
    '''
    matched_data = []
    choices = df2[col2].tolist() # Convert column to a list

    for _, row in df1.iterrows():
        name = row[col1]
        rating = row['rating'] # Retain 'Rating' column
        result = process.extractOne(name, choices, scorer=fuzz.ratio) # Fuzzy match, uses Levenshtein/edit distance

        if result:
            match, score = result[:2] # Extract match title and score

            if score >= threshold:
                movie_id = df2.loc[df2[col2] == match, 'movieId'].values  # Get movieId
                movie_id = movie_id[0] if len(movie_id) > 0 else None
            else:
                match, movie_id = None, None
        else:
            match, score, movie_id = None, 0, None

        matched_data.append((movie_id, name, match, rating, score))

    return pd.DataFrame(matched_data, columns=['movieId', col1, 'matched_title', 'rating', 'fuzzy_score'])

def svd_predict(new_user_ratings_df: pd.DataFrame, use_local_ratings_for_testing: bool = True) -> pd.DataFrame:
    '''
    Uses the surprise package's built-in SVD model to make recommendations to a
    user.

    Input:
        Takes a dataframe with unordered movie ratings from a user with at least
        the following columns;

        name, year, rating
        E.g.

        TO DO - Make example

    Output:
        Returns a list of recommended movies (that have not been seen by the
        user) ordered by the ratings predicted by the SVD model.
        E.g.

        title         | rating
        Heat          | 5.0
        Notting Hill  | 3.0
        Grown Ups     | 3.5
        etc
    '''

    ####### Preprocessing #######
    #############################

    # If users have rated out of 10 instead of 5, divide by 2. Does not catch
    # critical reviewers but MANAGEMENT has assured us we can assume there are
    # few or even no such customers.
    if ( new_user_ratings_df['rating'] >= 6 ).any():
        new_user_ratings_df['rating'] = new_user_ratings_df['rating'] / 2.0

    new_user_ratings_df = new_user_ratings_df[new_user_ratings_df['year'] <= 2022] # Grouplens has data up to 2022
    new_user_ratings_df = new_user_ratings_df[['title','rating']] #(columns=['date', 'year', 'letterboxduri'], inplace=True)
    new_user_ratings_df.dropna(inplace=True)

    # Loading the data from local cache and downsampling for testing.
    # Loading the local data breaks the fuzzy matching function
    # because it is expecting a dataframe with a title column.

    if use_local_ratings_for_testing:
        print("Loading mlratings locally")
        mlratings_time_start = t.time()
        grouplens_df = pd.read_csv("../raw_data/movielens_ratings.csv")
        mlratings_time_end = t.time()
        print(f'Loading mlratings took {mlratings_time_end-mlratings_time_start} seconds')
        grouplens_df = grouplens_df.iloc[:1_000_000]
    else:
        sample_query = """
            SELECT movieId, title
            FROM `film-wizard-453315.Grouplens.grouplens_movies`
        """

        client = bigquery.Client(project="film-wizard-453315")
        print("Loading glratings from BQ")
        glratingsbq_time_start = t.time()
        grouplens_df = client.query(sample_query).to_dataframe()
        glratingsbq_time_end = t.time()
        print(f'Loading glratings from BQ took {glratingsbq_time_end-glratingsbq_time_start} seconds')

    print("Starting fuzzy match")
    fuzz_time_start = t.time()
    matches_df = fuzzy_match(new_user_ratings_df, 'title', grouplens_df, 'title')
    fuzz_time_end = t.time()
    print(f'Fuzzy matching took {fuzz_time_end-fuzz_time_start} seconds')

    matches_df.dropna(inplace=True) # Drop where fuzzyscore is less than 80

    new_user_id = grouplens_df['userId'].max() + 1 # Ensure ID assigned to new user is unique
    matches_df['userId'] = new_user_id

    grouplens_df = pd.concat([grouplens_df, matches_df[['userId', 'movieId', 'rating']]], ignore_index=True)

    ############ SVD ############
    #############################

    # Build dataset with surprise tools
    reader = Reader(rating_scale=(0.5, 5))
    print("Building surprise dataset")
    sdata_time_start = t.time()
    data = Dataset.load_from_df(grouplens_df, reader)
    sdata_time_end = t.time()
    print(f'Building surprise dataset took {sdata_time_end-sdata_time_start} seconds')

    model = SVD()
    print("Fitting SVD")
    svd_time_start = t.time()
    model.fit(data)
    svd_time_end = t.time()
    print(f'Fitting SVD took {svd_time_end-svd_time_start} seconds')

    ####### Postprocessing #######
    ##############################

    all_movie_ids = grouplens_df['movieId']
    all_movie_ids = all_movie_ids.unique()

    predictions_list = []

    # Loop through all movie IDs and make predictions
    print("Getting predictions")
    pred_time_start = t.time()
    for movieId in all_movie_ids:
        prediction = model.predict(new_user_id, movieId)
        predictions_list.append({'movieId': movieId, 'estimated rating': prediction.est}) # Rename 'prediction' to 'estimated rating'
    pred_time_end = t.time()
    print(f'Getting predictions took {pred_time_end-pred_time_start} seconds')

    # Build a dataframe from results and clean for output
    predictions_df = pd.DataFrame(predictions_list)

    predictions_sorted_df = predictions_df.sort_values(by='estimated rating', ascending=False)

    predictions_df = predictions_sorted_df.merge(grouplens_df[['movieId', 'title']], on='movieId', how='inner')

    # top_n = get_top_n(predictions, n=3)
    print("###############################\n###############################\n###############################\n###############################\n")
    return predictions_df.head(10)

# Testing
if __name__ == "__main__":
    olivia_df = pd.read_csv("../raw_data/olivia_movies.csv")
    # frontend_test_csv = pd.read_csv("../raw_data/sample_ratings_from_frontend.csv")

    print(olivia_df.keys())
    print("-----------------------")
    print(svd_predict(new_user_ratings_df = olivia_df, use_local_ratings_for_testing = False))
