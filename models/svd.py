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


def clusters_filter(user_df, recom_df, clusters_df):
    '''
    Filters recommendations based on the clusters found in user_df.
    Parameters:
    user_df (DataFrame): Contains user movie preferences, including a Cluster column.
    recom_df (DataFrame): Contains movie recommendations, without a Cluster column.
    clusters_df (DataFrame): Contains movie IDs and their corresponding clusters.
    Returns:
    DataFrame: Filtered recommendations based on matching clusters.
    '''
    # Merge user_df with clusters_df to get the cluster info
    user_df = user_df.merge(clusters_df, on='movieId', how='left')
    unique_clusters = user_df['Cluster'].unique()
    # Create a DataFrame of clusters and corresponding movies
    user_df = user_df.dropna()
    # Merge recom_df with clusters_df to get the ‘Cluster’ column
    recom_df = recom_df.merge(clusters_df, on='movieId', how='left')
    # Filter recommendations based on user clusters
    filtered_recom_df = recom_df[recom_df['Cluster'].isin(unique_clusters)]
    # Remove movies that are already in user_df
    filtered_recom_df = filtered_recom_df[~filtered_recom_df['movieId'].isin(user_df['movieId'])]
    return filtered_recom_df

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

    # Loading the dataset of ratings from BQ can take ages, so only take up to BQ_RATINGS_LIMIT
    # Need to decide whether to take randomly or how to deal with bottleneck.
    BQ_RATINGS_LIMIT = 1_000_000

    client = bigquery.Client(project="film-wizard-453315")

    if use_local_ratings_for_testing:
        print("Loading glratings locally")
        glratings_local_time_start = t.time()
        glratings_df = pd.read_csv("../raw_data/movielens_ratings.csv") # Named it wrong locally, it's called grouplens table in BQ
        glratings_local_time_end = t.time()
        print(f'Loading glratings locally took {round(glratings_local_time_end-glratings_local_time_start,2)} seconds')
        glratings_df = glratings_df.iloc[:int(BQ_RATINGS_LIMIT)]
    else:
        glratings_query = f"""
            SELECT userId, movieId, rating
            FROM `film-wizard-453315.Grouplens.raw_grouplens_ratings`
            ORDER BY RAND()
            LIMIT {int(BQ_RATINGS_LIMIT)}
        """
        print("Loading glratings from BQ")
        glratings_BQ_time_start = t.time()
        glratings_df = client.query(glratings_query).to_dataframe()
        glratings_BQ_time_end = t.time()
        print(f'Loading mlratings took {round(glratings_BQ_time_end-glratings_BQ_time_start,2)} seconds')

    glmovies_query = """
        SELECT movieId, title
        FROM `film-wizard-453315.Grouplens.grouplens_movies`
    """

    print("Loading glmovies from BQ")
    glmovies_BQ_time_start = t.time()
    glmovies_df = client.query(glmovies_query).to_dataframe()
    glmovies_BQ_time_end = t.time()
    print(f'Loading glmovies from BQ took {round(glmovies_BQ_time_end-glmovies_BQ_time_start,2)} seconds')

    print("Starting fuzzy match")
    fuzz_time_start = t.time()
    matches_df = fuzzy_match(new_user_ratings_df, 'title', glmovies_df, 'title') #  is users_df in Tigran's notebook
    fuzz_time_end = t.time()
    print(f'Fuzzy matching took {round(fuzz_time_end-fuzz_time_start,2)} seconds')

    matches_df.dropna(inplace=True) # Drop where fuzzyscore is less than 80

    new_user_id = glratings_df['userId'].max() + 1 # Ensure ID assigned to new user is unique
    matches_df['userId'] = new_user_id

    glratings_df = pd.concat([glratings_df, matches_df[['userId', 'movieId', 'rating']]], ignore_index=True)

    ############ SVD ############
    #############################

    # Build dataset with surprise tools
    reader = Reader(rating_scale=(0.5, 5))
    print("Building surprise dataset")
    sdata_time_start = t.time()
    data = Dataset.load_from_df(glratings_df, reader)
    trainset = data.build_full_trainset()
    sdata_time_end = t.time()
    print(f'Building surprise dataset took {round(sdata_time_end-sdata_time_start,2)} seconds')

    model = SVD()
    print("Fitting SVD")
    svd_time_start = t.time()
    model.fit(trainset)
    svd_time_end = t.time()
    print(f'Fitting SVD took {round(svd_time_end-svd_time_start,2)} seconds')

    ####### Postprocessing #######
    ##############################

    all_movie_ids = glmovies_df['movieId'].unique()

    predictions_list = []

    # Loop through all movie IDs and make predictions
    print("Getting predictions")
    pred_time_start = t.time()
    for movieId in all_movie_ids:
        prediction = model.predict(new_user_id, movieId)
        predictions_list.append({'movieId': movieId, 'estimated rating': prediction.est}) # Rename 'prediction' to 'estimated rating'
    pred_time_end = t.time()
    print(f'Getting predictions took {round(pred_time_end-pred_time_start,2)} seconds')

    # Build a dataframe from results and clean for output
    predictions_df = pd.DataFrame(predictions_list)

    predictions_sorted_df = predictions_df.sort_values(by='estimated rating', ascending=False)

    predictions_df = predictions_sorted_df.merge(glmovies_df[['movieId', 'title']], on='movieId', how='inner')

    # top_n = get_top_n(predictions, n=3)
    print("###############################\n###############################\n###############################\n###############################\n")
    return predictions_df.head(10)

def svd_cluster_predict(new_user_ratings_df: pd.DataFrame, use_local_ratings_for_testing: bool = False) -> pd.DataFrame:
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

    # Loading the dataset of ratings from BQ can take ages, so only take up to BQ_RATINGS_LIMIT
    # Need to decide whether to take randomly or how to deal with bottleneck.
    BQ_RATINGS_LIMIT = 1_000_000

    client = bigquery.Client(project="film-wizard-453315")

    if use_local_ratings_for_testing:
        print("Loading glratings locally")
        glratings_local_time_start = t.time()
        glratings_df = pd.read_csv("../raw_data/movielens_ratings.csv") # Named it wrong locally, it's called grouplens table in BQ
        glratings_local_time_end = t.time()
        print(f'Loading glratings locally took {round(glratings_local_time_end-glratings_local_time_start,2)} seconds')
        glratings_df = glratings_df.iloc[:int(BQ_RATINGS_LIMIT)]
    else:
        glratings_query = f"""
            SELECT userId, movieId, rating
            FROM `film-wizard-453315.Grouplens.raw_grouplens_ratings`
            ORDER BY RAND()
            LIMIT {int(BQ_RATINGS_LIMIT)}
        """
        print("Loading glratings from BQ")
        glratings_BQ_time_start = t.time()
        glratings_df = client.query(glratings_query).to_dataframe()
        glratings_BQ_time_end = t.time()
        print(f'Loading mlratings took {round(glratings_BQ_time_end-glratings_BQ_time_start,2)} seconds')

    glmovies_query = """
        SELECT movieId, title
        FROM `film-wizard-453315.Grouplens.grouplens_movies`
    """

    print("Loading glmovies from BQ")
    glmovies_BQ_time_start = t.time()
    glmovies_df = client.query(glmovies_query).to_dataframe()
    glmovies_BQ_time_end = t.time()
    print(f'Loading glmovies from BQ took {round(glmovies_BQ_time_end-glmovies_BQ_time_start,2)} seconds')

    print("Starting fuzzy match")
    fuzz_time_start = t.time()
    matches_df = fuzzy_match(new_user_ratings_df, 'title', glmovies_df, 'title') #  is users_df in Tigran's notebook
    fuzz_time_end = t.time()
    print(f'Fuzzy matching took {round(fuzz_time_end-fuzz_time_start,2)} seconds')

    matches_df.dropna(inplace=True) # Drop where fuzzyscore is less than 80

    new_user_id = glratings_df['userId'].max() + 1 # Ensure ID assigned to new user is unique
    matches_df['userId'] = new_user_id

    glratings_df = pd.concat([glratings_df, matches_df[['userId', 'movieId', 'rating']]], ignore_index=True)

    ############ SVD ############
    #############################

    # Build dataset with surprise tools
    reader = Reader(rating_scale=(0.5, 5))
    print("Building surprise dataset")
    sdata_time_start = t.time()
    data = Dataset.load_from_df(glratings_df, reader)
    trainset = data.build_full_trainset()
    sdata_time_end = t.time()
    print(f'Building surprise dataset took {round(sdata_time_end-sdata_time_start,2)} seconds')

    model = SVD()
    print("Fitting SVD")
    svd_time_start = t.time()
    model.fit(trainset)
    svd_time_end = t.time()
    print(f'Fitting SVD took {round(svd_time_end-svd_time_start,2)} seconds')

    ####### Postprocessing #######
    ##############################

    all_movie_ids = glmovies_df['movieId'].unique()

    predictions_list = []

    # Loop through all movie IDs and make predictions
    print("Getting predictions")
    pred_time_start = t.time()
    for movieId in all_movie_ids:
        prediction = model.predict(new_user_id, movieId)
        predictions_list.append({'movieId': movieId, 'estimated rating': round(prediction.est,1)}) # Rename 'prediction' to 'estimated rating'
    pred_time_end = t.time()
    print(f'Getting predictions took {round(pred_time_end-pred_time_start,2)} seconds')

    # Build a dataframe from results and clean for output
    predictions_df = pd.DataFrame(predictions_list)

    predictions_sorted_df = predictions_df.sort_values(by='estimated rating', ascending=False)

    predictions_df = predictions_sorted_df.merge(glmovies_df[['movieId', 'title']], on='movieId', how='inner')

    cluster_query = """
        SELECT *
        FROM `film-wizard-453315.clustered_movies.clusters_ids_metadata`
    """
    # Fetch data from BigQuery
    cluster_ids_df = client.query(cluster_query).to_dataframe()

    recommedations_df = clusters_filter(matches_df, predictions_df, cluster_ids_df)

     # Fetch `poster_path` using `tmdbId`
    tmdb_ids = recommedations_df["tmdbId"].dropna().astype(str).tolist()
    movie_details_query = f"""
        SELECT tmdbId, poster_path
        FROM `film-wizard-453315.tmdb_metadata.movie_details`
        WHERE tmdbId IN ({", ".join(tmdb_ids)})
    """
    print("🔍 Fetching movie details from BQ using tmdbId")
    movie_details_df = client.query(movie_details_query).to_dataframe()

    print("✅ movie_details_df columns:", movie_details_df.columns)
    print("✅ Sample movie_details_df:\n", movie_details_df.head())

    if "tmdbId" not in movie_details_df.columns:
        print("⚠️ `tmdbId` column not found in movie_details_df, checking alternative names...")
        movie_details_df.rename(columns={"tmdbid": "tmdbId"}, inplace=True)

    # Merge `tmdbId` with recommendations
    recommedations_df = recommedations_df.merge(movie_details_df, on="tmdbId", how="left")

    # Construct full poster URL
    poster_base_url = "https://image.tmdb.org/t/p/w200/"
    recommedations_df["poster_url"] = poster_base_url + recommedations_df["poster_path"].fillna("")

    # top_n = get_top_n(predictions, n=3)
    print("###############################\n###############################\n###############################\n###############################\n")
    
    output = recommedations_df[['imdbId','estimated rating','title','genres','runtime', 'Cluster','poster_url']].head(100)
    output.columns = ['IMDB ID','Estimated Rating', 'Title', 'Genres', 'Duration (min)', 'Cluster','poster_url']
    return output # predictions_df.head(100)


def cluster_descriptor(recom_df):
    """
    Filters cluster_dict based on the unique clusters present in recom_df.

    Parameters:
    recom_df (DataFrame): DataFrame containing movie recommendations with a 'Cluster' column.
    cluster_dict (DataFrame): DataFrame containing cluster descriptions.

    Returns:
    DataFrame: Filtered cluster_dict with only the relevant clusters.
    """
    client = bigquery.Client(project="film-wizard-453315")
    
    # Fetch data from BigQuery
    cluster_dict_query = """ SELECT * FROM `film-wizard-453315.clustered_movies.cluster_dict` """
    cluster_dict = client.query(cluster_dict_query).to_dataframe()

    # Get unique cluster values from recom_df
    unique_clusters = recom_df['Cluster'].unique()

    # Filter cluster_dict where 'Cluster' is in the unique_clusters list
    filtered_cluster_dict = cluster_dict[cluster_dict['Cluster'].isin(unique_clusters)].reset_index(drop = True)

    return filtered_cluster_dict

# Testing
if __name__ == "__main__":
    olivia_df = pd.read_csv("../raw_data/olivia_movies.csv")
    # frontend_test_csv = pd.read_csv("../raw_data/sample_ratings_from_frontend.csv")

    print(olivia_df.keys())
    print("-----------------------")
    print(svd_cluster_predict(new_user_ratings_df = olivia_df, use_local_ratings_for_testing = False))
    #print(svd_predict(new_user_ratings_df = olivia_df, use_local_ratings_for_testing = False))
