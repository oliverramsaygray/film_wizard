from surprise import SVD, Dataset, Reader
import pandas as pd
# from google.cloud import bigquery

def get_top_n(predictions: list, n: int) -> dict:
    '''
    Given a list of predictions, returns a list of dictionaries in which each
    key is a user and each value is a list of the top n movie IDs with
    associated predicted ratings for that user.

    At the moment the list of predictions is being generated using the
    model.test(dataset) which already knows the user ratings for movies.

    TO DO - Change to model.predict(uid, iid) and refactor to take one user ID,
    i.e. the user we've just added to the dataset, and output their top n rated
    movies as predicted by the model.

    Input:

    predictions is a list in which each entry is ofm the following form;

    Prediction(uid=34613, iid=2003, r_ui=3.5, est=2.3194206306982457, details={'was_impossible': False})

    i.e.

    [
        Prediction(uid=34613, iid=2003, r_ui=3.5, est=2.3194206306982457, details={'was_impossible': False}),
        Prediction(uid=130034, iid=3409, r_ui=2.0, est=2.7961314274559177, details={'was_impossible': False}),
        Prediction(uid=92914, iid=661, r_ui=2.0, est=3.3350042877679957, details={'was_impossible': False}),
        Prediction(uid=140388, iid=93721, r_ui=3.5, est=3.334605249869981, details={'was_impossible': False})
    ]

    uid is the movielens userID, iid is itemID (i.e. movie ID), r_ui
    is the actual rating of movie i by user u and est is the value predicted by
    the model.
    '''

    # This could potentially be rewritten as a list/dict comprehension.
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    # Sort the predictions for each user and return the top n
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def svd_surprise(user_ratings_df: pd.DataFrame) -> None:
    '''
    Uses the surprise package's built-in SVD model to make recommendations to a
    user.

    Input:
        Takes a dataframe with unordered movie ratings from a user in the form
        (Title, Rating).
        E.g.

        title         | rating
        The Big Short | 8.5
        Big           | 8.0
        Aftersun      | 9.0
        The Room      | 10.0

    Output:
        Returns a list of recommended movies (that have not been seen by the
        user) ordered by the ratings predicted by the SVD model.
        E.g.

        title         | rating
        Heat          | 9.5
        Notting Hill  | 8.0
        Grown Ups     | 7.0
        etc
    '''

    # Loading the data from local cache and downsampling for testing.
    # TO DO - Connect to BigQuery?
    movielens_df = pd.read_csv("../raw_data/movielens_ratings.csv")
    movielens_df = movielens_df.iloc[:1_000_000]

    # Initialise a Surprise Reader to build a dataset.
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(movielens_df, reader)

    model = SVD()
    model.fit(data)

    # This needs to be updated.
    predictions = model.predict(data)

    top_n = get_top_n(predictions, n=3)



    return None
