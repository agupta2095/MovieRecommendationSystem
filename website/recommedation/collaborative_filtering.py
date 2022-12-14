import numpy as np
from .read_data_to_matrix import path_to_matrix
from .matrix_factorization import matrix_factorization
import os
from .utils import read_pickle, grab_highest_rated, get_genre_vector
import pandas as pd
from .utils import normalize_predictions
input_read_file = "website/recommedation/dataset/ml-latest-small/"
output_read_file = "website/recommedation/dataset/UM_dictionary.pkl"
input_mf_file= "website/recommedation/dataset/UM_dictionary_old.pkl"
output_mf_file= "website/recommedation/dataset/matrix_factorized.pkl"

if not os.path.exists(output_read_file):
    print(os.getcwd())
    path_to_matrix(input_read_file, output_read_file)

if not os.path.exists(output_mf_file):
    matrix_factorization(input_mf_file, output_mf_file)


def reshape_ratings(pred_ratings):
    pred_ratings=np.concatenate([pred_ratings, np.ones((18,))], axis=0)
    return pred_ratings
def example():
    data_read = read_pickle(output_read_file)
    data_mf= read_pickle(output_mf_file)


    movieID=grab_highest_rated(matrix=data_read["matrix"], unique_movies=data_read["unique_movies"], rec_movies=40)

    movie_names = pd.read_csv(input_read_file+"movies.csv")
    watched_movies=[]
    liked_genre="Comedy"
    user_ratings={}
    user_ratings["watched_moviesID"]=[]
    user_ratings["ratings"]=[]

    print("Watched Movies")
    for i, id in enumerate(movieID):
        num=movie_names.index[movie_names["movieId"]==id].item()
        movie_genre=movie_names.iloc[num]["genres"]

        movie_title=movie_names.iloc[num]["title"]


        watched_movies.append(movie_title)

        if liked_genre in movie_genre:

            user_ratings["watched_moviesID"].append(id)
            user_ratings["ratings"].append(5)
        # else:
        #     user_ratings["ratings"].append(1)
            rat=user_ratings["ratings"][-1]
            print(f"{i} {movie_title}: {rat} {movie_genre}")
    print("----------------")

    #predicted movie
    predicted_ratings=get_genre_vector(user_ratings, data_read["unique_movies"], data_mf["V"])

    watch_threshold=0.2
    total_users=data_read["matrix"].shape[0]


    predicted_ratings=predicted_ratings*(np.sum((data_read["matrix"]>0), axis=0)>(watch_threshold*total_users))
    print("Predicting movies")
    for i in range(10):
        max_index=np.argmax(predicted_ratings)

        predicted_movie_id=data_read["unique_movies"][max_index]


        num=movie_names.index[movie_names["movieId"]==predicted_movie_id].item()
        movie_genre=movie_names.iloc[num]["genres"]
        movie_title = movie_names.iloc[num]["title"]
        pred_rating=predicted_ratings[max_index]

        pred_rating=pred_rating
        print(f"{i}: {movie_title} {int(pred_rating)} {movie_genre}")

        predicted_ratings[max_index]=0
    print("----------------")


def get_ratings_collaborative(user_ratings):
    data_read = read_pickle(output_read_file)
    data_mf = read_pickle(output_mf_file)

    unique_movies=data_read["unique_movies"]
    V=data_mf["V"]
    watch_threshold = 0.2
    total_users = data_read["matrix"].shape[0]

    predicted_ratings = get_genre_vector(user_ratings, unique_movies, V)

    predicted_ratings = predicted_ratings * (np.sum((data_read["matrix"] > 0), axis=0) > (watch_threshold * total_users))
    predicted_ratings = reshape_ratings(predicted_ratings)
    predicted_ratings=normalize_predictions(predicted_ratings)
    return predicted_ratings





