import numpy as np
from read_data_to_matrix import path_to_matrix
from matrix_factorization import matrix_factorization
import os
from utils import read_pickle, grab_highest_rated, get_genre_vector
import pandas as pd

input_read_file = "./dataset/ml-latest-small/"
output_read_file = "dataset/UM_dictionary.pkl"
input_mf_file="./dataset/UM_dictionary.pkl"
output_mf_file="./dataset/matrix_factorized.pkl"

if not os.path.exists(output_read_file):
    path_to_matrix(input_read_file, output_read_file)

if not os.path.exists(output_mf_file):
    matrix_factorization(input_mf_file, output_mf_file)


data_read = read_pickle(output_read_file)
data_mf= read_pickle(output_mf_file)


movieID=grab_highest_rated(matrix=data_read["matrix"], unique_movies=data_read["unique_movies"], rec_movies=30)

movie_names = pd.read_csv(input_read_file+"movies.csv")
watched_movies=[]
liked_genre="Action"
user_ratings={}
user_ratings["watched_moviesID"]=[]
user_ratings["ratings"]=[]

print("Watched Movies")
for i, id in enumerate(movieID):
    num=movie_names.index[movie_names["movieId"]==id].item()
    movie_genre=movie_names.iloc[num]["genres"]
    movie_title=movie_names.iloc[num]["title"]


    watched_movies.append(movie_title)
    user_ratings["watched_moviesID"].append(id)
    if liked_genre in movie_genre:
        user_ratings["ratings"].append(5)
    else:
        user_ratings["ratings"].append(1)
    rat=user_ratings["ratings"][-1]
    print(f"{i} {movie_title}: {rat} {movie_genre}")
print("----------------")





#predicted movie
predicted_ratings=get_genre_vector(user_ratings, data_read["unique_movies"], data_mf["V"])
watch_threshold=0.05
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
    print(f"{i}: {movie_title} {pred_rating} {movie_genre}")

    predicted_ratings[max_index]=0
print("----------------")








