import pickle
import random
import numpy as np

def read_pickle(file_path):
    file_to_read = open(file_path, "rb")
    data_read = pickle.load(file_to_read)
    return data_read


def random(x):
    return 2*x

def normalize_predictions(pred_ratings):
  if pred_ratings.any():
    pred_ratings = 4 * (pred_ratings - min(pred_ratings)) / (max(pred_ratings) - min(pred_ratings)) + 1.0
  else:
    pred_ratings[:] = 3
  return pred_ratings

def grab_highest_rated(matrix, unique_movies, rec_movies):
    average_movie_rating=np.sum(matrix, axis=0)/np.sum((matrix>0), axis=0)
    total_users=matrix.shape[0]
    watch_threshold=0.1
    average_movie_rating=average_movie_rating*(np.sum((matrix>0), axis=0)>(watch_threshold*total_users))
    indices=sorted(range(len(average_movie_rating)), key=lambda x: -average_movie_rating[x])[:rec_movies]
    return [unique_movies[ind] for ind in indices]

#(V*V_transpose)^-1H* V_transpose = u
def get_genre_vector(user_ratings_dict,unique_movies, V):

    if "watched_moviesID" not in user_ratings_dict:
        watched=[]
        ratings=[]
        user_ratings={}
        for key, val in user_ratings_dict.items():
            watched.append(key)
            ratings.append(val)

        user_ratings["watched_moviesID"]=watched
        user_ratings["ratings"]=ratings
    else:
        watched=user_ratings_dict["watched_moviesID"]
        ratings=user_ratings_dict["ratings"]

    movieIDs=watched
    movie_ratings = np.array((ratings)).reshape(1, len(ratings))

    #movie_ratings=np.array((user_ratings["ratings"])).reshape(1,len(user_ratings["ratings"]))
    movie_ratings=(movie_ratings-3)/2.0
    V_watched=np.zeros((V.shape[0], len(movieIDs)))
    for i, movie_index in enumerate(movieIDs):
        col_num=unique_movies.index(movie_index)
        V_watched[:,i]=V[:,col_num]
    tmp_matrix=np.matmul(V_watched,np.transpose(V_watched))

    try:
        inverse=np.linalg.inv(tmp_matrix)
    except np.linalg.LinAlgError:
        energy=np.linalg.norm(tmp_matrix)
        coefficient=1
        tmp_matrix=tmp_matrix+coefficient*np.identity(tmp_matrix.shape[0])
        inverse = np.linalg.inv(tmp_matrix)

    principal=np.matmul(np.transpose(V_watched),(inverse))
    user_genre= np.matmul(movie_ratings, principal)
    predicted_rating=np.transpose((np.matmul(user_genre, V)))

    return predicted_rating.reshape(-1)

def get_poster_link(id):
    import csv
    import pandas as pd

    path = 'website/recommedation/dataset/ml-latest-small/movie_poster.csv'

    df = pd.read_csv(path, delimiter=',', header=None,  engine='python')
    for i, row in df.iterrows():
       if  row[0]==id:
           return row[1]
       elif row[0]>id:
           return ''
    return ''

def get_poster(movie_ids):
    links=[]
    for id in movie_ids:
        link=get_poster_link(id)
        links.append(link)
    return links


