import pickle
import random
import numpy as np
def read_pickle(file_path):
    file_to_read = open(file_path, "rb")
    data_read = pickle.load(file_to_read)
    return data_read


def random(x):
    return 2*x

def grab_highest_rated(matrix, unique_movies, rec_movies):
    average_movie_rating=np.sum(matrix, axis=0)/np.sum((matrix>0), axis=0)
    total_users=matrix.shape[0]
    watch_threshold=0.1
    average_movie_rating=average_movie_rating*(np.sum((matrix>0), axis=0)>(watch_threshold*total_users))
    indices=sorted(range(len(average_movie_rating)), key=lambda x: -average_movie_rating[x])[:rec_movies]
    return [unique_movies[ind] for ind in indices]


def get_genre_vector(user_ratings,unique_movies, V):

    movieIDs=user_ratings["watched_moviesID"]

    movie_ratings=np.array((user_ratings["ratings"]))
    movie_ratings=(movie_ratings-3)/2.0
    V_watched=np.zeros((V.shape[0], len(movieIDs)))
    for i, movie_index in enumerate(movieIDs):
        col_num=unique_movies.index(movie_index)
        V_watched[:,i]=V[:,col_num]
    inverse=np.linalg.inv(np.matmul(V_watched,np.transpose(V_watched)))
    principal=np.matmul(inverse,V_watched)

    user_genre=np.matmul(movie_ratings,np.transpose(principal))
    predicted_rating=np.matmul(user_genre, V)
    return predicted_rating






