import pandas as pd
import numpy as np
import pickle

#Function to read data from dataset
def path_to_matrix(input_file="website/recommendation/dataset/ml-latest-small/" , output_file="website/recommendation/dataset/UM_dictionary_old.pkl" ):

    data = pd.read_csv(input_file+"ratings.csv")
    for col in data:
      if col == "userId":
        unique_users=data[col].unique()
        unique_users = list(unique_users)
        unique_users.sort()
        ROWS = len(unique_users)
      elif col == "movieId":
        unique_movies=data[col].unique()
        unique_movies=list(unique_movies)
        unique_movies.sort()
        COLS = len(unique_movies)

    user_movie=np.zeros((ROWS, COLS))
    for _, row in data.iterrows():

        user=(row["userId"])
        movie = (row["movieId"])
        row_num=unique_users.index(user) #row_num of matrix
        col_num=unique_movies.index(movie)#col_num of matrix
        user_movie[row_num, col_num]=row["rating"]
    UM={}
    UM["matrix"]=user_movie
    UM["unique_users"]=unique_users
    UM["unique_movies"]=unique_movies

    f = open(output_file, "wb")

    # write the python object (dict) to pickle file
    pickle.dump(UM, f)

    # close file
    f.close()









