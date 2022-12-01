import pandas as pd
import numpy as np


def get_dict_movie():
  dict_movie_id={}
  data = pd.read_csv("website/recommedation/dataset/ml-latest-small/"+"movies.csv")
  list_movie_id=data["movieId"].unique()
  list_movie_id.sort()
  total_movies=len(list_movie_id)
  for num, id in enumerate(list_movie_id):
    dict_movie_id[id]=num
  return dict_movie_id, total_movies

def get_dict_tag():
  dict_tag={}
  data = pd.read_csv("website/recommedation/dataset/ml-latest-small/"+"tags.csv")
  list_tag=data["tag"].unique()
  total_tags=len(list_tag)
  for num, tag in enumerate(list_tag):
    dict_tag[tag.lower()]=num
  return dict_tag, total_tags


def get_movie_matrix():
  dict_movie_id, total_movies= get_dict_movie()
  dict_tag, total_tags =get_dict_tag()
  movie_tag_matrix=np.zeros((total_tags,total_movies))
  data = pd.read_csv("website/recommedation/dataset/ml-latest-small/" + "tags.csv")
  for _, entries in data.iterrows():


          movieId = entries["movieId"]
          col_num=dict_movie_id[movieId]
          tag=entries["tag"].lower()
          row_num=dict_tag[tag]

          movie_tag_matrix[row_num, col_num]=movie_tag_matrix[row_num, col_num]+1
  return movie_tag_matrix


def get_correlation_matrix(movie_tag_matrix):
  movie_tag_corr=np.matmul(np.transpose(movie_tag_matrix),movie_tag_matrix )
  for i in range(len(movie_tag_corr)):
    movie_tag_corr[i,i]=0
  return movie_tag_corr



def user_rating_to_vector(user_ratings, dict_movie_id):
  user_vector=np.zeros(( len(dict_movie_id), 1))
  for movieId, rating in user_ratings.items():
     row_num=dict_movie_id[movieId]
     user_vector[row_num]=0.5*rating-1.5
  return user_vector


def content_based_ratings(movie_tag_corr, user_ratings, dict_movie_id):
  user_vector=user_rating_to_vector(user_ratings, dict_movie_id)
  pred_ratings=np.matmul(movie_tag_corr, user_vector)

  if pred_ratings.any():
    pred_ratings=4*(pred_ratings-min(pred_ratings))/(max(pred_ratings)-min(pred_ratings))+1.0
  else:
    pred_ratings[:]=3

  for movieId, _ in user_ratings.items():
    ind= dict_movie_id[movieId]
    pred_ratings[ind]=1.0

  return pred_ratings

def val_to_key(val, dictionary):
  return list(dictionary.keys())[list(dictionary.values()).index(val)]


def get_dict_id_name():
  movie_names=pd.read_csv("website/recommedation/dataset/ml-latest-small/"+"movies.csv")
  dict_id_name={}
  for _,entries in movie_names.iterrows():
    dict_id_name[entries["movieId"]]=entries["title"]
  return dict_id_name

def search_movie(search_tag, movie_tag_matrix, dict_tag, dict_movie_id, dict_id_name, num_movies=4):
  for key, _ in dict_tag.items():
    if search_tag.lower() in key.lower():
      tag=key
      break

  row_num=dict_tag[tag]
  movie_vector=movie_tag_matrix[row_num].copy()
  movieIds=[]
  for i in range(num_movies):
    ind=np.argmax(movie_vector)
    movieId=val_to_key(ind, dict_movie_id)
    movieIds.append(movieId)
    movie_vector[ind]=-1e8
  return movieIds, [dict_id_name[movieId] for movieId in movieIds]

def example():
  dict_movie_id, _ = get_dict_movie()
  dict_tag, total_tags =get_dict_tag()
  dict_id_name = get_dict_id_name()
  movie_tag_matrix = get_movie_matrix()
  movie_tag_corr = get_correlation_matrix(movie_tag_matrix)


  movieIds, movieNames=search_movie("caprio", movie_tag_matrix, dict_tag, dict_movie_id, dict_id_name, num_movies=2)

  user_ratings={}

  for movieId in movieIds:
    user_ratings[movieId]=5
  pred_ratings=content_based_ratings(movie_tag_corr, user_ratings, dict_movie_id)


  print("Watched Movies")
  for key, val in user_ratings.items():
    print(dict_id_name[key], val)

  print("================")
  print("Predicted Movies")
  for i in range(5):
    ind=np.argmax(pred_ratings)
    movie_id_pred=val_to_key(ind, dict_movie_id)

    print(dict_id_name[movie_id_pred], pred_ratings[ind])
    pred_ratings[ind]=-1e8



def get_recommended_movies(user_ratings):
  dict_movie_id, _=get_dict_movie()
  dict_id_name =get_dict_id_name()
  movie_tag_matrix = get_movie_matrix()
  movie_tag_corr = get_correlation_matrix(movie_tag_matrix)

  pred_ratings = content_based_ratings(movie_tag_corr, user_ratings, dict_movie_id)


  for key, val in user_ratings.items():
    print(dict_id_name[key], val)

  print("================")
  print("Predicted Movies")

  output=["Recommended Movies List"]
  movie_ids=[]
  for i in range(5):
    ind = np.argmax(pred_ratings)
    movie_id_pred = val_to_key(ind, dict_movie_id)
    movie_ids.append(movie_id_pred)
    print(dict_id_name[movie_id_pred], pred_ratings[ind], movie_ids)
    pred_ratings[ind] = -1e8

    output.append(dict_id_name[movie_id_pred])

  return output,movie_ids


if __name__=="__main__":
  example()





