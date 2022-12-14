
from flask import Blueprint,request, render_template
import pandas as pd
from website.recommedation.content_based_filtering import get_dict_id_name,get_dict_movie, get_recommended_movies, get_ratings_content,val_to_key
from website.recommedation.collaborative_filtering import get_ratings_collaborative
from website.recommedation.utils import read_pickle, grab_highest_rated, get_poster
from flask_login import login_required, current_user
from .models import Watched, Recommendation
from . import db
import numpy as np
movie = Blueprint('movie', __name__)

ratings = pd.read_csv('website/recommedation/dataset/ml-latest-small/ratings.csv')
movies = pd.read_csv('website/recommedation/dataset/ml-latest-small/movies.csv')
alpha=1.0
output_read_file = "website/recommedation/dataset/UM_dictionary_old.pkl"
input_read_file = "website/recommedation/dataset/ml-latest-small/"
data_read = read_pickle(output_read_file)

user_ratings2={}
user_ratings_ids = {}
movie_names = pd.read_csv(input_read_file+"movies.csv")

def get_movie_id(names, dict_id_name):
    movieIds=[]
    for name in names:
        for key, val in dict_id_name.items():
            if name.lower() in val.lower():
                break
        movieIds.append(key)
    return movieIds

def get_movie_ids(user_rating_dict):
    dict_id_name = get_dict_id_name()
    for movieKey, rating in user_rating_dict.items():
        for key, val in dict_id_name.items():
            if movieKey.lower() in val.lower():
                break
        user_ratings_ids[key] = rating
    return user_ratings_ids


def list_to_string(lists ):
    string = ""
    for i, l in enumerate(lists):
        if i == len(lists)-1:
            string += str(l)
        else:
            string += str(l)+","
    return string
def string_to_list(string):
    lists=string.split(",")
    print(lists)
    return [int(l) for l in lists if len(l)>0]



def get_recommendataions(user_ratings):
    dict_movie_id, _ = get_dict_movie()
    dict_id_name = get_dict_id_name()
    pred_ratings_content = get_ratings_content(user_ratings)
    pred_ratings_collaborative = get_ratings_collaborative(user_ratings)

    pred_ratings=alpha*pred_ratings_content.reshape(-1)+(1-alpha)*pred_ratings_collaborative.reshape(-1)


    for key, val in user_ratings.items():
        print(dict_id_name[key], val)

    print("================")
    print("Predicted Movies")

    output = ["Recommended Movies List"]
    movie_ids = []
    for i in range(5):
        ind = np.argmax(pred_ratings)
        movie_id_pred = val_to_key(ind, dict_movie_id)
        movie_ids.append(movie_id_pred)
        print(dict_id_name[movie_id_pred], pred_ratings[ind], movie_ids)
        pred_ratings[ind] = -1e8

        output.append(dict_id_name[movie_id_pred])

    return output, movie_ids
#----------------------------------------------------------------------------------------------------------------------------------------------------

# root api direct to index.html (home page)
@movie.route('/')
@login_required
def home():
    return render_template('index2.html', user=current_user)


@movie.route('/home_page', methods =['POST'])
@login_required
def home_page():
    return render_template('index2.html', user=current_user)

@movie.route('/rate_movies',methods=['GET','POST'])
@login_required
def rate_movies():
    movieIds = grab_highest_rated(matrix=data_read["matrix"],
                                  unique_movies=data_read["unique_movies"],
                                  rec_movies=7)
    print(movieIds)
    print(current_user.id)
    for r in current_user.recommended:
        movieIds=string_to_list(r.data)

    movie_titles = []
    for id in movieIds:
        print(id)
        num = movie_names.index[movie_names["movieId"] == id].item()
        movie_titles.append(movie_names.iloc[num]["title"])
    posters=get_poster(movieIds)
    print("length",len(posters), len(movie_titles))
    return render_template('index2.html', user=current_user, top_rated = movie_titles, posters=posters)

@movie.route('/submit_top_rated', methods=['POST'])
@login_required
def submit_top_rated():
    content = 0
    movieName = 'default'
    features = [str(x) for x in request.form.values()]
    movie_rating = int(features[0])
    if request.method == 'POST':
        if 'movie' in request.form:
            movieName = (request.form['movie'])

    new_watched = Watched(data=f"{movieName}:{movie_rating}", user_id=current_user.id)
    db.session.add(new_watched)
    db.session.commit()
    for w in current_user.rated:
        print(w.data)
    user_ratings2[movieName] = movie_rating
    return '', 204

@movie.route('/recommend2', methods=['POST'])
@login_required
def recommend2():
    user_ratings={}
    for w in current_user.rated:
        lists=w.data.rsplit(":", 1)
        if len(lists) > 1:
            name=lists[0]

            rating=int(lists[1])
            user_ratings[name]=rating

    user_ratings_ids=get_movie_ids(user_ratings)
    output, movie_ids =  get_recommendataions(user_ratings_ids)

    new_recommendation=Recommendation(data=list_to_string(movie_ids), user_id=current_user.id)
    db.session.add(new_recommendation)
    db.session.commit()
    print("Recommended movie ids")
    for r in current_user.recommended:
        print(r.data)
    posters = get_poster(movie_ids)
    return render_template('index.html', user=current_user, recommended_movie=output, posters=posters)













