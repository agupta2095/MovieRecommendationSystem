wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip

Design a web UI based application to recommend movies to user based on their feedbacks and watch history, and other users rating. The application has two parts:-
1. Web based User UI -> It consists of login page, in which a new user or an existing user
can login. After user logins, he/she will see a movie recommendation page. Depending on whether a new user logins or an old user, the web application will display following:-

(I) If a new user logins, the user will see a list of movies which the user can provide feedback about in terms of rating. Then the user can click on the ‘Recommend Movies’ action button which will display movies based on the user’s feedback. If the user don’t rate any of the listed movies, then the    recommender would show popular movies.

(II) If an old user logins, then user will see the list of the latest recommended movies which were previously recommended to the user. The application will ask the user to rate these movies incase they have been watched by the user. Now depending on user feedback, the recommender will fine tune the list of recommended movies for the user. If the user don’t rate any of the listed movies, then the recommender will show the same set of movies again

2. Recommender System -> This will be composed of two algorithms. First algorithm would use Collaborative filtering (based on matrix factorization) which will enable the system to recommend movies to a given user by incorporating ratings from other users. The second algorithm will be based on content based recommendation (using co-relation matrix) which will recommend movies whose contents are similar to movies liked by a given user. The mathematical formula to calculate the similarity between any two movies could utilize the tags associated with the movie (as given in the Group lens/Movie Lens movies data). The more the tags are shared between two movies the more they are assumed to be similar.

Recommendation from both these algorithm will be combined in order to provide final 5 movies recommendation to the user. These algorithms will be initially trained on already available user ratings and tags which will be obtained from Group/Movie lens movies data and will be continuously updated with new incoming user feedbacks.

CORE FEATURES
1.  Web application
2. Web page for user login
3.  Action buttons to rate movies
4.  Display title, ratings, poster.
5.  Movie Recommender ML model that is interactive and dynamic based on user feedback and watch history
