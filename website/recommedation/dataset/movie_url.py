import csv
import pandas as pd
import urllib.parse
import urllib.request
import io

#Function to obtain movie posters from IMDB site
#The movie posters URL are stored in movies.csv along with movie ID
path='website/recommedation/dataset/ml-latest-small/movies.csv'

df = pd.read_csv(path, delimiter=',', header=None,names=['movie_id', 'title', 'genre'], engine='python')
for i, row in df.iterrows():
    if i==0:
        continue
    movie_id = row['movie_id']
    if int(movie_id)<=6528:
        continue
    movie_title = row['title']
    domain = 'http://www.imdb.com'
    search_url = domain + '/find?q=' + urllib.parse.quote_plus(movie_title)
    print(search_url)
    try:
        response= urllib.request.urlopen(search_url)
        html = response.read()

        try:
            f = io.BytesIO(html)
            text = f"{f.read()}"
            str1 = "titlePosterImageModel"
            str2 = ".jpg"
            if "titleResults" in text:
                ind = text.find(str1)
                text_cut = text[ind:ind + 1000]
                ind1 = text_cut.find("https://m.media-amazon.com/images/")
                ind2 = text_cut.find(".jpg")
                link = text_cut[ind1:ind2 + 4]
            with open('website/recommedation/dataset/ml-latest-small/movie_poster.csv', 'a', newline='') as out_csv:
                writer = csv.writer(out_csv, delimiter=',')
                writer.writerow([movie_id, link])
        except AttributeError:
                pass
    except urllib.error.HTTPError:
            pass

