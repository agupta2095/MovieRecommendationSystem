import csv
import pandas as pd
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup


path='website/recommedation/dataset/ml-latest-small/movies.csv'

df = pd.read_csv(path, delimiter=',', header=None,names=['movie_id', 'title', 'genre'], engine='python')
for i, row in df.iterrows():
    print(i)
    if i==0:
        continue

    movie_id = row['movie_id']
    movie_title = row['title']
    domain = 'http://www.imdb.com'
    search_url = domain + '/find?q=' + urllib.parse.quote_plus(movie_title)

    with urllib.request.urlopen(search_url) as response:

        html = response.read()
        import io
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
