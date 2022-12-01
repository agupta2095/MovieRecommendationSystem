import csv
import urllib.request
from bs4 import BeautifulSoup
import time 

start = time.time()
print('start', start)
row_names = ['movie_id', 'movie_url']
with open('movie_url.csv', 'r', newline='') as in_csv:
    reader = csv.DictReader(in_csv, fieldnames=row_names, delimiter=',')

    for row in reader:
        movie_id = row['movie_id']
        movie_url = row['movie_url']
        domain = 'http://www.imdb.com'

        with urllib.request.urlopen(movie_url) as response:
            html = response.read()
            soup = BeautifulSoup(html, 'html.parser')
            # Get url of poster image
            try:
                image_url = soup.find('div', class_='poster').a.img['src']
                extension = image_url[-4:]
                image_url = ''.join(image_url.partition('_')[0]) + extension
                filename = 'img/' + movie_id + extension
                with urllib.request.urlopen(image_url) as response:
                    with open('movie_poster.csv', 'a', newline='') as out_csv:
                        writer = csv.writer(out_csv, delimiter=',')
                        writer.writerow([movie_id, image_url])
            # Ignore cases where no poster image is present
            except AttributeError:
                pass
        print('movie_id:', movie_id)
print('time:', time.time()-start)