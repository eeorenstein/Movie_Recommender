import pandas as pd
import requests
from PIL import Image

movies = pd.read_csv('~/Movie_Recommender/movies.csv')
ratings = pd.read_csv('~/Movie_Recommender/ratings.csv')

# movies['year'] = movies.title.str[-5:-1]
# movies['title'] = movies.title.str[:-7]
genre_dummies = movies.genres.str.get_dummies(sep='|')
genre_dummies.drop('(no genres listed)', axis=1, inplace=True)
movies = pd.concat([movies, genre_dummies], axis=1).drop('genres', axis=1)

rpm_df = ratings.groupby('movieId').rating.count()
movies_rpm = movies.join(rpm_df, on='movieId').rename(columns={'rating': 'n_ratings'}).fillna(0)
movies_rpm = movies_rpm.sort_values(by='n_ratings', ascending=False)

movie_ids = list(movies_rpm.movieId)
movie_titles = list(movies_rpm.title)
movie_dict = {movie_ids[i]: movie_titles[i] for i in range(len(movie_ids))}
rating_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
genre_list = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
              'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

def get_poster(movieId):
    movie_id = int(movieId)
    key = '0bf3d77b45573e6cf57441dc7ed2fb3a'
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}&language=en-US'.format(movie_id, key))
    poster_path = response.json()['poster_path']
    im = Image.open(requests.get('https://image.tmdb.org/t/p/original{}'.format(poster_path), stream=True).raw)
    imdb_id = response.json()['imdb_id']
    url = 'https://www.imdb.com/title/{}/'.format(imdb_id)
    return im.resize((200,300)), url

