import ast
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity as similarity

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break

    return L

def convert_cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


def get_movies_by_actor(movies, actor_name):
    # create boolean masks to filter movies that the actor appears in each of the three cast columns
    mask1 = movies['cast1'].apply(lambda x: pd.notna(x) and actor_name in x)
    mask2 = movies['cast2'].apply(lambda x: pd.notna(x) and actor_name in x)
    mask3 = movies['cast3'].apply(lambda x: pd.notna(x) and actor_name in x)

    # combine the three masks using logical OR to get a single mask for all movies featuring the actor
    mask = mask1 | mask2 | mask3

    # subset the movies dataframe to include only the desired columns
    subset = movies.loc[mask, ['title', 'cast', 'Director', 'genres']]

    return subset






def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


def stem(text):
    y = []

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

def weighted_rating(df, ):
    c = np.mean(df['vote_average'])

    m = np.quantile(df['vote_count'],0.1)
    v = df['vote_count']
    R = df['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * c)

def rating (df,rating):
        a = df[df['ratings'] == rating][['title', 'genres', 'cast', 'Director', 'ratings']].head(10)
        return a


def movies_director(mov, director_name):
    # Filter the DataFrame to include only movies with the specified director
    movies_by_director = mov[mov['Director'].apply(lambda x: director_name in x)]

    # Return a new DataFrame with the desired columns
    return movies_by_director[['title', 'genres', 'cast', 'Director']]


def recommended_Director_movies(mov,director_name,similarity):
    # Check if any movies are available for the specified director
    if mov[mov['Director'].apply(lambda x: director_name in x)].empty:
        return "No movies found for director " + director_name

    movie_index = mov[mov['Director'].apply(lambda x: director_name in x)].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
    recommended_movies = mov.iloc[[i[0] for i in movies_list]][['title', 'genres', 'cast', 'Director']]

    return recommended_movies


def movies_by_genres2(movies,genres_list):
    # Filter the DataFrame to include only movies with the specified genres
    movies_by_genres = movies[movies['genres'].apply(lambda x: all(g in x for g in genres_list))]

    # Check if any movies match the specified genres or not
    if movies_by_genres.empty:
        print('No movies found for the specified genres.')
        return None

    # Return a new DataFrame with the desired columns
    return movies_by_genres[['title', 'genres', 'cast', 'Director']]


def recommended_genres_movies(genre_list, movies, similarity):
    # Check if any movies are available for the specified director
    movies_by_genres = movies[movies['genres'].apply(lambda x: all(g in x for g in genre_list))]
    if len(movies_by_genres) == 0:
        print("No movies found that match the specified genres.")
        return None

    genre_index = movies_by_genres.index

    #movie_index = movies[movies['genres'].apply(lambda x: all(g in x for g in genre_list))].index[0]
    filtered_movies = movies_by_genres
    if len(filtered_movies) > 0:
        movie_index = filtered_movies.index[0]
    else:
        print("No movies found that match the specified genres.")

    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
    genrewise_movies = movies.iloc[[i[0] for i in movies_list]][['title', 'genres', 'cast', 'Director']]

    return genrewise_movies[['title', 'genres', 'cast', 'Director']]

