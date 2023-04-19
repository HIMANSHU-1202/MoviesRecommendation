import streamlit as st
import pandas as pd
import numpy as np
import helper
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text  import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies= pd.read_csv('tmdb_5000_movies.csv')
credits= pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits,on='title')

movies=movies[['movie_id','title','overview','genres','keywords','cast', 'crew','vote_count', 'vote_average']]

movies.dropna(inplace=True)



#Required Functions:
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity



def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


def recommend(movie):
    movie_index= new_df[new_df['title']==movie].index[0]
    distances= similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:10]
    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(new_df.iloc[i[0]].title)
    return recommended_movies

#Columns Normalisation :
#1) Genres column:
movies['genres']=movies['genres'].apply(helper.convert)
#2) Keywords column:
movies['keywords']=movies['keywords'].apply(helper.convert)
#3) Cast columns:
# Apply the function to the 'cast' column to create three new columns
movies[['cast1', 'cast2', 'cast3']] = movies['cast'].apply(helper.convert_cast).apply(pd.Series)

movies['cast']=movies['cast'].apply(helper.convert3)
#4) Crew column:
movies['crew']=movies['crew'].apply(helper.fetch_director)
#5) overview column:
movies['overview']= movies['overview'].apply(lambda x:x.split())

mov = movies.copy()

#Removing Space between names
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies=movies.rename(columns={'crew': 'Director'})
#Creating Tag column:

movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['Director']

new_df=movies[['movie_id','title','tags']]

new_df['tags']= new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
ps=PorterStemmer()

new_df['tags']=new_df['tags'].apply(stem)

#Text Vectorisation:

cv = CountVectorizer(max_features=5000,stop_words= 'english')

vectors= cv.fit_transform(new_df['tags']).toarray()

similarity=cosine_similarity(vectors)




st.header('Movies Recommendation')

st.sidebar.title("Movies Recommendation")

user_menu = st.sidebar.radio(
    'Select an Option',
    ('Movies','Actor','Director','Genres','Ratings')
)

if user_menu == 'Movies':
#Recommedndation by Movies Names:
    st.sidebar.header('Recommendation on Names')


    movies_list=new_df['title'].unique().tolist()
    movies_list.sort()
    selected_movie=st.sidebar.selectbox("Select Movies",movies_list)
    st.header('Recommendations for ' +selected_movie)
    x=recommend(selected_movie)

    st.dataframe(x)



if user_menu == 'Ratings':

#Ratingswise Movies

    st.header('Ratingswise Movies')
    m = np.quantile(movies['vote_count'],0.1)
    q_movies = movies.copy().loc[movies['vote_count'] >= m]
    q_movies['ratings'] = q_movies.apply(helper.weighted_rating, axis=1).astype('int')
    q_movies = q_movies.sort_values('ratings', ascending=False)
    q_movies[['title', 'vote_count', 'vote_average', 'ratings']].head(10)
    selecte_ratings=q_movies['ratings'].unique()
    selected_rating=st.selectbox('Select Rating',selecte_ratings)
    rated_movies=helper.rating(q_movies,selected_rating)
    st.dataframe(rated_movies)


if user_menu=='Actor':
#ActorsWise Movies
    cast_columns = ['cast1', 'cast2', 'cast3']
    actors = set(str(actor) for col in cast_columns for actor in movies[col].fillna(''))
    sorted_actors = sorted(actors)

    actor_name=st.selectbox('**Select Actor**',sorted_actors)
    st.subheader(actor_name +' Movies')
    movie_subset = helper.get_movies_by_actor(movies,actor_name)
    st.dataframe(movie_subset)

if user_menu== 'Director':
#Directorwise Recommendation :

    st.subheader(' Directorwise Recommendation ')

    mov = mov.rename(columns={'crew': 'Director'})
    # Extract the 'Director' column as a list of lists
    directors = mov['Director'].tolist()

    # Flatten the list of lists into a single list and convert to set to remove duplicates
    unique_di= set(director for sublist in directors for director in sublist)

    # Sort the set of unique directors in alphabetical order and convert back to list
    sorted_dir = sorted(unique_di)


    Dire = st.selectbox('Select Director',sorted_dir,key='selectbox1')
    recommended_mov=helper.recommended_Director_movies(mov,Dire,similarity)
    st.dataframe(recommended_mov)


#Directors Movies:

    st.subheader('Directors Movies')

    mov = mov.rename(columns={'crew': 'Director'})
    # Extract the 'Director' column as a list of lists
    director = mov['Director'].tolist()

    # Flatten the list of lists into a single list and convert to set to remove duplicates
    unique_directors = set(director for sublist in directors for director in sublist)

    # Sort the set of unique directors in alphabetical order and convert back to list
    sorted_directors = sorted(unique_directors)

    selected_Director=st.selectbox("Select Director",sorted_directors,key='selectbox2')
    x= helper.movies_director(mov,selected_Director)
    st.dataframe(x)

if user_menu=='Genres':


#Movies Based on Genres:
    st.header('Movies Based on Genres')
    # Create an empty list to store all genres

    all_genres = []

    # Iterate over each row in the 'genres' column
    for row in movies['genres']:
        # Iterate over each genre in the row
        for genre in row:
            # If the genre is not already in the all_genres list, add it
            if genre not in all_genres:
                all_genres.append(genre)

    selected_genres=st.multiselect('Select Genres',all_genres,key='multiselect1')
    movies_gen =helper.movies_by_genres2(movies,selected_genres)
    st.dataframe(movies_gen)

# Genrewise Recommendation :

    st.header('Recommendation Based on Genres')


    sel_genres = st.multiselect('Select Genres', all_genres , key='multiselect2')
    genrewise_recommendation = helper.recommended_genres_movies(sel_genres, movies, similarity)
    st.dataframe(genrewise_recommendation)