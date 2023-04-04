import pandas as pd
import streamlit as st
import joblib
from surprise import accuracy, Dataset, SVD, KNNBasic, Reader
from surprise.model_selection import train_test_split, GridSearchCV

from utils import movie_ids, movie_titles, movie_dict, rating_list, movies_rpm, genre_list, get_poster
import time

#Read in data and model
ratings = pd.read_csv('ratings.csv')
links = pd.read_csv('links.csv')

#Initializing session state
if 'movie_counter' not in st.session_state:
    st.session_state['movie_counter'] = 0

if 'seen_movies' not in st.session_state:
    st.session_state['seen_movies'] = []

if 'unseen_movies' not in st.session_state:
    st.session_state['unseen_movies'] = movie_titles

if 'add_trainset' not in st.session_state:
    st.session_state['add_trainset'] = []

if 'user_history' not in st.session_state:
    st.session_state['user_history'] = []

if 'genre_recs' not in st.session_state:
    st.session_state['genre_recs'] = []

if 'remaining_genres' not in st.session_state:
    st.session_state['remaining_genres'] = genre_list

#Streamlit app design
tab0, tab1, tab2 = st.tabs(['Welcome!', 'Add Movie Ratings', 'Find Recommended Movies'])

with tab0:
    st.title('Movie Recommender')
    st.write('Welcome, movie watcher! Rate movies you have seen to generate personalized movie recommendations based on your (and similar movie watchers\') preferences.')
    with st.expander("Click here for instructions"):
        st.info(''' 1) *Use the \"Add Movie Ratings\" tab to add movie ratings to your profile*
                    \n First, find a movie using the dropdown menu (you can also search by typing into the bar) and give it a rating using the sliding scale (5 is the best). 
                    When you are ready to add the movie rating, click the \"Add rating\" button. 
                    Then, click the \"New rating\" button to refresh the Movie Title list. 
                    Repeat until you have added all your ratings.
                    You can rate any number of movies, but we recommend at least 10 (the more you rate, the better we can learn your preferences).
                    
                    \n 2) *Use the \"Find Recommended Movies\" tab to customize your movie recommendations*
                    \n If desired, filter your movie recommendations based on genre(s) and popularity. Then, click the \"Find my recommended movies\" button at the bottom of the page for your movie recs.
                    \n When you're done, you can refresh the page to start over. Happy movie watching!''')
with tab1:
    st.markdown('### Add rated movies to your user profile')

    #Input movies seen
    movie = st.selectbox('Movie Title', options=st.session_state['unseen_movies'])
    movieId = [i for i in movie_dict.keys() if movie_dict[i]==movie][0]
    rating = st.slider('Movie Rating', min_value=0.5, max_value=5.0, value=3.0, step=0.5)

    #Preparing train set
    session_userId = ratings.userId.max() + 1

    col1_t1, col2_t1 = st.columns((1,1))
    with col1_t1:
        add_rating_button = st.button("Add rating", use_container_width=True)
    with col2_t1:
        new_rating_button = st.button("New rating", use_container_width=True)

    if add_rating_button:
        st.session_state['movie_counter'] += 1
        st.session_state['last_movie'] = movie
        st.session_state['add_trainset'].append((session_userId, movieId, rating, time.time()))
        st.session_state['user_history'].append((movie, rating))
        st.session_state['seen_movies'].append(movie)
        st.session_state['unseen_movies'] = [i for i in movie_titles if i not in st.session_state['seen_movies']]
        st.success('You have successfully added {} ({} :star: out of 5.0) to your rated movies'.format(movie, rating))

    if new_rating_button:
        movie = st.empty()
        rating = st.empty()
   
    st.markdown('##')

    with st.expander('Click here for a table of your rated movies'):
        if st.session_state['movie_counter'] == 0:
            st.write('No movies have been rated')
        else:
            users_ratings_df = pd.DataFrame(st.session_state['user_history'], columns=['Title','Rating'])
            st.dataframe(users_ratings_df)

#Movie rec filters
with tab2:
    st.markdown('### Customize your movie recs using these filters')
    st.markdown('Note: This step is optional. Skipping these filters will include all movie types.')

    col1_t2, col2_t2 = st.columns((1,1))

    with col1_t2:
        st.write('_Select genre(s) of movie recs_')
        genre = st.selectbox('Genre', options=st.session_state['remaining_genres'])

        col3_t2, col4_t2 = st.columns((1,1))
        with col3_t2:
            add_genre_button = st.button("Add genre", use_container_width=True)

        with col4_t2:
            new_genre_button = st.button("New genre", use_container_width=True)          

        if add_genre_button:
            st.session_state['genre_recs'].append(genre)
            st.session_state['remaining_genres'] = [i for i in genre_list if i not in st.session_state['genre_recs']]
        
        if new_genre_button:
            genre = st.empty()

        if len(st.session_state['genre_recs']) > 0:
            for i in range(len(st.session_state['genre_recs'])):
                st.write(i+1, st.session_state['genre_recs'][i])
        
    with col2_t2:
        st.write('_Select popularity of movie recs_')
        popularity = st.selectbox('Movie Popularity', options = ['All', 'Mainstream', 'Sleepers'])

#Collaborative filtering recommender
    def get_preds(genres, popularity):
        prog_bar = st.empty()
        prog_bar.progress(0, text='Starting...')
        time.sleep(2)
        
        prog_bar.progress(2, text='Processing...')

        #Add users' ratings to the train set
        add_trainset = pd.DataFrame(st.session_state['add_trainset'], columns = ratings.columns)
        add_trainset = add_trainset.astype({'userId': 'int64', 'movieId': 'int64', 'rating': 'float64'})
        ratings_updated = pd.concat([ratings, add_trainset], ignore_index=True)

        #Find all movies not rated by the user
        unseen_movieIds = [i for i in movie_dict.keys() if movie_dict[i] in st.session_state['unseen_movies']]
        unseen_movies_df = movies_rpm[movies_rpm.movieId.isin(unseen_movieIds)]

        #Include unseen movies that meet the rec filters
        filtered_unseen_movies_df = unseen_movies_df[unseen_movies_df[genres].sum(axis=1) > 0]
        if popularity == 'Mainstream':
            filtered_unseen_movies_df = filtered_unseen_movies_df[filtered_unseen_movies_df.n_ratings >= 50]
        elif popularity == 'Lesser Known':
            filtered_unseen_movies_df = filtered_unseen_movies_df[filtered_unseen_movies_df.n_ratings <= 50]

        filtered_unseen_movieIds = list(filtered_unseen_movies_df.movieId)

        #Generate test set
        user_test_set = [(session_userId, i, None) for i in filtered_unseen_movieIds]
        prog_bar.progress(33, text='Learning your preferences...')

        time.sleep(1)
        prog_bar.progress(34, text='Learning your preferences...')

        #Train SVD recommender on full dataset using hyperparameters found through grid search CV
        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(ratings_updated[["userId", "movieId", "rating"]], reader=reader)
        train_set = data.build_full_trainset()
        model = joblib.load('SVD.joblib')
        model.fit(train_set)
        prog_bar.progress(66, text='Generating your recommended movies...')

        time.sleep(1)
        prog_bar.progress(67, text='Generating your recommended movies...')

        #Make predictions on test set
        preds = model.test(user_test_set)
        preds_df = pd.DataFrame(preds).astype({'uid': 'int64', 'iid': 'int64'})
        preds_df.drop('details', axis=1, inplace=True)

        #Generate users' top recommended movies
        n_preds = 10
        best_preds_df = preds_df.sort_values(by=['est'], ascending=False).iloc[:n_preds]
        best_preds_df = best_preds_df.join(links.set_index('movieId'), on='iid').reset_index()
        prog_bar.progress(99, text='Almost there...')

        time.sleep(1)
        prog_bar.progress(100, text="All done!")
        time.sleep(2)
        prog_bar.empty()
        st.markdown('## Your top {} movie recommendations:'.format(n_preds))
        return best_preds_df

    st.markdown('##')
    recs_button = st.button("Find my recommended movies", use_container_width=True)
    if recs_button:
        if st.session_state['genre_recs'] == []:
            best_preds_df = get_preds(st.session_state['remaining_genres'], popularity)
        else:
            best_preds_df = get_preds(st.session_state['genre_recs'], popularity)
        for i in range(best_preds_df.shape[0]):
            movie_key = best_preds_df.iid.iloc[i]
            im, url = get_poster(best_preds_df.tmdbId.iloc[i])
            st.write('[{}. {}](%s)'.format(i+1, movie_dict[movie_key]) % url)
            st.image(im)
        st.write('Source: Posters from TMDB')
