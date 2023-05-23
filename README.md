# Customizable Movie Recommender

Click [here](https://eeorenstein-movie-recommender-app-q5vime.streamlit.app/) to access the app.

## Background
Recommender systems play a powerful role in our daily lives, prioritizing (and narrowing down) the options available to a user. These systems are widespread, ultimately impacting what we watch, listen to, and purchase. However, many popular movie and TV streaming platforms generate recommendations with limited input from the user. 

## Methodology
The goal of this project is to create and deploy a movie recommender web app that allows users to be more explicit with their preferences. 
The web app uses a collaborative-filtering recommender (i.e., recommendations are based on similar users’ preferences). 
Specifically, I used the surprise library’s SVD algorithm and found optimal hyperparameters using grid-search cross-validation. 

## Data
Data used to build the recommender includes: 
* 100,000 ratings from MovieLens
  * UserId, MovieId, Rating, Timestamp
  * CSV
* 9,000 movies from MovieLens
  * MovieId, Title, Genre
  * CSV
* Movie posters and URLs were sourced from TMDB’s API

## Key Insights
An analysis of the MovieLens dataset lended some valuable insights:
The distribution of ratings was right-skewed
More than one third of the 9,000 movies were rated only once
Nearly one fifth of users rated fewer than 30 movies (the minimum number of movies rated by any user is 20)

This skewness may result in a common problem of recommendation systems: feedback loops. Popular, well-liked movies are likely to be recommended to users, which results in more users watching and rating these movies, leading to them becoming even more popular. 

I then sought out to better understand prediction quality. Using a baseline recommender, I generated predicted ratings on a test set. The number of users per movie (UPM) appears to play a role in prediction quality, with poor quality ratings having lower UPM than high quality ratings. A movie’s rating variance may also affect prediction quality, signifying that more polarizing movies are harder to accurately predict. It’s difficult to tease apart these effects, but it’s clear that one, both, or a combination of the two are important factors in prediction quality. 

An example visual:

![picture alt](https://github.com/eeorenstein/Movie_Recommender/blob/main/ratings_per_movie_dist.png)

For the full EDA, see the EDA_movies notebook.

## Recommender Details
The web app ultimately helps the user take back control of their recommendations. This web app utilizes explicit ratings provided by the user, whereas streaming services’ recommendations often rely on implicit ratings, which serve as a proxy for explicit ratings but are not as granular or accurate. Additionally, the web app provides recommendation filters (e.g., based on genre), which allow for more relevant recommendations for the user.

## Tools
* Pandas
* Seaborn
* Surprise
* Joblib
* Requests
* PIL
* Streamlit

## Future Work
On the design side, I would like to add more user features, including the ability to remove a rating. Also, as it is currently constructed, user's ratings are not added to the dataset. Storing a user's ratings for future use would allow users to return to the app to add and amend ratings and the additional data could help improve the quality of recommendations.
