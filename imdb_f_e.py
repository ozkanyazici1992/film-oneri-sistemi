import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
import logging
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import os

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.2f}'.format

# Google Drive dosya bilgisi
DATA_URL = "https://drive.google.com/uc?id=1gl_iJXRyEaSzhHlgfBUdTzQZMer4gdsS"
LOCAL_FILE = "movies_imdb_2.csv"

# --- Fonksiyonlar ---
def weighted_rating(rating, votes, min_votes, mean_rating):
    denominator = votes + min_votes
    if denominator == 0:
        return 0
    return (votes / denominator) * rating + (min_votes / denominator) * mean_rating

def normalize_title(title):
    return ''.join(
        c for c in unicodedata.normalize('NFD', title)
        if unicodedata.category(c) != 'Mn'
    ).lower().strip()

@st.cache_data(show_spinner=True)
def download_and_prepare_data(file_url=DATA_URL, local_path=LOCAL_FILE, vote_threshold=1000, min_votes=2500):
    # Dosya yoksa indir
    if not os.path.exists(local_path):
        gdown.download(file_url, local_path, quiet=False)
    
    df = pd.read_csv(local_path)
    df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")
    df["TIME"] = pd.to_datetime(df["TIME"], dayfirst=True, errors='coerce')
    df.dropna(subset=["TITLE", "YEAR", "TIME", "RATING"], inplace=True)
    df["YEAR"] = df["YEAR"].astype(int)
    df["RATING_10"] = df["RATING"] * 2
    vote_counts = df.groupby("TITLE")["RATING"].count()
    df["NUM_VOTES"] = df["TITLE"].map(vote_counts)
    mean_rating = df["RATING_10"].mean()
    movie_stats = df.groupby("TITLE").agg({"RATING_10": "mean", "NUM_VOTES": "max"}).reset_index()
    movie_stats["IMDB_SCORE"] = movie_stats.apply(
        lambda x: weighted_rating(x["RATING_10"], x["NUM_VOTES"], min_votes, mean_rating), axis=1
    )
    df["IMDB_SCORE"] = df["TITLE"].map(movie_stats.set_index("TITLE")["IMDB_SCORE"])
    popular_titles = vote_counts[vote_counts >= vote_threshold].index
    df_filtered = df[df["TITLE"].isin(popular_titles)].copy()
    user_movie_matrix = df_filtered.pivot_table(
        index="USERID",
        columns="TITLE",
        values="RATING_10",
        aggfunc='mean'
    ).fillna(0)
    movie_similarity_df = pd.DataFrame(
        cosine_similarity(user_movie_matrix.T),
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns
    )
    normalized_titles_dict = {normalize_title(t): t for t in movie_similarity_df.columns}
    logging.info("Data preparation completed successfully.")
    return df, df_filtered, user_movie_matrix, movie_similarity_df, normalized_titles_dict

def find_best_match(input_title, normalized_titles_dict):
    normalized_input = normalize_title(input_title)
    close_matches = difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=1)
    return normalized_titles_dict[close_matches[0]] if close_matches else None

def suggest_alternatives(input_title, normalized_titles_dict):
    normalized_input = normalize_title(input_title)
    return [normalized_titles_dict[t] for t in difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=3)]

def recommend_by_title(title, similarity_df, top_n=5, watched=None, normalized_titles_dict=None):
    watched = watched or set()
    match = find_best_match(title, normalized_titles_dict)
    if not match:
        return [], suggest_alternatives(title, normalized_titles_dict)
    scores = similarity_df[match].drop(labels=watched.union({match}), errors="ignore")
    return scores.sort_values(ascending=False).head(top_n).index.tolist(), []

def recommend_by_user(user_id, user_matrix, similarity_df, top_n=5):
    if user_id not in user_matrix.index:
        return []
    user_ratings = user_matrix.loc[user_id]
    watched = user_ratings[user_ratings > 0]
    if watched.empty:
        return []
    scores = similarity_df[watched.index].dot(watched)
    scores = scores.drop(watched.index, errors='ignore')
    return scores.sort_values(ascending=False).head(top_n).index.tolist()

def top_movies_by_year(df, year, top_n=5):
    year_movies = df[df['YEAR'] == int(year)]
    if year_movies.empty:
        return []
    top = year_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(top_n)
    return top.index.tolist()

def recommend_by_genre(df, genre, top_n=5):
    genre_movies = df[df["GENRES"].str.contains(genre, case=False, na=False)]
    if genre_movies.empty:
        return []
    top = genre_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(top_n)
    return top.index.tolist()

# --- Streamlit UI ---
st.set_page_config(page_title="KodBlessYou Movie Recommendations", layout="wide")
st.title("🎬 KodBlessYou Movie Recommendation System")

df, df_filtered, user_movie_matrix, similarity_df, norm_titles = download_and_prepare_data()

tabs = st.tabs(["By Title", "By User History", "Top Movies by Year", "By Genre"])

# --- Tab 1: By Title ---
with tabs[0]:
    st.header("🎬 Movie Recommendations by Title")
    movie_input = st.text_input("Enter a movie title:")
    if st.button("Get Title Recommendations", key="title_btn"):
        recommendations, alternatives = recommend_by_title(movie_input, similarity_df, normalized_titles_dict=norm_titles)
        if recommendations:
            st.success("Recommended Movies:")
            for rec in recommendations:
                score = df[df["TITLE"] == rec]["IMDB_SCORE"].mean()
                st.write(f"{rec} - IMDb Score: {score:.2f}")
        elif alternatives:
            st.warning("Movie not found. Did you mean:")
            for alt in alternatives:
                st.write(f"- {alt}")
        else:
            st.error("No recommendations found.")

# --- Tab 2: By User History ---
with tabs[1]:
    st.header("🧑‍💻 Recommendations by User History")
    user_id_input = st.number_input("Enter User ID:", min_value=1, step=1)
    if st.button("Get User Recommendations", key="user_btn"):
        recommendations = recommend_by_user(user_id_input, user_movie_matrix, similarity_df)
        if recommendations:
            st.success("Recommended Movies:")
            for rec in recommendations:
                score = df[df["TITLE"] == rec]["IMDB_SCORE"].mean()
                st.write(f"{rec} - IMDb Score: {score:.2f}")
        else:
            st.warning("No recommendations found or user not in dataset.")

# --- Tab 3: Top Movies by Year ---
with tabs[2]:
    st.header("📅 Top Movies by Year")
    year_input = st.number_input("Enter a year:", min_value=1900, max_value=2050, step=1)
    if st.button("Get Top Movies by Year", key="year_btn"):
        top_movies = top_movies_by_year(df_filtered, year_input)
        if top_movies:
            st.success(f"Top movies for {year_input}:")
            for rec in top_movies:
                score = df[df["TITLE"] == rec]["IMDB_SCORE"].mean()
                st.write(f"{rec} - IMDb Score: {score:.2f}")
        else:
            st.warning("No movies found for this year.")

# --- Tab 4: By Genre ---
with tabs[3]:
    st.header("🎭 Recommendations by Genre")
    genre_input = st.text_input("Enter a genre (e.g., Comedy, Action):")
    if st.button("Get Genre Recommendations", key="genre_btn"):
        top_genre_movies = recommend_by_genre(df_filtered, genre_input)
        if top_genre_movies:
            st.success(f"Top movies in {genre_input}:")
            for rec in top_genre_movies:
                score = df[df["TITLE"] == rec]["IMDB_SCORE"].mean()
                st.write(f"{rec} - IMDb Score: {score:.2f}")
        else:
            st.warning("No movies found for this genre.")
