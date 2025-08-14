import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
from sklearn.metrics.pairwise import cosine_similarity
import gdown

st.set_page_config(page_title="KodBlessYou Movie Recommendations", layout="wide")

# ------------------- FONKSİYONLAR -------------------
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
def download_and_prepare_data(drive_url, local_path="movies_imdb.parquet", vote_threshold=1000, min_votes=2500):
    # Drive'dan parquet dosyasını indir
    gdown.download(drive_url, local_path, quiet=False)
    
    # Parquet dosyasını oku
    df = pd.read_parquet(local_path)
    
    # TITLE ve YEAR ayrıştırması
    df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")
    
    # TIME sütunu datetime
    df["TIME"] = pd.to_datetime(df["TIME"], dayfirst=True, errors='coerce')
    
    # Eksik veri temizliği
    df.dropna(subset=["TITLE", "YEAR", "TIME", "RATING"], inplace=True)
    df["YEAR"] = df["YEAR"].astype(int)
    
    # 10 üzerinden normalize edilmiş puan
    df["RATING_10"] = df["RATING"] * 2
    
    # Oy sayısı
    vote_counts = df.groupby("TITLE")["RATING"].count()
    df["NUM_VOTES"] = df["TITLE"].map(vote_counts)
    
    # Ortalama rating
    mean_rating = df["RATING_10"].mean()
    
    # Ağırlıklı IMDb skoru
    movie_stats = df.groupby("TITLE").agg({"RATING_10": "mean", "NUM_VOTES": "max"}).reset_index()
    movie_stats["IMDB_SCORE"] = movie_stats.apply(
        lambda x: weighted_rating(x["RATING_10"], x["NUM_VOTES"], min_votes, mean_rating),
        axis=1
    )
    df["IMDB_SCORE"] = df["TITLE"].map(movie_stats.set_index("TITLE")["IMDB_SCORE"])
    
    # Popüler filmler
    popular_titles = vote_counts[vote_counts >= vote_threshold].index
    df_filtered = df[df["TITLE"].isin(popular_titles)].copy()
    
    # Kullanıcı-film matrisi
    user_movie_matrix = df_filtered.pivot_table(
        index="USERID",
        columns="TITLE",
        values="RATING_10",
        aggfunc='mean'
    ).fillna(0)
    
    # Cosine similarity matrisi
    movie_similarity_df = pd.DataFrame(
        cosine_similarity(user_movie_matrix.T),
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns
    )
    
    # Normalize edilmiş başlık sözlüğü
    normalized_titles_dict = {normalize_title(t): t for t in movie_similarity_df.columns}
    
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
    try:
        year = int(year)
        year_movies = df[df['YEAR'] == year]
        if year_movies.empty:
            return []
        top = year_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(top_n)
        return top.index.tolist()
    except ValueError:
        return []

def recommend_by_genre(df, genre, top_n=5):
    genre = genre.strip().title()
    genre_movies = df[df["GENRES"].str.contains(genre, case=False, na=False)]
    if genre_movies.empty:
        return []
    top = genre_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(top_n)
    return top.index.tolist()

# ------------------- STREAMLIT APP -------------------
st.title("🎬 KodBlessYou Movie Recommendations")

# Drive URL'nizi buraya ekleyin (gdown uyumlu)
drive_url = "https://drive.google.com/uc?id=13UKG6Dox3hUVg4_VZUWoQuz2pn3jOVZe"

df, df_filtered, user_movie_matrix, similarity_df, norm_titles = download_and_prepare_data(drive_url)
watched_movies = set()

option = st.radio(
    "Choose recommendation type:",
    ("By Movie Title", "By User History", "Top Movies by Year", "By Genre")
)

if option == "By Movie Title":
    movie_input = st.text_input("Enter a movie title you like:")
    if movie_input:
        recommendations, alternatives = recommend_by_title(movie_input, similarity_df, top_n=5, watched=watched_movies, normalized_titles_dict=norm_titles)
        if recommendations:
            st.success(f"Recommendations based on '{movie_input}':")
            for i, rec in enumerate(recommendations, 1):
                score = df[df["TITLE"] == rec]["IMDB_SCORE"].mean()
                st.write(f"{i}. {rec} - IMDb Score: {score:.2f}")
                watched_movies.add(rec)
        elif alternatives:
            st.warning("Movie not found. Did you mean:")
            for alt in alternatives:
                st.write(f"- {alt}")
        else:
            st.info("No recommendations found.")

elif option == "By User History":
    top_users = df["USERID"].value_counts().head(10).index.tolist()
    user_input = st.selectbox("Select a User ID:", top_users)
    recommendations = recommend_by_user(user_input, user_movie_matrix, similarity_df)
    if recommendations:
        st.success(f"Recommendations for User {user_input}:")
        for i, rec in enumerate(recommendations, 1):
            score = df[df["TITLE"] == rec]["IMDB_SCORE"].mean()
            st.write(f"{i}. {rec} - IMDb Score: {score:.2f}")
    else:
        st.info("No recommendations found for this user.")

elif option == "Top Movies by Year":
    year_input = st.text_input("Enter a year (e.g., 2015):")
    if year_input:
        top_movies = top_movies_by_year(df_filtered, year_input)
        if top_movies:
            st.success(f"Top movies for {year_input}:")
            for i, title in enumerate(top_movies, 1):
                score = df_filtered[df_filtered["TITLE"] == title]["IMDB_SCORE"].mean()
                st.write(f"{i}. {title} - IMDb Score: {score:.2f}")
        else:
            st.info("No movies found for this year.")

elif option == "By Genre":
    genre_input = st.text_input("Enter a genre (e.g., Comedy):")
    if genre_input:
        top_genre_movies = recommend_by_genre(df_filtered, genre_input)
        if top_genre_movies:
            st.success(f"Top movies in genre '{genre_input}':")
            for i, title in enumerate(top_genre_movies, 1):
                score = df_filtered[df_filtered["TITLE"] == title]["IMDB_SCORE"].mean()
                st.write(f"{i}. {title} - IMDb Score: {score:.2f}")
        else:
            st.info("No movies found for this genre.")
