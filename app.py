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

# Streamlit sayfa ayarları
st.set_page_config(page_title="🎬 Film Öneri Sistemi", layout="wide")

# Drive dosya bilgileri
FILE_ID = "13UKG6Dox3hUVg4_VZUWoQuz2pn3jOVZe"
FILE_NAME = "movies_imdb.parquet"

# Unicode normalize fonksiyonu
def normalize_title(title):
    return ''.join(
        c for c in unicodedata.normalize('NFD', title)
        if unicodedata.category(c) != 'Mn'
    ).lower().strip()

# IMDb ağırlıklı puan hesaplama
def weighted_rating(rating, votes, min_votes, mean_rating):
    denominator = votes + min_votes
    if denominator == 0:
        return 0
    return (votes / denominator) * rating + (min_votes / denominator) * mean_rating

# Veriyi indir ve Parquet olarak kaydet
def download_data():
    if not os.path.exists(FILE_NAME):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        csv_file = "movies_imdb.csv"
        gdown.download(url, csv_file, quiet=False)
        df_csv = pd.read_csv(csv_file)
        df_csv.to_parquet(FILE_NAME, index=False)
        os.remove(csv_file)

# Veri hazırlama
@st.cache_data(show_spinner=True)
def prepare_data(vote_threshold=1000, min_votes=2500):
    download_data()
    df = pd.read_parquet(FILE_NAME)

    df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")
    df["TIME"] = pd.to_datetime(df["TIME"], dayfirst=True, errors='coerce')
    df.dropna(subset=["TITLE", "YEAR", "TIME", "RATING"], inplace=True)
    df["YEAR"] = df["YEAR"].astype(int)
    df["RATING_10"] = df["RATING"] * 2
    vote_counts = df.groupby("TITLE")["RATING"].count()
    df["NUM_VOTES"] = df["TITLE"].map(vote_counts)
    mean_rating = df["RATING_10"].mean()
    movie_stats = df.groupby("TITLE").agg({
        "RATING_10": "mean",
        "NUM_VOTES": "max"
    }).reset_index()
    movie_stats["IMDB_SCORE"] = movie_stats.apply(
        lambda x: weighted_rating(x["RATING_10"], x["NUM_VOTES"], min_votes, mean_rating),
        axis=1
    )
    df["IMDB_SCORE"] = df["TITLE"].map(movie_stats.set_index("TITLE")["IMDB_SCORE"])
    popular_titles = vote_counts[vote_counts >= vote_threshold].index
    df_filtered = df[df["TITLE"].isin(popular_titles)].copy()

    if "USERID" not in df_filtered.columns:
        st.error("Veri setinde USERID sütunu bulunamadı.")
        return df, df_filtered, pd.DataFrame(), pd.DataFrame(), {}

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
    return df, df_filtered, user_movie_matrix, movie_similarity_df, normalized_titles_dict

# Film adına göre öneri
def recommend_by_title(title, similarity_df, normalized_titles_dict, top_n=5, watched=None):
    watched = watched or set()
    normalized = normalize_title(title)
    if normalized not in normalized_titles_dict:
        close_matches = difflib.get_close_matches(normalized, normalized_titles_dict.keys(), n=3)
        return [normalized_titles_dict[m] for m in close_matches] if close_matches else []
    real_title = normalized_titles_dict[normalized]
    scores = similarity_df[real_title].drop(labels=watched.union({real_title}), errors='ignore')
    return scores.sort_values(ascending=False).head(top_n).index.tolist()

# Kullanıcıya göre öneri
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

# Yıla göre en iyi filmler
def top_movies_by_year(df, year, top_n=5):
    year_movies = df[df['YEAR'] == year]
    if year_movies.empty:
        return []
    top = year_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(top_n)
    return top.index.tolist()

# Türüne göre öneri
def recommend_by_genre(df, genre, top_n=5):
    genre_movies = df[df["GENRES"].str.contains(genre, case=False, na=False)]
    if genre_movies.empty:
        return []
    top = genre_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(top_n)
    return top.index.tolist()

# Streamlit arayüz
st.title("🎬 Film Öneri Sistemi")

df, df_filtered, user_movie_matrix, movie_similarity_df, normalized_titles_dict = prepare_data()

menu = ["Film Adına Göre", "Kullanıcıya Göre", "Yıla Göre", "Türe Göre"]
choice = st.sidebar.selectbox("Öneri Tipi Seçin:", menu)

if choice == "Film Adına Göre":
    movie_input = st.text_input("Film Adı Giriniz:")
    if movie_input:
        recs = recommend_by_title(movie_input, movie_similarity_df, normalized_titles_dict)
        if recs:
            st.subheader("Önerilen Filmler:")
            for i, r in enumerate(recs, 1):
                score = df[df["TITLE"] == r]["IMDB_SCORE"].mean()
                st.write(f"{i}. {r} - IMDb Skoru: {score:.2f}")
        else:
            st.warning("Film bulunamadı veya alternatif öneriler yok.")

elif choice == "Kullanıcıya Göre":
    top_users = df["USERID"].value_counts().head(10).index.tolist()
    user_input = st.selectbox("Kullanıcı Seçin:", top_users)
    if user_input:
        recs = recommend_by_user(user_input, user_movie_matrix, movie_similarity_df)
        if recs:
            st.subheader("Önerilen Filmler:")
            for i, r in enumerate(recs, 1):
                score = df[df["TITLE"] == r]["IMDB_SCORE"].mean()
                st.write(f"{i}. {r} - IMDb Skoru: {score:.2f}")
        else:
            st.warning("Bu kullanıcıya göre öneri bulunamadı.")

elif choice == "Yıla Göre":
    year_input = st.number_input("Yıl Giriniz:", min_value=int(df['YEAR'].min()), max_value=int(df['YEAR'].max()), step=1)
    if year_input:
        recs = top_movies_by_year(df_filtered, year_input)
        if recs:
            st.subheader(f"{year_input} Yılının En İyi Filmleri:")
            for i, r in enumerate(recs, 1):
                score = df[df["TITLE"] == r]["IMDB_SCORE"].mean()
                st.write(f"{i}. {r} - IMDb Skoru: {score:.2f}")
        else:
            st.warning("Bu yıla ait film bulunamadı.")

elif choice == "Türe Göre":
    genre_input = st.text_input("Tür Giriniz (Örn: Action, Comedy, Drama):")
    if genre_input:
        recs = recommend_by_genre(df_filtered, genre_input)
        if recs:
            st.subheader(f"'{genre_input}' Türündeki En İyi Filmler:")
            for i, r in enumerate(recs, 1):
                score = df[df["TITLE"] == r]["IMDB_SCORE"].mean()
                st.write(f"{i}. {r} - IMDb Skoru: {score:.2f}")
        else:
            st.warning("Bu türe ait film bulunamadı.")
