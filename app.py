import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import os

# Google Drive dosya bilgisi
FILE_ID = "13UKG6Dox3hUVg4_VZUWoQuz2pn3jOVZe"
PARQUET_FILE = "movies_imdb.parquet"

# CSV dosyasını indir ve Parquet'e dönüştür
def download_and_convert():
    if not os.path.exists(PARQUET_FILE):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        csv_file = "movies_imdb.csv"
        gdown.download(url, csv_file, quiet=False)
        
        # Büyük CSV'yi chunks ile oku ve bozuk satırları atla
        chunks = pd.read_csv(csv_file, encoding='latin1', chunksize=1_000_000, on_bad_lines='skip')
        df_csv = pd.concat(chunks, ignore_index=True)
        
        df_csv.to_parquet(PARQUET_FILE, index=False)
        os.remove(csv_file)

# Veri hazırlama ve önbelleğe alma
@st.cache_data(show_spinner=True)
def prepare_data(vote_threshold=1000, min_votes=2500):
    download_and_convert()
    df = pd.read_parquet(PARQUET_FILE)

    # Başlık ve yılı ayır
    df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")
    df["TIME"] = pd.to_datetime(df["TIME"], errors='coerce', dayfirst=True)
    df.dropna(subset=["TITLE", "YEAR", "TIME", "RATING"], inplace=True)
    df["YEAR"] = df["YEAR"].astype(int)
    df["RATING_10"] = df["RATING"] * 2

    # Oy sayısı ve ağırlıklı IMDb skoru
    vote_counts = df.groupby("TITLE")["RATING"].count()
    df["NUM_VOTES"] = df["TITLE"].map(vote_counts)
    mean_rating = df["RATING_10"].mean()

    movie_stats = df.groupby("TITLE").agg({
        "RATING_10": "mean",
        "NUM_VOTES": "max"
    }).reset_index()

    def weighted_rating(rating, votes, min_votes, mean_rating):
        denom = votes + min_votes
        if denom == 0:
            return 0
        return (votes / denom) * rating + (min_votes / denom) * mean_rating

    movie_stats["IMDB_SCORE"] = movie_stats.apply(
        lambda x: weighted_rating(x["RATING_10"], x["NUM_VOTES"], min_votes, mean_rating),
        axis=1
    )
    df["IMDB_SCORE"] = df["TITLE"].map(movie_stats.set_index("TITLE")["IMDB_SCORE"])

    # Popüler filmleri filtrele
    popular_titles = vote_counts[vote_counts >= vote_threshold].index
    df_filtered = df[df["TITLE"].isin(popular_titles)].copy()

    # Kullanıcı-film matrisi
    user_movie_matrix = df_filtered.pivot_table(
        index="USERID",
        columns="TITLE",
        values="RATING_10",
        aggfunc='mean'
    ).fillna(0)

    # Film benzerliği
    movie_similarity_df = pd.DataFrame(
        cosine_similarity(user_movie_matrix.T),
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns
    )

    # Normalized title dict
    normalized_titles_dict = {unicodedata.normalize('NFD', t).encode('ascii', 'ignore').decode('utf-8').lower().strip(): t
                              for t in movie_similarity_df.columns}

    return df, df_filtered, user_movie_matrix, movie_similarity_df, normalized_titles_dict

# Yardımcı fonksiyonlar
def normalize_title(title):
    return ''.join(c for c in unicodedata.normalize('NFD', title) if unicodedata.category(c) != 'Mn').lower().strip()

def find_best_match(input_title, normalized_titles_dict):
    normalized_input = normalize_title(input_title)
    matches = difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=1)
    return normalized_titles_dict[matches[0]] if matches else None

def suggest_alternatives(input_title, normalized_titles_dict):
    normalized_input = normalize_title(input_title)
    return [normalized_titles_dict[t] for t in difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=3)]

def recommend_by_title(title, similarity_df, top_n=5, watched=None, normalized_titles_dict=None):
    watched = watched or set()
    match = find_best_match(title, normalized_titles_dict)
    if not match:
        return suggest_alternatives(title, normalized_titles_dict)
    scores = similarity_df[match].drop(labels=watched.union({match}), errors="ignore")
    return scores.sort_values(ascending=False).head(top_n).index.tolist()

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
    year_movies = df[df['YEAR'] == year]
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

# Streamlit arayüz
st.title("🎬 Film Öneri Sistemi")
df, df_filtered, user_movie_matrix, movie_similarity_df, normalized_titles = prepare_data()

menu = ["Film Adına Göre", "Kullanıcıya Göre", "Yıla Göre", "Türe Göre"]
choice = st.sidebar.selectbox("Öneri Tipi Seçin:", menu)

if choice == "Film Adına Göre":
    movie_input = st.text_input("Film Adı Girin:")
    if movie_input:
        recommendations = recommend_by_title(movie_input, movie_similarity_df, normalized_titles_dict=normalized_titles)
        if recommendations:
            st.success("🎯 Önerilen Filmler:")
            for i, rec in enumerate(recommendations, 1):
                score = df[df["TITLE"] == rec]["IMDB_SCORE"].mean()
                st.write(f"{i}. {rec} - IMDb Skoru: {score:.2f}")
        else:
            st.warning("Film bulunamadı. Önerilen alternatifler:")
            for alt in suggest_alternatives(movie_input, normalized_titles):
                st.write(f"- {alt}")

elif choice == "Kullanıcıya Göre":
    user_ids = df["USERID"].unique()
    user_input = st.number_input("Kullanıcı ID Girin:", min_value=int(user_ids.min()), max_value=int(user_ids.max()))
    if user_input:
        recommendations = recommend_by_user(user_input, user_movie_matrix, movie_similarity_df)
        if recommendations:
            st.success("🎯 Önerilen Filmler:")
            for i, rec in enumerate(recommendations, 1):
                score = df[df["TITLE"] == rec]["IMDB_SCORE"].mean()
                st.write(f"{i}. {rec} - IMDb Skoru: {score:.2f}")
        else:
            st.warning("Kullanıcıya ait izleme geçmişi bulunamadı.")

elif choice == "Yıla Göre":
    year_input = st.number_input("Yıl Girin:", min_value=int(df['YEAR'].min()), max_value=int(df['YEAR'].max()))
    if year_input:
        top = top_movies_by_year(df_filtered, year_input)
        if top:
            st.success(f"{year_input} yılına ait en iyi filmler:")
            for i, rec in enumerate(top, 1):
                score = df[df["TITLE"] == rec]["IMDB_SCORE"].mean()
                st.write(f"{i}. {rec} - IMDb Skoru: {score:.2f}")
        else:
            st.warning("Bu yıla ait film bulunamadı.")

elif choice == "Türe Göre":
    genre_input = st.text_input("Tür Girin (örn. Comedy, Drama):")
    if genre_input:
        top = recommend_by_genre(df_filtered, genre_input)
        if top:
            st.success(f"{genre_input} türüne ait en iyi filmler:")
            for i, rec in enumerate(top, 1):
                score = df[df["TITLE"] == rec]["IMDB_SCORE"].mean()
                st.write(f"{i}. {rec} - IMDb Skoru: {score:.2f}")
        else:
            st.warning("Bu türe ait film bulunamadı.")
