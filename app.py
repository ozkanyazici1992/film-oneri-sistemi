import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
import logging
from sklearn.metrics.pairwise import cosine_similarity
from colorama import init, Fore, Style
import gdown
import os

# Colorama ayarı
init(autoreset=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Streamlit ayarları
st.set_page_config(page_title="Film Öneri Sistemi", layout="wide")

# Dosya bilgileri
FILE_ID = "13UKG6Dox3hUVg4_VZUWoQuz2pn3jOVZe"
FILE_NAME = "movies_imdb.parquet"

# Normalize fonksiyonu
def normalize_title(title):
    return unicodedata.normalize("NFKD", title.lower())

# IMDb ağırlıklı puan hesaplama
def weighted_rating(R, v, M, C):
    return (v/(v+M) * R) + (M/(M+v) * C)

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
def prepare_data(vote_threshold=1000, M=5000):
    download_data()
    df = pd.read_parquet(FILE_NAME)

    df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")
    df["TIME"] = pd.to_datetime(df["TIME"], dayfirst=True, errors='coerce')
    df.dropna(subset=["TITLE", "YEAR", "TIME", "RATING"], inplace=True)
    df["YEAR"] = df["YEAR"].astype(int)
    df["RATING_10"] = df["RATING"] * 2

    vote_counts = df.groupby("TITLE")["RATING"].count()
    df["NUM_VOTES"] = df["TITLE"].map(vote_counts).astype('int32')
    C = df["RATING_10"].mean()

    movie_stats = df.groupby("TITLE").agg({
        "RATING_10": "mean",
        "NUM_VOTES": "max"
    }).reset_index()

    movie_stats["IMDB_SCORE"] = movie_stats.apply(
        lambda x: weighted_rating(x["RATING_10"], x["NUM_VOTES"], M, C), axis=1
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
def recommend_by_title(title, similarity_df, top_n=5):
    title_norm = normalize_title(title)
    if title_norm not in normalized_titles_dict:
        st.warning(f"Film '{title}' bulunamadı.")
        return []
    real_title = normalized_titles_dict[title_norm]
    sim_scores = similarity_df[real_title].sort_values(ascending=False)
    sim_scores = sim_scores.drop(real_title)
    return sim_scores.head(top_n).index.tolist()

# Streamlit arayüzü
st.title("🎬 Film Öneri Sistemi")

df, df_filtered, user_movie_matrix, movie_similarity_df, normalized_titles_dict = prepare_data()

movie_input = st.text_input("Film Adı Giriniz:")
if movie_input:
    recommendations = recommend_by_title(movie_input, movie_similarity_df)
    if recommendations:
        st.subheader("Önerilen Filmler:")
        for r in recommendations:
            st.write(r)

