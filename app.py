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

# Ayarlar
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.2f}'.format

FILE_ID = "101KY0_Smh3P_Li7Gfz_zPlduH7mfnMFY"
FILE_NAME = "movies_imdb_2.csv"


def download_data():
    if not os.path.exists(FILE_NAME):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, FILE_NAME, quiet=False)
    else:
        logging.info(f"{FILE_NAME} zaten mevcut, indirme atlandı.")


def weighted_rating(r, v, M, C):
    denom = v + M
    if denom == 0:
        return 0
    return (v / denom) * r + (M / denom) * C


def normalize_title(title):
    return ''.join(c for c in unicodedata.normalize('NFD', title) if unicodedata.category(c) != 'Mn').lower().strip()


@st.cache_data(show_spinner=True)
def prepare_data(vote_threshold=1000, M=5000):
    download_data()
    df = pd.read_csv(FILE_NAME)

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

    movie_similarity_df = pd.DataFrame()
    normalized_titles_dict = {normalize_title(t): t for t in df_filtered["TITLE"].unique()}

    return df, df_filtered, movie_similarity_df, normalized_titles_dict


def find_best_match(input_title, normalized_titles_dict):
    normalized_input = normalize_title(input_title)
    close = difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=1)
    return normalized_titles_dict[close[0]] if close else None


def suggest_alternatives(input_title, normalized_titles_dict):
    norm = normalize_title(input_title)
    return [normalized_titles_dict[t] for t in difflib.get_close_matches(norm, normalized_titles_dict.keys(), n=3)]


def recommend_by_title(title, n=5, watched=None, normalized_titles_dict=None):
    watched = watched or set()
    match = find_best_match(title, normalized_titles_dict)
    if not match:
        st.error("❌ Film bulunamadı. Belki şunları kastettiniz:")
        for alt in suggest_alternatives(title, normalized_titles_dict):
            st.write(f"- {alt}")
        return []
    st.info(f"🎯 '{match}' filmine göre önerilenler:")
    all_titles = set(normalized_titles_dict.values())
    recs = list(all_titles - watched - {match})
    return recs[:n]


def top_movies_by_year(df, year, n=5):
    try:
        year = int(year)
        year_movies = df[df['YEAR'] == year]
        if year_movies.empty:
            st.error(f"⚠️ {year} yılına ait film bulunamadı.")
            return []
        top = year_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(n)
        st.info(f"🗓️ {year} yılına ait en yüksek IMDb skoruna sahip filmler:")
        for i, (title, score) in enumerate(top.items(), 1):
            st.write(f"{i}. {title} - IMDb Skoru: {score:.2f}")
        return top.index.tolist()
    except ValueError:
        st.error("⚠️ Geçersiz yıl girdisi.")
        return []


def recommend_by_genre(df, genre, n=5):
    genre = genre.strip().title()
    genre_movies = df[df["GENRES"].str.contains(genre, case=False, na=False)]
    if genre_movies.empty:
        st.error(f"⚠️ '{genre}' türünde film bulunamadı.")
        return []
    top = genre_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(n)
    st.info(f"🎬 '{genre}' türünde en yüksek IMDb skoruna sahip filmler:")
    for i, (title, score) in enumerate(top.items(), 1):
        st.write(f"{i}. {title} - IMDb Skoru: {score:.2f}")
    return top.index.tolist()


def main():
    st.title("🎞️ KodBlessYou - IMDB Film Tavsiye Sistemi")

    df, df_filtered, sim_df, norm_dict = prepare_data()

    watched_movies = set()

    menu = st.sidebar.selectbox(
        "🔍 Seçim senin, sinema tutkun!",
        ["Film Tavsiye Edebilirim", "Yılın En İyileri", "Tür Kategorisinde En İyiler"]
    )

    if menu == "Film Tavsiye Edebilirim":
        film = st.text_input("🎬 İzlediğin ve unutamadığın o filmi yaz:")
        if film:
            recs = recommend_by_title(film, n=5, watched=watched_movies, normalized_titles_dict=norm_dict)
            if recs:
                st.success("✅ Önerilen Filmler:")
                for i, film in enumerate(recs, 1):
                    score = df[df["TITLE"] == film]["IMDB_SCORE"].mean()
                    st.write(f"{i}. {film} - IMDb Skoru: {score:.2f}")
                    watched_movies.add(film)
            else:
                st.warning("🔍 Öneri bulunamadı.")

    elif menu == "Yılın En İyileri":
        year_input = st.text_input("📅 Bir yıl girin (örnek: 2015), o yılın en iyilerini keşfedelim:")
        if year_input:
            top_movies_by_year(df_filtered, year_input)

    elif menu == "Tür Kategorisinde En İyiler":
        st.write("🎞️ Kullanabileceğiniz film türlerinden bazıları:")
        st.write(
            "Action | Comedy | Drama | Romance | Thriller | Sci-Fi | Horror | Adventure | Animation | Crime | Mystery | Fantasy | War | Western | Documentary | Musical | Family | Biography")
        genre_input = st.text_input("🎬 Film türü seç, sana en güzel önerileri getirelim:")
        if genre_input:
            recommend_by_genre(df_filtered, genre_input)


if __name__ == "__main__":
    main()
