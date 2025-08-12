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

FILE_ID = "1QF-RRX3vf1jxiLMbdJQEQTYygeHlupPE"
FILE_NAME = "movies_imdb_2.csv"

# --- CSS Styling ---
st.markdown("""
    <style>
    /* Genel gövde arkaplan ve font */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Başlıklar renk ve margin */
    .title {
        color: #4A90E2;
        font-weight: 700;
        margin-bottom: 10px;
    }
    /* Ortalanmış ve büyük input */
    .centered-input > div > input {
        margin-left: auto;
        margin-right: auto;
        display: block;
        width: 50%;
        font-size: 18px;
        padding: 8px 12px;
        border-radius: 8px;
        border: 1.5px solid #4A90E2;
        transition: border-color 0.3s ease-in-out;
    }
    .centered-input > div > input:focus {
        border-color: #357ABD;
        outline: none;
    }
    /* Sidebar başlık */
    .sidebar .sidebar-content h2 {
        color: #4A90E2;
        font-weight: 700;
    }
    /* Buton stil */
    .stButton>button {
        background-color: #4A90E2;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 0;
        width: 100%;
        transition: background-color 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #357ABD;
    }
    /* Aralıklar */
    .section {
        margin-top: 25px;
        margin-bottom: 25px;
    }
    /* Bilgilendirme mesaj renkleri */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 10px;
        padding: 15px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

def download_data():
    if not os.path.exists(FILE_NAME):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, FILE_NAME, quiet=False)
        st.success(f"{FILE_NAME} başarıyla indirildi!")
    else:
        st.info(f"{FILE_NAME} zaten mevcut, indirme atlandı.")

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

    if "USERID" not in df_filtered.columns:
        st.error("Veri setinde USERID sütunu bulunamadı.")
        return df, df_filtered, pd.DataFrame(), pd.DataFrame(), {}

    user_movie_matrix = df_filtered.pivot_table(
        index="USERID",
        columns="TITLE",
        values="RATING_10",
        aggfunc='mean'
    ).fillna(0)

    if user_movie_matrix.shape[0] == 0 or user_movie_matrix.shape[1] == 0:
        st.error("Öneri sistemi için yeterli kullanıcı-film verisi bulunamadı.")
        return df, df_filtered, user_movie_matrix, pd.DataFrame(), {}

    movie_similarity_df = pd.DataFrame(
        cosine_similarity(user_movie_matrix.T),
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns
    )

    normalized_titles_dict = {normalize_title(t): t for t in movie_similarity_df.columns}
    return df, df_filtered, user_movie_matrix, movie_similarity_df, normalized_titles_dict

def find_best_match(input_title, normalized_titles_dict):
    normalized_input = normalize_title(input_title)
    close = difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=1)
    return normalized_titles_dict[close[0]] if close else None

def suggest_alternatives(input_title, normalized_titles_dict):
    norm = normalize_title(input_title)
    return [normalized_titles_dict[t] for t in difflib.get_close_matches(norm, normalized_titles_dict.keys(), n=3)]

def recommend_by_title(title, sim_df, n=5, watched=None, normalized_titles_dict=None):
    watched = watched or set()
    match = find_best_match(title, normalized_titles_dict)
    if not match:
        st.error("❌ Film bulunamadı. Belki şunları kastettiniz:")
        for alt in suggest_alternatives(title, normalized_titles_dict):
            st.write(f"- {alt}")
        return []
    st.info(f"🎯 '{match}' filmine göre önerilenler:")
    scores = sim_df[match].drop(labels=watched.union({match}), errors="ignore")
    return scores.sort_values(ascending=False).head(n).index.tolist()

def recommend_by_user(user_id, user_matrix, sim_df, n=5):
    if user_id not in user_matrix.index:
        st.error(f"❌ Kullanıcı ID {user_id} bulunamadı.")
        return []
    user_ratings = user_matrix.loc[user_id]
    watched = user_ratings[user_ratings > 0]
    if watched.empty:
        st.warning("ℹ️ Kullanıcının izlediği film verisi yok.")
        return []
    scores = sim_df[watched.index].dot(watched)
    scores = scores.drop(watched.index, errors='ignore')
    return scores.sort_values(ascending=False).head(n).index.tolist()

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
    st.markdown("<h1 class='title'>🎞️ KodBlessYou - IMDB Film Tavsiye Sistemi</h1>", unsafe_allow_html=True)

    # Sidebar: Veri seti indirme ve menü
    st.sidebar.header("⚙️ Ayarlar")
    if st.sidebar.button("📥 Veri Setini İndir"):
        download_data()

    df, df_filtered, user_movie_matrix, sim_df, norm_dict = prepare_data()
    if sim_df.empty:
        st.error("Öneri sistemi için gerekli veriler eksik veya yetersiz.")
        return

    watched_movies = set()

    menu = st.sidebar.selectbox(
        "🔍 Seçim senin, sinema tutkun!",
        ["Film Tavsiye Edebilirim", "Kullanıcıya Göre Öneriler", "Yılın En İyileri", "Tür Kategorisinde En İyiler"]
    )

    if menu == "Film Tavsiye Edebilirim":
        st.markdown("<div class='section'><h4 style='color:#4A90E2;'>🎬 İzlediğin ve unutamadığın o filmi yaz:</h4></div>", unsafe_allow_html=True)
        film = st.text_input("", key="film_input")
        if film:
            recs = recommend_by_title(film, sim_df, n=5, watched=watched_movies, normalized_titles_dict=norm_dict)
            if recs:
                st.success("✅ Önerilen Filmler:")
                for i, film in enumerate(recs, 1):
                    score = df[df["TITLE"] == film]["IMDB_SCORE"].mean()
                    st.write(f"{i}. {film} - IMDb Skoru: {score:.2f}")
                    watched_movies.add(film)
            else:
                st.warning("🔍 Öneri bulunamadı.")

    elif menu == "Kullanıcıya Göre Öneriler":
        st.markdown("<div class='section centered-input'><h4 style='text-align:center; color:#4A90E2;'>Kullanıcı ID'sini giriniz:</h4></div>", unsafe_allow_html=True)
        user_id_input = st.text_input("", key="user_id_input")
        if user_id_input and user_id_input.strip():
            try:
                user_id = int(user_id_input.strip())
                recs = recommend_by_user(user_id, user_movie_matrix, sim_df)
                if recs:
                    st.success("✅ Önerilen Filmler:")
                    for i, film in enumerate(recs, 1):
                        score = df[df["TITLE"] == film]["IMDB_SCORE"].mean()
                        st.write(f"{i}. {film} - IMDb Skoru: {score:.2f}")
                else:
                    st.warning("🔍 Öneri bulunamadı.")
            except ValueError:
                st.error("❌ Geçersiz kullanıcı ID formatı. Lütfen sadece sayı girin.")
        else:
            st.info("Lütfen kullanıcı ID'si giriniz.")

    elif menu == "Yılın En İyileri":
        st.markdown("<div class='section'><h4 style='color:#4A90E2;'>📅 Bir yıl girin (örnek: 2015), o yılın en iyilerini keşfedelim:</h4></div>", unsafe_allow_html=True)
        year_input = st.text_input("", key="year_input")
        if year_input:
            top_movies_by_year(df_filtered, year_input)

    elif menu == "Tür Kategorisinde En İyiler":
        st.markdown("<div class='section'><h4 style='color:#4A90E2;'>🎞️ Kullanabileceğiniz film türlerinden bazıları:</h4></div>", unsafe_allow_html=True)
        st.write(
            "Action | Comedy | Drama | Romance | Thriller | Sci-Fi | Horror | Adventure | Animation | Crime | Mystery | Fantasy | War | Western | Documentary | Musical | Family | Biography")
        st.markdown("<div class='section'><h4 style='color:#4A90E2;'>🎬 Film türü seç, sana en güzel önerileri getirelim:</h4></div>", unsafe_allow_html=True)
        genre_input = st.text_input("", key="genre_input")
        if genre_input:
            recommend_by_genre(df_filtered, genre_input)

if __name__ == "__main__":
    main()
