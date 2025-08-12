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

# Pandas ayarları
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.2f}'.format

# Google Drive dosya ID
FILE_ID = "1QF-RRX3vf1jxiLMbdJQEQTYygeHlupPE"
FILE_NAME = "movies_imdb_2.csv"

# --- CSS Netflix Stili ---
st.markdown("""
    <style>
    body {
        background-color: #141414;
        color: #FFFFFF;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        color: #E50914;
        font-weight: 900;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 0;
        width: 100%;
        transition: background-color 0.3s ease-in-out;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #B20710;
    }
    .section {
        margin-top: 25px;
        margin-bottom: 25px;
    }
    .sidebar .sidebar-content {
        background-color: #141414;
        color: white;
    }
    .sidebar .sidebar-content h2 {
        color: #E50914;
    }
    .stTextInput>div>input {
        background-color: #1f1f1f;
        color: white;
        border: 1.5px solid #E50914;
        border-radius: 8px;
        padding: 8px;
    }
    .stTextInput>div>input:focus {
        border-color: #B20710;
        outline: none;
    }
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 10px;
        padding: 15px;
        font-weight: 600;
        background-color: #1f1f1f;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

def download_data():
    if not os.path.exists(FILE_NAME):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, FILE_NAME, quiet=False)
        st.success(f"📥 {FILE_NAME} indirildi!")
    else:
        st.info(f"✅ {FILE_NAME} zaten mevcut, indirme atlandı.")

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
        st.error("Veri setinde kullanıcı bilgisi yok.")
        return df, df_filtered, pd.DataFrame(), pd.DataFrame(), {}

    user_movie_matrix = df_filtered.pivot_table(
        index="USERID",
        columns="TITLE",
        values="RATING_10",
        aggfunc='mean'
    ).fillna(0)

    if user_movie_matrix.shape[0] == 0 or user_movie_matrix.shape[1] == 0:
        st.error("Öneri sistemi için yeterli veri yok.")
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

def recommend_by_title(title, sim_df, n=5, watched=None, normalized_titles_dict=None):
    watched = watched or set()
    match = find_best_match(title, normalized_titles_dict)
    if not match:
        st.error("❌ Film bulunamadı. Belki şunları kastettiniz:")
        return []
    st.info(f"🎯 **{match}** filmine göre önerilenler:")
    scores = sim_df[match].drop(labels=watched.union({match}), errors="ignore")
    return scores.sort_values(ascending=False).head(n).index.tolist()

def recommend_by_user(user_id, user_matrix, sim_df, n=5):
    if user_id not in user_matrix.index:
        st.error("❌ Kullanıcı bulunamadı.")
        return []
    user_ratings = user_matrix.loc[user_id]
    watched = user_ratings[user_ratings > 0]
    if watched.empty:
        st.warning("ℹ️ Kullanıcının izlediği film bilgisi yok.")
        return []
    scores = sim_df[watched.index].dot(watched)
    scores = scores.drop(watched.index, errors='ignore')
    return scores.sort_values(ascending=False).head(n).index.tolist()

def top_movies_by_year(df, year, n=5):
    try:
        year = int(year)
        year_movies = df[df['YEAR'] == year]
        if year_movies.empty:
            st.error(f"{year} yılına ait film bulunamadı.")
            return []
        top = year_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(n)
        st.info(f"📅 {year} yılının en iyileri:")
        for i, (title, score) in enumerate(top.items(), 1):
            st.write(f"{i}. {title} ⭐ {score:.2f}")
        return top.index.tolist()
    except ValueError:
        st.error("⚠️ Geçersiz yıl.")
        return []

def recommend_by_genre(df, genre, n=5):
    genre = genre.strip().title()
    genre_movies = df[df["GENRES"].str.contains(genre, case=False, na=False)]
    if genre_movies.empty:
        st.error(f"'{genre}' türünde film bulunamadı.")
        return []
    top = genre_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(n)
    st.info(f"🎭 {genre} türünde en iyiler:")
    for i, (title, score) in enumerate(top.items(), 1):
        st.write(f"{i}. {title} ⭐ {score:.2f}")
    return top.index.tolist()

def main():
    st.markdown("<h1 class='title'>🎬 Netflix Tarzı IMDb Film Öneri Sistemi</h1>", unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("🎥 Menü")
    if st.sidebar.button("📥 Veri Setini İndir"):
        download_data()

    df, df_filtered, user_movie_matrix, sim_df, norm_dict = prepare_data()
    if sim_df.empty:
        return

    watched_movies = set()

    menu = st.sidebar.selectbox(
        "Menü Seç",
        ["🎯 Sana Özel Öneriler", "👤 Kullanıcıya Göre", "📅 Yılın En İyileri", "🎭 Türüne Göre"]
    )

    if menu == "🎯 Sana Özel Öneriler":
        film = st.text_input("İzlediğin ve beğendiğin bir filmi yaz:")
        if film:
            recs = recommend_by_title(film, sim_df, n=5, watched=watched_movies, normalized_titles_dict=norm_dict)
            for i, movie in enumerate(recs, 1):
                score = df[df["TITLE"] == movie]["IMDB_SCORE"].mean()
                st.write(f"{i}. {movie} ⭐ {score:.2f}")
                watched_movies.add(movie)

    elif menu == "👤 Kullanıcıya Göre":
        user_id_input = st.text_input("Kullanıcı ID'sini gir:")
        if user_id_input:
            try:
                user_id = int(user_id_input.strip())
                recs = recommend_by_user(user_id, user_movie_matrix, sim_df)
                for i, movie in enumerate(recs, 1):
                    score = df[df["TITLE"] == movie]["IMDB_SCORE"].mean()
                    st.write(f"{i}. {movie} ⭐ {score:.2f}")
            except ValueError:
                st.error("Geçersiz ID.")

    elif menu == "📅 Yılın En İyileri":
        year_input = st.text_input("Bir yıl gir (örn: 2015):")
        if year_input:
            top_movies_by_year(df_filtered, year_input)

    elif menu == "🎭 Türüne Göre":
        st.write("🎬 Örnek türler: Action, Comedy, Drama, Romance, Sci-Fi, Horror, Adventure...")
        genre_input = st.text_input("Film türü gir:")
        if genre_input:
            recommend_by_genre(df_filtered, genre_input)

if __name__ == "__main__":
    main()
