import pandas as pd
import numpy as np
import unicodedata
import difflib
import logging
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def weighted_rating(r, v, M, C):
    denom = v + M
    if denom == 0:
        return 0
    return (v / denom) * r + (M / denom) * C

def normalize_title(title):
    return ''.join(
        c for c in unicodedata.normalize('NFD', title)
        if unicodedata.category(c) != 'Mn'
    ).lower().strip()

@st.cache_data(show_spinner=True)
def load_raw_data(path="others/movies_imdb.csv"):
    df = pd.read_csv(path)
    return df

@st.cache_data(show_spinner=True)
def preprocess_data(df, vote_threshold=1000, M=2500):
    df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")
    df["TIME"] = pd.to_datetime(df["TIME"], dayfirst=True, errors='coerce')
    df.dropna(subset=["TITLE", "YEAR", "TIME", "RATING"], inplace=True)
    df["YEAR"] = df["YEAR"].astype(int)
    df["RATING_10"] = df["RATING"] * 2
    vote_counts = df.groupby("TITLE")["RATING"].count()
    df["NUM_VOTES"] = df["TITLE"].map(vote_counts)
    C = df["RATING_10"].mean()
    movie_stats = df.groupby("TITLE").agg({
        "RATING_10": "mean",
        "NUM_VOTES": "max"
    }).reset_index()
    movie_stats["IMDB_SCORE"] = movie_stats.apply(
        lambda x: weighted_rating(x["RATING_10"], x["NUM_VOTES"], M, C),
        axis=1
    )
    df["IMDB_SCORE"] = df["TITLE"].map(movie_stats.set_index("TITLE")["IMDB_SCORE"])
    popular_titles = vote_counts[vote_counts >= vote_threshold].index
    df_filtered = df[df["TITLE"].isin(popular_titles)].copy()
    return df, df_filtered

@st.cache_data(show_spinner=True)
def build_user_movie_matrix(df_filtered):
    user_movie_matrix = df_filtered.pivot_table(
        index="USERID",
        columns="TITLE",
        values="RATING_10",
        aggfunc='mean'
    ).fillna(0)
    return user_movie_matrix

@st.cache_data(show_spinner=True)
def build_movie_similarity(user_movie_matrix):
    movie_similarity_df = pd.DataFrame(
        cosine_similarity(user_movie_matrix.T),
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns
    )
    return movie_similarity_df

@st.cache_data(show_spinner=True)
def build_normalized_titles_dict(movie_similarity_df):
    normalized_titles_dict = {normalize_title(t): t for t in movie_similarity_df.columns}
    return normalized_titles_dict

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
        return None, suggest_alternatives(title, normalized_titles_dict)
    scores = sim_df[match].drop(labels=watched.union({match}), errors="ignore")
    return match, scores.sort_values(ascending=False).head(n).index.tolist()

def recommend_by_user(user_id, user_matrix, sim_df, n=5):
    if user_id not in user_matrix.index:
        return None
    user_ratings = user_matrix.loc[user_id]
    watched = user_ratings[user_ratings > 0]
    if watched.empty:
        return []
    scores = sim_df[watched.index].dot(watched)
    scores = scores.drop(watched.index, errors='ignore')
    return scores.sort_values(ascending=False).head(n).index.tolist()

def top_movies_by_year(df, year, n=5):
    try:
        year = int(year)
        year_movies = df[df['YEAR'] == year]
        if year_movies.empty:
            return None
        top = year_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(n)
        return top
    except ValueError:
        return None

def recommend_by_genre(df, genre, n=5):
    genre = genre.strip().title()
    genre_movies = df[df["GENRES"].str.contains(genre, case=False, na=False)]
    if genre_movies.empty:
        return None
    top = genre_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(n)
    return top

def app():
    st.title("🎞️ KodBlessYou Film Tavsiye Sistemi")

    # Yalnızca buton ile veri yükle ve işle, otomatik değil
    if st.sidebar.button("Veriyi Yükle ve Hazırla"):
        with st.spinner("Veriler hazırlanıyor..."):
            df = load_raw_data()
            df, df_filtered = preprocess_data(df)
            user_movie_matrix = build_user_movie_matrix(df_filtered)
            sim_df = build_movie_similarity(user_movie_matrix)
            norm_dict = build_normalized_titles_dict(sim_df)
            # Veri cache'lenir ve globalde tutmak için session_state kullanalım
            st.session_state['df'] = df
            st.session_state['df_filtered'] = df_filtered
            st.session_state['user_movie_matrix'] = user_movie_matrix
            st.session_state['sim_df'] = sim_df
            st.session_state['norm_dict'] = norm_dict
            st.success("Veriler hazırlandı! Menüden seçim yapabilirsiniz.")

    if 'df' not in st.session_state:
        st.info("Lütfen önce yukarıdaki butona basarak veriyi yükleyip hazırlayın.")
        return

    df = st.session_state['df']
    df_filtered = st.session_state['df_filtered']
    user_movie_matrix = st.session_state['user_movie_matrix']
    sim_df = st.session_state['sim_df']
    norm_dict = st.session_state['norm_dict']

    menu = st.sidebar.selectbox("Seçim yapınız:", [
        "Film adına göre öneri",
        "Kullanıcı ID'ye göre öneri",
        "Yıla göre en iyi filmler",
        "Türüne göre film önerisi"
    ])

    if menu == "Film adına göre öneri":
        film = st.text_input("🎬 Film adı girin:")
        if film:
            match, recs = recommend_by_title(film, sim_df, n=5, normalized_titles_dict=norm_dict)
            if match is None:
                st.error("❌ Film bulunamadı. Belki şunları kastettiniz:")
                for alt in recs:
                    st.warning(f"- {alt}")
            else:
                st.success(f"🎯 '{match}' filmine göre önerilenler:")
                for i, rec in enumerate(recs, 1):
                    score = df[df["TITLE"] == rec]["IMDB_SCORE"].mean()
                    st.write(f"{i}. {rec} — IMDb Skoru: {score:.2f}")

    elif menu == "Kullanıcı ID'ye göre öneri":
        top_users = df["USERID"].value_counts().head(10).index.tolist()
        st.write(f"En aktif kullanıcı ID'leri: {', '.join(map(str, top_users))}")
        user_id = st.number_input("Bir kullanıcı ID girin:", min_value=int(df["USERID"].min()), max_value=int(df["USERID"].max()), step=1)
        if user_id:
            recs = recommend_by_user(user_id, user_movie_matrix, sim_df)
            if recs is None:
                st.error(f"❌ Kullanıcı ID {user_id} bulunamadı.")
            elif not recs:
                st.info("ℹ️ Kullanıcının izlediği film verisi yok.")
            else:
                st.success("✅ Önerilen Filmler:")
                for i, rec in enumerate(recs, 1):
                    score = df[df["TITLE"] == rec]["IMDB_SCORE"].mean()
                    st.write(f"{i}. {rec} — IMDb Skoru: {score:.2f}")

    elif menu == "Yıla göre en iyi filmler":
        year = st.text_input("Bir yıl girin (ör: 2015):")
        if year:
            top = top_movies_by_year(df_filtered, year)
            if top is None:
                st.error(f"⚠️ {year} yılına ait film bulunamadı veya geçersiz yıl.")
            else:
                st.success(f"🗓️ {year} yılına ait en yüksek IMDb skoruna sahip filmler:")
                for i, (title, score) in enumerate(top.items(), 1):
                    st.write(f"{i}. {title} — IMDb Skoru: {score:.2f}")

    elif menu == "Türüne göre film önerisi":
        st.write("🎞️ Kullanabileceğiniz film türlerinden bazıları:")
        st.write("Action, Comedy, Drama, Romance, Thriller, Biography, Horror, Adventure, Animation, Crime, Mystery, Fantasy, War, Western, Documentary, Musical")
        genre = st.text_input("Film türü girin:")
        if genre:
            top = recommend_by_genre(df_filtered, genre)
            if top is None:
                st.error(f"⚠️ '{genre}' türünde film bulunamadı.")
            else:
                st.success(f"🎬 '{genre}' türünde en yüksek IMDb skoruna sahip filmler:")
                for i, (title, score) in enumerate(top.items(), 1):
                    st.write(f"{i}. {title} — IMDb Skoru: {score:.2f}")

if __name__ == "__main__":
    app()
