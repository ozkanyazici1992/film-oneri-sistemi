```python
import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
import gdown
from sklearn.metrics.pairwise import cosine_similarity
import logging

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.2f}'.format

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

# --- GOOGLE DRIVE VERİ SETİ BİLGİLERİ ---
DRIVE_FILE_ID = "1mdXIj3yZWd6cNV8hc5T2rTMjjAEbfgfL"
DATASET_FILE_NAME = "movies_imdb.csv"
# --- --- --- --- --- --- --- --- --- ---

@st.cache_data
def prepare_data(filepath=DATASET_FILE_NAME, vote_threshold=1000, min_votes=2500):
    try:
        gdown.download(f'https://drive.google.com/uc?id={DRIVE_FILE_ID}', filepath, quiet=False)
        st.success(f"'{DATASET_FILE_NAME}' dosyası başarıyla indirildi.")

    except Exception as e:
        st.error(f"Google Drive'dan dosya indirilemedi: {e}")
        st.info("Lütfen Drive dosya ID'sinin ve dosyanın adının doğru olduğundan emin olun ve dosyanızın herkese açık olarak paylaşıldığından emin olun.")
        return None, None, None, None, None
        
    df = pd.read_csv(filepath)
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

    st.success("Veri hazırlığı tamamlandı. Uygulama kullanıma hazır!")
    return df, df_filtered, user_movie_matrix, movie_similarity_df, normalized_titles_dict

def find_best_match(input_title, normalized_titles_dict):
    normalized_input = normalize_title(input_title)
    close_matches = difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=1)
    return normalized_titles_dict[close_matches[0]] if close_matches else None

def suggest_alternatives(input_title, normalized_titles_dict):
    normalized_input = normalize_title(input_title)
    return [normalized_titles_dict[t] for t in difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=3)]

def recommend_by_title(title, similarity_df, top_n, watched, normalized_titles_dict):
    match = find_best_match(title, normalized_titles_dict)

    if not match:
        st.warning(f"❌ '{title}' bulunamadı. Şunlardan biri olabilir mi?")
        for alternative in suggest_alternatives(title, normalized_titles_dict):
            st.info(f"- {alternative}")
        return []

    st.success(f"🎯 '{match}' filmine göre tavsiyeler:")
    scores = similarity_df[match].drop(labels=watched.union({match}), errors="ignore")
    return scores.sort_values(ascending=False).head(top_n).index.tolist()

def recommend_by_user(user_id, user_matrix, similarity_df, top_n=5):
    if user_id not in user_matrix.index:
        st.error(f"❌ Kullanıcı ID'si '{user_id}' bulunamadı.")
        return []

    user_ratings = user_matrix.loc[user_id]
    watched = user_ratings[user_ratings > 0]

    if watched.empty:
        st.warning("ℹ️ Bu kullanıcı için izleme geçmişi bulunamadı.")
        return []

    scores = similarity_df[watched.index].dot(watched)
    scores = scores.drop(watched.index, errors='ignore')
    return scores.sort_values(ascending=False).head(top_n).index.tolist()

def top_movies_by_year(df, year, top_n=5):
    try:
        year = int(year)
        year_movies = df[df['YEAR'] == year]
        if year_movies.empty:
            st.warning(f"⚠️ {year} yılı için film bulunamadı.")
            return []
        
        top = year_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(top_n)
        st.success(f"🗓️ {year} yılı için en iyi IMDb puanlı filmler:")
        return top.index.tolist(), top
    except ValueError:
        st.error("⚠️ Geçersiz yıl girişi.")
        return [], None

def recommend_by_genre(df, genre, top_n=5):
    genre = genre.strip().title()
    genre_movies = df[df["GENRES"].str.contains(genre, case=False, na=False)]
    if genre_movies.empty:
        st.warning(f"⚠️ '{genre}' türünde film bulunamadı.")
        return []
    
    top = genre_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(top_n)
    st.success(f"🎬 '{genre}' türündeki en iyi IMDb puanlı filmler:")
    return top.index.tolist(), top

def main_streamlit():
    st.set_page_config(page_title="KodBlessYou Film Tavsiyeleri", layout="centered")
    st.title("🎞️ KodBlessYou Film Tavsiyeleri")
    st.markdown("🔍 **Aradığınız filmleri bulmak için bir yöntem seçin.**")

    df, df_filtered, user_movie_matrix, similarity_df, norm_titles = prepare_data()
    if df is None:
        return

    watched_movies = set()
    
    choice = st.selectbox(
        "Tavsiye türü seçiniz:",
        ("Film Adına Göre", "Kullanıcı Geçmişine Göre", "Yıla Göre", "Türe Göre")
    )
    
    st.markdown("---")

    if choice == "Film Adına Göre":
        st.header("🎥 Film Adına Göre Tavsiyeler")
        movie_input = st.text_input("Sevdiğiniz bir filmin adını girin:")
        if st.button("Tavsiye Et", key="title_button"):
            if movie_input:
                recommendations = recommend_by_title(movie_input, similarity_df, top_n=5, watched=watched_movies, normalized_titles_dict=norm_titles)
                if recommendations:
                    st.subheader("✅ Önerilen Filmler:")
                    for i, rec_movie in enumerate(recommendations, 1):
                        score = df[df["TITLE"] == rec_movie]["IMDB_SCORE"].mean()
                        st.write(f"{i}. **{rec_movie}** - IMDb Puanı: {score:.2f}")
                        watched_movies.add(rec_movie)
                else:
                    st.info("🔍 Tavsiye bulunamadı.")
            else:
                st.warning("⚠️ Film adı boş bırakılamaz.")

    elif choice == "Kullanıcı Geçmişine Göre":
        st.header("🧑‍💻 Kullanıcı Geçmişine Göre Tavsiyeler")
        top_users = df["USERID"].value_counts().head(10).index.tolist()
        st.info(f"En aktif 10 Kullanıcı ID'si: {', '.join(map(str, top_users))}")
        user_input = st.text_input("Bir Kullanıcı ID'si girin:")
        if st.button("Tavsiye Et", key="user_button"):
            try:
                user_id = int(user_input)
                recommendations = recommend_by_user(user_id, user_movie_matrix, similarity_df)
                if recommendations:
                    st.subheader("✅ Önerilen Filmler:")
                    for i, rec_movie in enumerate(recommendations, 1):
                        score = df[df["TITLE"] == rec_movie]["IMDB_SCORE"].mean()
                        st.write(f"{i}. **{rec_movie}** - IMDb Puanı: {score:.2f}")
                else:
                    st.info("🔍 Tavsiye bulunamadı.")
            except ValueError:
                st.error("⚠️ Geçersiz Kullanıcı ID'si girişi. Lütfen sayısal bir değer girin.")

    elif choice == "Yıla Göre":
        st.header("📅 Yıla Göre En İyi Filmler")
        year_input = st.text_input("Bir yıl girin (örn: 2015):")
        if st.button("Listele", key="year_button"):
            if year_input:
                recommendations, top_scores = top_movies_by_year(df_filtered, year_input)
                if recommendations:
                    st.subheader("✅ En İyi Filmler:")
                    for i, (rec_movie, score) in enumerate(zip(recommendations, top_scores), 1):
                        st.write(f"{i}. **{rec_movie}** - IMDb Puanı: {score:.2f}")
            else:
                st.warning("⚠️ Yıl boş bırakılamaz.")

    elif choice == "Türe Göre":
        st.header("🎭 Türe Göre Tavsiyeler")
        genres = ["Action", "Comedy", "Drama", "Romance", "Thriller", "Biography", "Horror", "Adventure", "Animation", "Crime", "Mystery", "Fantasy", "War", "Western", "Documentary", "Musical"]
        genre_input = st.selectbox("Bir tür seçin:", genres)
        if st.button("Tavsiye Et", key="genre_button"):
            if genre_input:
                recommendations, top_scores = recommend_by_genre(df_filtered, genre_input)
                if recommendations:
                    st.subheader("✅ En İyi Filmler:")
                    for i, (rec_movie, score) in enumerate(zip(recommendations, top_scores), 1):
                        st.write(f"{i}. **{rec_movie}** - IMDb Puanı: {score:.2f}")
            else:
                st.warning("⚠️ Tür boş bırakılamaz.")

if __name__ == "__main__":
    main_streamlit()
```
