import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
import logging
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import os

# ——— Logging ayarları ———
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ——— Pandas gösterim ayarları ———
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.2f}'.format

FILE_ID = "1QF-RRX3vf1jxiLMbdJQEQTYygeHlupPE"
FILE_NAME = "movies_imdb_2.csv"

# ——— Dosya indirme fonksiyonu ———
def download_data():
    if not os.path.exists(FILE_NAME):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        try:
            gdown.download(url, FILE_NAME, quiet=False)
            logging.info(f"{FILE_NAME} başarıyla indirildi.")
        except Exception as e:
            logging.error(f"Dosya indirme hatası: {e}")
            st.error(f"Dosya indirme sırasında hata oluştu: {e}")
            return False
    else:
        logging.info(f"{FILE_NAME} zaten mevcut, indirme atlandı.")
    return True

# ——— IMDb ağırlıklı skor hesaplama ———
def weighted_rating(r, v, M, C):
    denom = v + M
    if denom == 0:
        return 0
    return (v / denom) * r + (M / denom) * C

# ——— Film başlıklarını normalize etme ———
def normalize_title(title):
    return ''.join(c for c in unicodedata.normalize('NFD', title) if unicodedata.category(c) != 'Mn').lower().strip()

# ——— Veriyi hazırla ve cache'le ———
@st.cache_data(show_spinner=True)
def prepare_data(vote_threshold=1000, M=5000):
    if not download_data():
        return None, None, None, None, None
    
    df = pd.read_csv(FILE_NAME)
    
    # Başlık ve yıl ayıklama
    df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")
    
    # Tarih formatı
    df["TIME"] = pd.to_datetime(df["TIME"], dayfirst=True, errors='coerce')
    
    # Gerekli sütunlarda eksik varsa çıkar
    df.dropna(subset=["TITLE", "YEAR", "TIME", "RATING"], inplace=True)
    
    df["YEAR"] = df["YEAR"].astype(int)
    df["RATING_10"] = df["RATING"] * 2
    
    # Oy sayısı hesapla
    vote_counts = df.groupby("TITLE")["RATING"].count()
    df["NUM_VOTES"] = df["TITLE"].map(vote_counts).astype('int32')
    
    # Ortalama genel puan
    C = df["RATING_10"].mean()
    
    # Film bazında ortalama ve oy sayısı
    movie_stats = df.groupby("TITLE").agg({
        "RATING_10": "mean",
        "NUM_VOTES": "max"
    }).reset_index()
    
    # IMDb ağırlıklı puan hesapla
    movie_stats["IMDB_SCORE"] = movie_stats.apply(
        lambda x: weighted_rating(x["RATING_10"], x["NUM_VOTES"], M, C), axis=1
    )
    
    df["IMDB_SCORE"] = df["TITLE"].map(movie_stats.set_index("TITLE")["IMDB_SCORE"])
    
    # Popüler filmler (oy sayısı eşik üstü)
    popular_titles = vote_counts[vote_counts >= vote_threshold].index
    df_filtered = df[df["TITLE"].isin(popular_titles)].copy()

    # USERID sütunu kontrolü
    if "USERID" not in df_filtered.columns:
        st.error("Veri setinde USERID sütunu bulunamadı. Öneri sistemi kullanıcıya göre çalışmayacak.")
        return df, df_filtered, pd.DataFrame(), pd.DataFrame(), {}

    # Kullanıcı-film matrisi
    user_movie_matrix = df_filtered.pivot_table(
        index="USERID",
        columns="TITLE",
        values="RATING_10",
        aggfunc='mean'
    ).fillna(0)

    if user_movie_matrix.shape[0] == 0 or user_movie_matrix.shape[1] == 0:
        st.error("Öneri sistemi için yeterli kullanıcı-film verisi bulunamadı.")
        return df, df_filtered, user_movie_matrix, pd.DataFrame(), {}

    # Film-film benzerlik matrisi (cosine similarity)
    movie_similarity_df = pd.DataFrame(
        cosine_similarity(user_movie_matrix.T),
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns
    )

    # Normalize başlıklar dict
    normalized_titles_dict = {normalize_title(t): t for t in movie_similarity_df.columns}
    
    return df, df_filtered, user_movie_matrix, movie_similarity_df, normalized_titles_dict

# ——— Benzer başlık bul ———
def find_best_match(input_title, normalized_titles_dict):
    normalized_input = normalize_title(input_title)
    close = difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=1)
    return normalized_titles_dict[close[0]] if close else None

# ——— Alternatif film isimleri öner ———
def suggest_alternatives(input_title, normalized_titles_dict):
    norm = normalize_title(input_title)
    matches = difflib.get_close_matches(norm, normalized_titles_dict.keys(), n=3)
    return [normalized_titles_dict[t] for t in matches]

# ——— Film adına göre öneri ———
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

# ——— Kullanıcıya göre öneri ———
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

# ——— Yıla göre en iyi filmler ———
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

# ——— Tür bazlı öneri ———
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

# ——— Ana uygulama ———
def main():
    st.set_page_config(page_title="KodBlessYou IMDB Film Tavsiye Sistemi", layout="wide")
    st.title("🎞️ KodBlessYou - IMDB Film Tavsiye Sistemi")

    # Parametre ayarları kullanıcıdan alınabilir hale getirildi
    vote_threshold = st.sidebar.slider("Popülerlik için minimum oy sayısı (vote threshold):", 100, 5000, 1000, 100)
    M = st.sidebar.slider("IMDb ağırlıklı puan için minimum oy sayısı (M):", 100, 10000, 5000, 100)

    # Veri hazırla
    with st.spinner("Veriler yükleniyor ve işleniyor..."):
        df, df_filtered, user_movie_matrix, sim_df, norm_dict = prepare_data(vote_threshold, M)

    if df is None:
        st.error("Veri yükleme başarısız. Lütfen tekrar deneyin.")
        return

    if sim_df.empty or user_movie_matrix.empty:
        st.error("Öneri sistemi için yeterli veri bulunamadı.")
        return

    # Kullanıcının izlediği filmler takibi için session_state kullanımı
    if "watched_movies" not in st.session_state:
        st.session_state.watched_movies = set()

    menu = st.sidebar.selectbox(
        "🔍 Seçim senin, sinema tutkun!",
        ["Film Tavsiye Edebilirim", "Kullanıcıya Göre Öneriler", "Yılın En İyileri", "Tür Kategorisinde En İyiler"]
    )

    if menu == "Film Tavsiye Edebilirim":
        film = st.text_input("🎬 İzlediğin ve unutamadığın o filmi yaz:")
        num_recs = st.slider("Kaç öneri görmek istersin?", 1, 20, 5)

        if film:
            recs = recommend_by_title(film, sim_df, n=num_recs, watched=st.session_state.watched_movies, normalized_titles_dict=norm_dict)
            if recs:
                st.success("✅ Önerilen Filmler:")
                for i, film in enumerate(recs, 1):
                    score = df[df["TITLE"] == film]["IMDB_SCORE"].mean()
                    st.write(f"{i}. {film} - IMDb Skoru: {score:.2f}")
                    # İzlenen filmlere ekle
                    st.session_state.watched_movies.add(film)
            else:
                st.warning("🔍 Öneri bulunamadı.")

    elif menu == "Kullanıcıya Göre Öneriler":
        input_uid = st.text_input("Kullanıcı ID'sini giriniz:")
        num_recs = st.slider("Kaç öneri görmek istersin?", 1, 20, 5)

        if input_uid.strip():
            try:
                user_id = int(input_uid.strip())
                recs = recommend_by_user(user_id, user_movie_matrix, sim_df, n=num_recs)
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
        year_input = st.text_input("📅 Bir yıl girin (örnek: 2015), o yılın en iyilerini keşfedelim:")
        num_recs = st.slider("Kaç öneri görmek istersin?", 1, 20, 5)

        if year_input:
            top_movies_by_year(df_filtered, year_input, n=num_recs)

    elif menu == "Tür Kategorisinde En İyiler":
        st.write("🎞️ Kullanabileceğiniz film türlerinden bazıları:")
        st.write(
            "Action | Comedy | Drama | Romance | Thriller | Sci-Fi | Horror | Adventure | Animation | Crime | Mystery | Fantasy | War | Western | Documentary | Musical | Family | Biography"
        )
        genre_input = st.text_input("🎬 Film türü seç, sana en güzel önerileri getirelim:")
        num_recs = st.slider("Kaç öneri görmek istersin?", 1, 20, 5)

        if genre_input:
            recommend_by_genre(df_filtered, genre_input, n=num_recs)

if __name__ == "__main__":
    main()
