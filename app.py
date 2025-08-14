import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
import gdown
from sklearn.metrics.pairwise import cosine_similarity
import io

# --- Veri Yükleme ve Önbellekleme ---
# `movies_imdb.parquet` dosyasını Google Drive'dan indirir ve okur.
# `@st.cache_data` dekoratörü sayesinde bu işlem yalnızca bir kez çalışır ve sonuçları önbelleğe alır.
# Büyük veri setleri için idealdir ve uygulamanın hızını artırır.

@st.cache_data
def load_data_from_drive(file_id):
    """Google Drive'dan Parquet dosyasını indirir ve DataFrame olarak döndürür."""
    st.info("Veri seti Google Drive'dan indiriliyor ve yükleniyor. Bu işlem biraz zaman alabilir...")
    
    # Google Drive indirme URL'si
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        # Dosyayı bellekte tutarak indirme
        downloaded_data = gdown.download(url, output=None, quiet=False)
        
        # İndirilen veriyi bellekteki ikili veri akışı olarak DataFrame'e çevirme
        df = pd.read_parquet(io.BytesIO(downloaded_data))
        
        st.success("Veri seti başarıyla yüklendi!")
        return df
    except Exception as e:
        st.error(f"Veri seti yüklenirken bir hata oluştu: {e}")
        return None

# --- Yardımcı Fonksiyonlar ---
# Projenizdeki mevcut fonksiyonlar, Streamlit arayüzü ile uyumlu hale getirildi.

def weighted_rating(rating, votes, min_votes, mean_rating):
    """Ağırlıklı derecelendirme puanını hesaplar."""
    denominator = votes + min_votes
    if denominator == 0:
        return 0
    return (votes / denominator) * rating + (min_votes / denominator) * mean_rating

def normalize_title(title):
    """Film başlıklarını normalleştirir."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', title)
        if unicodedata.category(c) != 'Mn'
    ).lower().strip()

# --- Veri Hazırlığı ve Benzerlik Hesaplamaları ---
# Bu fonksiyon da `@st.cache_data` ile önbelleğe alınarak performans artırıldı.
# Veri hazırlığı ve matris hesaplamaları yalnızca bir kez yapılacaktır.

@st.cache_data
def prepare_and_analyze_data(df, vote_threshold=1000, min_votes=2500):
    """Veri hazırlığını ve benzerlik matrisi hesaplamalarını yapar."""
    st.info("Veriler işleniyor ve benzerlik matrisi oluşturuluyor...")
    
    df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")
    df.dropna(subset=["TITLE", "YEAR"], inplace=True)
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
    
    st.success("Veri hazırlığı tamamlandı! Uygulama kullanıma hazır.")
    return df, df_filtered, user_movie_matrix, movie_similarity_df, normalized_titles_dict

# --- Öneri Fonksiyonları ---
# Bu fonksiyonlar kullanıcı girdisine göre çağrılacak ve sonuçları döndürecektir.

def find_best_match(input_title, normalized_titles_dict):
    """Kullanıcı girdisine en yakın film başlığını bulur."""
    normalized_input = normalize_title(input_title)
    close_matches = difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=1)
    return normalized_titles_dict[close_matches[0]] if close_matches else None

def recommend_by_title(title, similarity_df, top_n, normalized_titles_dict):
    """Başlığa göre benzer filmleri önerir."""
    match = find_best_match(title, normalized_titles_dict)
    if not match:
        return None, difflib.get_close_matches(normalize_title(title), normalized_titles_dict.keys(), n=3)
    
    scores = similarity_df[match].drop(labels={match}, errors="ignore")
    return scores.sort_values(ascending=False).head(top_n).index.tolist(), None

def recommend_by_user(user_id, user_matrix, similarity_df, top_n=5):
    """Kullanıcı geçmişine göre filmleri önerir."""
    if user_id not in user_matrix.index:
        return [], "Kullanıcı ID'si bulunamadı."
    
    user_ratings = user_matrix.loc[user_id]
    watched = user_ratings[user_ratings > 0]
    
    if watched.empty:
        return [], "Kullanıcı için izleme geçmişi bulunamadı."
    
    scores = similarity_df[watched.index].dot(watched)
    scores = scores.drop(watched.index, errors='ignore')
    return scores.sort_values(ascending=False).head(top_n).index.tolist(), None

def top_movies_by_year(df, year, top_n=5):
    """Yıla göre en iyi filmleri listeler."""
    try:
        year_movies = df[df['YEAR'] == int(year)]
        if year_movies.empty:
            return [], "Bu yıl için film bulunamadı."
        
        top = year_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(top_n)
        return top.index.tolist(), None
    except ValueError:
        return [], "Geçersiz yıl formatı."

def recommend_by_genre(df, genre, top_n=5):
    """Janra göre en iyi filmleri önerir."""
    genre = genre.strip().title()
    genre_movies = df[df["GENRES"].str.contains(genre, case=False, na=False)]
    if genre_movies.empty:
        return [], f"'{genre}' janrında film bulunamadı."
    
    top = genre_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(top_n)
    return top.index.tolist(), None

# --- Ana Streamlit Arayüzü ---

def main():
    """Streamlit uygulamasının ana fonksiyonu."""
    st.set_page_config(page_title="IMDb Film Öneri Sistemi", layout="wide")
    st.title("🎬 IMDb Veri Seti ile Film Öneri Sistemi")
    st.markdown("---")
    
    FILE_ID = "13UKG6Dox3hUVg4_VZUWoQuz2pn3jOVZe"
    
    # Veri setini ve benzerlik matrisini önbellekleyerek yükleyin
    df_raw = load_data_from_drive(FILE_ID)
    
    if df_raw is not None:
        # Veri hazırlığı ve analizini önbellekleyerek yapın
        df, df_filtered, user_movie_matrix, similarity_df, norm_titles = prepare_and_analyze_data(df_raw.copy())

        st.sidebar.title("Menü")
        menu_choice = st.sidebar.radio(
            "Öneri Tipi Seçin:",
            ("Film Başlığına Göre", "Kullanıcı Geçmişine Göre", "Yıla Göre En İyiler", "Janra Göre En İyiler")
        )

        st.markdown("---")

        if menu_choice == "Film Başlığına Göre":
            st.header("🎥 Film Başlığına Göre Öneri")
            movie_title = st.text_input("Örnek: The Dark Knight", key="title_input")
            if st.button("Öner", key="title_btn"):
                if movie_title:
                    recommendations, alternatives = recommend_by_title(movie_title, similarity_df, 5, norm_titles)
                    if recommendations:
                        st.subheader(f"'{movie_title}' için Önerilen Filmler:")
                        for i, rec_movie in enumerate(recommendations, 1):
                            st.write(f"{i}. **{rec_movie}**")
                    else:
                        if alternatives:
                            st.error(f"Film bulunamadı. Şunları mı demek istediniz? {', '.join([norm_titles[alt] for alt in alternatives])}")
                        else:
                            st.error(f"'{movie_title}' ile ilgili herhangi bir film bulunamadı.")
                else:
                    st.warning("Lütfen bir film başlığı girin.")
        
        elif menu_choice == "Kullanıcı Geçmişine Göre":
            st.header("🧑‍💻 Kullanıcı Geçmişine Göre Öneri")
            top_users = df_filtered["USERID"].value_counts().head(10).index.tolist()
            user_id = st.selectbox("Bir Kullanıcı ID'si seçin:", top_users)
            if st.button("Öner", key="user_btn"):
                if user_id:
                    recommendations, error_msg = recommend_by_user(user_id, user_movie_matrix, similarity_df)
                    if recommendations:
                        st.subheader(f"Kullanıcı {user_id} için Önerilen Filmler:")
                        for i, rec_movie in enumerate(recommendations, 1):
                            st.write(f"{i}. **{rec_movie}**")
                    else:
                        st.warning(error_msg)
                else:
                    st.warning("Lütfen bir kullanıcı ID'si seçin.")

        elif menu_choice == "Yıla Göre En İyiler":
            st.header("📅 Yıla Göre En İyiler")
            year = st.text_input("Örnek: 2015", key="year_input")
            if st.button("Göster", key="year_btn"):
                if year:
                    recommendations, error_msg = top_movies_by_year(df_filtered, year)
                    if recommendations:
                        st.subheader(f"Yıl {year} için En İyi Filmler:")
                        for i, rec_movie in enumerate(recommendations, 1):
                            score = df_filtered[df_filtered["TITLE"] == rec_movie]["IMDB_SCORE"].mean()
                            st.write(f"{i}. **{rec_movie}** - IMDb Puanı: **{score:.2f}**")
                    else:
                        st.warning(error_msg)
                else:
                    st.warning("Lütfen bir yıl girin.")
        
        elif menu_choice == "Janra Göre En İyiler":
            st.header("🎭 Janra Göre En İyiler")
            genre_list = ['Action', 'Comedy', 'Drama', 'Romance', 'Thriller', 'Biography', 'Horror', 'Adventure', 'Animation', 'Crime', 'Mystery', 'Fantasy', 'War', 'Western', 'Documentary', 'Musical']
            genre = st.selectbox("Bir janr seçin:", genre_list)
            if st.button("Göster", key="genre_btn"):
                if genre:
                    recommendations, error_msg = recommend_by_genre(df_filtered, genre)
                    if recommendations:
                        st.subheader(f"'{genre}' Janrındaki En İyi Filmler:")
                        for i, rec_movie in enumerate(recommendations, 1):
                            score = df_filtered[df_filtered["TITLE"] == rec_movie]["IMDB_SCORE"].mean()
                            st.write(f"{i}. **{rec_movie}** - IMDb Puanı: **{score:.2f}**")
                    else:
                        st.warning(error_msg)
                else:
                    st.warning("Lütfen bir janr seçin.")

if __name__ == "__main__":
    main()
