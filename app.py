import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
import gdown
import os
import gc # Hafıza yönetimi için eklendi
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. SAYFA VE PERFORMANS AYARLARI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CineAI | Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS: Koyu Tema ve Stabilite Ayarları
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* Arka Planı Zorla Siyah Yap (Flash Effect'i önler) */
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117 !important;
        background-image: radial-gradient(circle at top left, #1a2a3a, #000000) !important;
    }
    
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }

    /* Yazı Tipleri ve Renkler */
    .stApp, p, h1, h2, h3, h4, h5, h6, span, div, label {
        color: #e0e0e0 !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* FORM ALANI TASARIMI (Arama Kutusu) */
    [data-testid="stForm"] {
        background-color: rgba(255, 255, 255, 0.03);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(64, 224, 208, 0.2);
    }

    /* Kart Tasarımı */
    div.movie-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        transition: transform 0.3s ease;
        height: 100%;
        backdrop-filter: blur(10px);
    }
    
    div.movie-card:hover {
        transform: translateY(-5px);
        border-color: #40E0D0;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.5);
    }

    .card-title {
        color: #40E0D0 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-bottom: 10px;
        height: 3.5em;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }

    .score-badge {
        background-color: #40E0D0;
        color: #000 !important;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.8rem;
    }

    /* Butonlar */
    .stButton > button {
        background: linear-gradient(45deg, #40E0D0, #008B8B) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. VERİ YÜKLEME VE İŞLEME (BELLEK OPTİMİZASYONLU)
# -----------------------------------------------------------------------------

@st.cache_resource(ttl=3600)
def load_data_engine():
    """Veriyi indirir ve hesaplamaları yapar. Sadece 1 kere çalışır."""
    try:
        # 1. Dosya İndirme
        file_id = "1gl_iJXRyEaSzhHlgfBUdTzQZMer4gdsS"
        output_file = "movies_imdb_2.csv"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        if not os.path.exists(output_file):
            gdown.download(url, output_file, quiet=False)

        # 2. Veri Okuma
        df = pd.read_csv(output_file)
        
        # 3. Temizleme
        df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")
        df["TIME"] = pd.to_datetime(df["TIME"], dayfirst=True, errors='coerce')
        df.dropna(subset=["TITLE", "YEAR", "TIME", "RATING"], inplace=True)
        df["YEAR"] = df["YEAR"].astype(int)
        df["RATING_10"] = df["RATING"] * 2
        
        # 4. İstatistikler
        vote_counts = df.groupby("TITLE")["RATING"].count()
        df["NUM_VOTES"] = df["TITLE"].map(vote_counts)
        mean_rating = df["RATING_10"].mean()
        min_votes = 2500
        
        # Weighted Rating
        movie_stats = df.groupby("TITLE").agg({
            "RATING_10": "mean",
            "NUM_VOTES": "max"
        }).reset_index()
        
        def weighted_rating(x):
            v = x["NUM_VOTES"]
            R = x["RATING_10"]
            return (v / (v + min_votes) * R) + (min_votes / (v + min_votes) * mean_rating)

        movie_stats["IMDB_SCORE"] = movie_stats.apply(weighted_rating, axis=1)
        df["IMDB_SCORE"] = df["TITLE"].map(movie_stats.set_index("TITLE")["IMDB_SCORE"])
        
        # 5. MATRİS OPTİMİZASYONU (Donmayı engelleyen kısım)
        # Sadece en popüler filmleri matrise alıyoruz (Bellek tasarrufu)
        POPULARITY_THRESHOLD = 3000 # En çok oy alan ilk 3000 film ile matris kur
        popular_titles = vote_counts.sort_values(ascending=False).head(POPULARITY_THRESHOLD).index
        df_filtered = df[df["TITLE"].isin(popular_titles)].copy()
        
        user_movie_matrix = df_filtered.pivot_table(
            index="USERID", columns="TITLE", values="RATING_10", aggfunc='mean'
        ).fillna(0)
        
        # Matrix'i float32'ye çevir (RAM kullanımını yarıya indirir)
        matrix_sparse = user_movie_matrix.T.astype('float32')
        
        movie_similarity_df = pd.DataFrame(
            cosine_similarity(matrix_sparse),
            index=user_movie_matrix.columns,
            columns=user_movie_matrix.columns
        )
        
        # Başlık normalizasyonu
        def normalize(t): 
            return ''.join(c for c in unicodedata.normalize('NFD', str(t)) if unicodedata.category(c) != 'Mn').lower().strip()
            
        normalized_titles_dict = {normalize(t): t for t in movie_similarity_df.columns}
        
        # Gereksiz değişkenleri sil ve RAM'i temizle
        del user_movie_matrix
        del matrix_sparse
        gc.collect()
        
        return df, df_filtered, movie_similarity_df, normalized_titles_dict
        
    except Exception as e:
        st.error(f"Veri motoru hatası: {str(e)}")
        return None, None, None, None

def normalize_title(title):
    return ''.join(c for c in unicodedata.normalize('NFD', title) if unicodedata.category(c) != 'Mn').lower().strip()

def recommend_movies(title, similarity_df, df, top_n=5, normalized_dict=None):
    norm_input = normalize_title(title)
    
    # Tam eşleşme veya en yakın eşleşme bul
    matches = difflib.get_close_matches(norm_input, normalized_dict.keys(), n=1)
    
    if not matches:
        # Bulunamazsa alternatif öner
        alternatives = [normalized_dict[t] for t in difflib.get_close_matches(norm_input, normalized_dict.keys(), n=3)]
        return None, alternatives
    
    match_title = normalized_dict[matches[0]]
    
    # Benzerlik skorlarını al
    scores = similarity_df[match_title].sort_values(ascending=False).drop(match_title).head(top_n)
    
    results = []
    for movie_title, score in scores.items():
        info = df[df["TITLE"] == movie_title].iloc[0]
        results.append({
            "Film": movie_title,
            "Benzerlik": score,
            "IMDb": info['IMDB_SCORE'],
            "Yıl": int(info["YEAR"]),
            "Türler": str(info["GENRES"]).replace("|", ", ")
        })
        
    return results, match_title

# -----------------------------------------------------------------------------
# 3. ARAYÜZ (MAIN)
# -----------------------------------------------------------------------------

def main():
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 class="main-title">CineAI Pro</h1>
            <p style="color: #a0a0a0;">Donmayan, Hızlı ve Akıllı Film Öneri Sistemi</p>
        </div>
    """, unsafe_allow_html=True)

    # Veriyi Yükle (Spinner ile)
    with st.spinner('🚀 Sistem hazırlanıyor... (İlk açılışta 30sn sürebilir)'):
        df, df_filtered, similarity_df, norm_dict = load_data_engine()

    if df is None:
        st.stop()

    # --- SEKME YAPISI ---
    tab1, tab2, tab3 = st.tabs(["🔍 Film Ara", "🏆 Trendler", "📊 Analiz"])

    # ----------------------
    # SEKME 1: ARAMA (FORM İLE OPTİMİZE EDİLDİ)
    # ----------------------
    with tab1:
        col_center, _ = st.columns([2, 1])
        
        # FORM BAŞLANGICI (BU SAYEDE HER TUŞA BASINCA SAYFA YENİLENMEZ)
        with st.form(key='search_form'):
            c1, c2 = st.columns([3, 1])
            with c1:
                search_input = st.text_input("Film Adı", placeholder="Örn: Inception, Matrix...", label_visibility="collapsed")
            with c2:
                # Formu tetikleyen buton
                submit_button = st.form_submit_button("🔍 FİLMİ BUL")
            
            # Sayı seçimi
            count_option = st.slider("Öneri Sayısı", 4, 12, 4, 4)

        # Butona basıldıysa işlem yap (Donmayı önleyen mantık burası)
        if submit_button and search_input:
            recommendations, result_info = recommend_movies(
                search_input, similarity_df, df, count_option, norm_dict
            )
            
            if recommendations:
                st.success(f"✅ **{result_info}** için önerilerimiz:")
                st.markdown("---")
                
                # Sonuçları Kart Olarak Göster
                cols = st.columns(4)
                for idx, movie in enumerate(recommendations):
                    with cols[idx % 4]:
                        html = f"""
                        <div class="movie-card">
                            <div style="font-size: 2.5rem; text-align: center;">🎬</div>
                            <div class="card-title" title="{movie['Film']}">{movie['Film']}</div>
                            <div style="display: flex; justify-content: space-between; color: #aaa; font-size: 0.9rem;">
                                <span>📅 {movie['Yıl']}</span>
                                <span class="score-badge">★ {movie['IMDb']:.1f}</span>
                            </div>
                            <div style="margin-top: 10px; font-size: 0.8rem; color: #40E0D0;">
                                Uyum: %{int(movie['Benzerlik']*100)}
                            </div>
                        </div>
                        """
                        st.markdown(html, unsafe_allow_html=True)
                        st.write("") # Boşluk
            
            elif result_info: # Alternatifler
                st.warning("Tam eşleşme bulunamadı. Bunları mı kastettiniz?")
                for alt in result_info:
                    st.info(f"👉 {alt}")

    # ----------------------
    # SEKME 2: TRENDLER
    # ----------------------
    with tab2:
        st.markdown("### 📅 Yılın En İyileri")
        years = sorted(df['YEAR'].unique(), reverse=True)
        selected_year = st.selectbox("Yıl Seçiniz", years)
        
        top_movies = df[df['YEAR'] == selected_year].sort_values('IMDB_SCORE', ascending=False).head(8)
        
        # Basit tablo gösterimi (Hız için)
        cols_trend = st.columns(4)
        for idx, (_, row) in enumerate(top_movies.iterrows()):
            with cols_trend[idx % 4]:
                st.markdown(f"""
                <div class="movie-card">
                    <div style="font-size: 2rem; text-align: center;">🏆</div>
                    <div class="card-title">{row['TITLE']}</div>
                    <span class="score-badge">★ {row['IMDB_SCORE']:.1f}</span>
                </div>
                """, unsafe_allow_html=True)
                st.write("")

    # ----------------------
    # SEKME 3: ANALİZ
    # ----------------------
    with tab3:
        st.markdown("### 📊 Veri Analizi")
        # Hızlı grafik
        genre_counts = df['GENRES'].str.get_dummies('|').sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=genre_counts.values, y=genre_counts.index, orientation='h', title="En Popüler Türler")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        fig.update_traces(marker_color='#40E0D0')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
