import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
import gdown
import os
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI VE TASARIM
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MovieMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700;900&display=swap');

    /* ARKA PLAN */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* GENEL YAZI TİPİ */
    .stApp, p, span, div, label, button {
        color: #e0e0e0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif !important;
        color: #ffffff !important;
    }

    /* HERO SECTION (BAŞLIK ALANI) */
    .hero-container {
        text-align: center;
        padding: 4rem 0 3rem 0;
        background: radial-gradient(circle at center, rgba(120, 80, 255, 0.15) 0%, transparent 70%);
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
    }
    
    .subtitle {
        font-size: 1.4rem !important;
        color: #b0b0d0 !important;
        font-weight: 300 !important;
        letter-spacing: 1px;
    }

    /* FİLM KARTLARI */
    div.movie-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px 16px 0 0;
        padding: 24px;
        height: 240px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
    }

    div.movie-card:hover {
        border-color: #00c6ff;
        box-shadow: 0 10px 30px rgba(0, 198, 255, 0.15);
    }

    .card-title {
        color: #ffffff !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        margin-bottom: 12px;
        height: 3.2em;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        line-height: 1.3;
    }

    .meta-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }

    .year-badge {
        font-size: 0.9rem;
        color: #a0a0c0 !important;
        background: rgba(255,255,255,0.1);
        padding: 2px 8px;
        border-radius: 6px;
    }

    .score-badge {
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        color: white !important;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        box-shadow: 0 2px 10px rgba(0, 114, 255, 0.3);
    }

    .genre-text {
        font-size: 0.8rem;
        color: #8888aa !important;
        margin-bottom: 10px;
    }
    
    .match-rate {
        font-size: 0.85rem;
        color: #00c6ff !important;
        font-weight: 600;
        margin-top: auto;
    }

    /* ETKİLEŞİM BUTONLARI (KART ALTI) */
    div[data-testid="column"] button {
        background: rgba(0, 198, 255, 0.1) !important;
        border: 1px solid rgba(0, 198, 255, 0.3) !important;
        color: #00c6ff !important;
        border-radius: 0 0 16px 16px !important;
        margin-top: -5px !important; /* Karta yapışık olsun */
        transition: all 0.3s !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="column"] button:hover {
        background: linear-gradient(90deg, #00c6ff, #0072ff) !important;
        color: white !important;
        border-color: transparent !important;
        transform: translateY(2px);
    }

    /* ARAMA ÇUBUĞU */
    .stTextInput input {
        border-radius: 50px !important;
        padding: 18px 25px !important;
        background: rgba(0, 0, 0, 0.3) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        font-size: 1.1rem !important;
        color: white !important;
    }
    
    .stTextInput input:focus {
        border-color: #00c6ff !important;
        box-shadow: 0 0 20px rgba(0, 198, 255, 0.2) !important;
    }

    /* ADAY BUTONLARI (ÜST KISIM) */
    .candidate-btn button {
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: #cccccc !important;
        border-radius: 30px !important;
    }
    
    .candidate-btn button:hover {
        border-color: #00c6ff !important;
        color: white !important;
    }

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MOTOR (HESAPLAMA & VERİ)
# -----------------------------------------------------------------------------

@st.cache_resource(ttl=3600)
def download_data_from_drive(file_id):
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        output_file = "movies_imdb_2.csv"
        if not os.path.exists(output_file):
            gdown.download(url, output_file, quiet=False)
        return output_file
    except Exception as e:
        st.error(f"Veri bağlantı hatası: {str(e)}")
        return None

@st.cache_data
def normalize_title(title):
    return ''.join(
        c for c in unicodedata.normalize('NFD', title)
        if unicodedata.category(c) != 'Mn'
    ).lower().strip()

@st.cache_resource(ttl=3600, show_spinner=False)
def prepare_data(filepath, vote_threshold=1000, min_votes=2500):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")
        df.dropna(subset=["TITLE", "YEAR", "RATING"], inplace=True)
        df["YEAR"] = df["YEAR"].astype(int)
        df["RATING_10"] = df["RATING"] * 2
        
        vote_counts = df.groupby("TITLE", sort=False)["RATING"].count()
        df["NUM_VOTES"] = df["TITLE"].map(vote_counts)
        mean_rating = df["RATING_10"].mean()
        
        movie_stats = df.groupby("TITLE", sort=False).agg({
            "RATING_10": "mean",
            "NUM_VOTES": "max",
            "YEAR": "first",
            "GENRES": "first"
        }).reset_index()
        
        movie_stats["IMDB_SCORE"] = (
            (movie_stats["NUM_VOTES"] / (movie_stats["NUM_VOTES"] + min_votes)) * movie_stats["RATING_10"] +
            (min_votes / (movie_stats["NUM_VOTES"] + min_votes)) * mean_rating
        )
        
        movie_metadata = movie_stats.set_index("TITLE")[["IMDB_SCORE", "YEAR", "GENRES"]].to_dict('index')

        popular_titles = vote_counts[vote_counts >= vote_threshold].index
        df_filtered = df[df["TITLE"].isin(popular_titles)].copy()
        
        user_movie_matrix = df_filtered.pivot_table(
            index="USERID", columns="TITLE", values="RATING_10", aggfunc='mean'
        ).fillna(0)
        
        movie_similarity_df = pd.DataFrame(
            cosine_similarity(user_movie_matrix.T),
            index=user_movie_matrix.columns,
            columns=user_movie_matrix.columns
        )
        
        normalized_titles_dict = {normalize_title(t): t for t in movie_similarity_df.columns}
        
        return movie_similarity_df, normalized_titles_dict, movie_metadata
        
    except Exception as e:
        return None, None, None

def find_movie_candidates(query, _normalized_titles_dict, top_n=5):
    normalized_input = normalize_title(query)
    close_matches = difflib.get_close_matches(normalized_input, _normalized_titles_dict.keys(), n=top_n, cutoff=0.3)
    return [_normalized_titles_dict[m] for m in close_matches]

@st.cache_data
def get_recommendations_for_selected(_similarity_df, _movie_metadata, selected_movie, top_n):
    if selected_movie not in _similarity_df.columns:
        return None
    scores = _similarity_df[selected_movie].drop(labels=[selected_movie], errors="ignore")
    recommendations = scores.nlargest(top_n)
    
    rec_data = []
    for movie, similarity_score in recommendations.items():
        if movie in _movie_metadata:
            meta = _movie_metadata[movie]
            rec_data.append({
                "Film": movie,
                "Benzerlik": float(similarity_score),
                "IMDb": float(meta['IMDB_SCORE']),
                "Yıl": int(meta['YEAR']),
                "Türler": meta['GENRES'].replace("|", ", ")
            })
    return rec_data

# -----------------------------------------------------------------------------
# 3. ETKİLEŞİMLİ KARTLAR (DÖNGÜ MEKANİZMASI)
# -----------------------------------------------------------------------------
def display_interactive_cards(movies_data, col_count=5):
    cols = st.columns(col_count)
    
    for idx, movie in enumerate(movies_data):
        with cols[idx % col_count]:
            # Görsel HTML Kısmı
            st.markdown(f"""
            <div class="movie-card">
                <div style="font-size: 3.5rem; text-align: center; margin-bottom: 15px; text-shadow: 0 0 20px rgba(255,255,255,0.2);">🎬</div>
                <div class="card-title" title="{movie['Film']}">{movie['Film']}</div>
                <div class="meta-row">
                    <span class="year-badge">{movie['Yıl']}</span>
                    <span class="score-badge">{movie['IMDb']:.1f}</span>
                </div>
                <div class="genre-text">
                    {movie.get('Türler', 'Genel')[:25]}...
                </div>
                <div class="match-rate">
                    %{int(movie["Benzerlik"]*100)} Eşleşme
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Etkileşim Butonu
            if st.button(f"Bunu Analiz Et ⚡", key=f"rec_{idx}", use_container_width=True):
                st.session_state.selected_movie_final = movie['Film']
                st.session_state.candidates = [] # Temiz sayfa
                st.rerun()

# -----------------------------------------------------------------------------
# 4. ANA UYGULAMA
# -----------------------------------------------------------------------------

def main():
    # Session State Başlatma
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.movie_similarity_df = None
        st.session_state.normalized_titles_dict = None
        st.session_state.movie_metadata = None
    
    if 'candidates' not in st.session_state:
        st.session_state.candidates = []
    if 'selected_movie_final' not in st.session_state:
        st.session_state.selected_movie_final = None

    # Hero Alanı
    st.markdown("""
        <div class="hero-container">
            <h1 class="main-title">MovieMind AI</h1>
            <p class="subtitle">Yapay Zeka Tabanlı Akıllı Tavsiye Motoru</p>
        </div>
    """, unsafe_allow_html=True)

    # Veri Yükleme (Sessiz)
    if not st.session_state.data_loaded:
        with st.spinner('🧠 Nöral ağlar yükleniyor...'):
            FILE_ID = "1gl_iJXRyEaSzhHlgfBUdTzQZMer4gdsS"
            filepath = download_data_from_drive(FILE_ID)
            if filepath:
                result = prepare_data(filepath)
                if result[0] is not None:
                    (st.session_state.movie_similarity_df, 
                     st.session_state.normalized_titles_dict, 
                     st.session_state.movie_metadata) = result
                    st.session_state.data_loaded = True
                    st.rerun()
                else:
                    st.error("Sistem başlatılamadı.")
                    st.stop()
            else:
                st.stop()

    # --- ANA ARAYÜZ (TEK SAYFA) ---
    
    # 1. ARAMA MOTORU
    c1, c2, c3 = st.columns([1, 6, 1]) # Ortalamak için
    with c2:
        search_query = st.text_input("movie_search", 
                                   placeholder="Hangi filmi sevdiniz? (Örn: Matrix, Interstellar...)", 
                                   label_visibility="collapsed")
        
        # Enter'a basıldığında veya buton kullanıldığında
        if search_query:
            # Sadece yeni bir arama yapıldıysa adayları güncelle
            if 'last_query' not in st.session_state or st.session_state.last_query != search_query:
                st.session_state.candidates = find_movie_candidates(search_query, st.session_state.normalized_titles_dict)
                st.session_state.last_query = search_query
                st.session_state.selected_movie_final = None # Yeni aramada seçimi sıfırla

    # 2. ADAY FİLMLER (Sadece arama yapıldığında ve seçim yoksa görünür)
    if st.session_state.candidates and not st.session_state.selected_movie_final:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; color:#888; margin-bottom:15px;'>🤔 <b>'{search_query}'</b> için bulduklarımız:</div>", unsafe_allow_html=True)
        
        # Butonları ortala
        cols = st.columns(len(st.session_state.candidates))
        for i, movie in enumerate(st.session_state.candidates):
            # Özel stil sınıfı eklemek için container kullanabiliriz ama basit tutalım
            if cols[i].button(movie, key=f"cand_{i}", use_container_width=True):
                st.session_state.selected_movie_final = movie
                st.session_state.candidates = [] # Seçim yapıldı, listeyi kaldır
                st.rerun()

    # 3. SONUÇLAR VE ANALİZ
    if st.session_state.selected_movie_final:
        st.markdown("---")
        
        # Başlık ve Geri Dönme Hissi
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:center; gap:10px; margin-bottom:30px;">
            <span style="font-size:1.5rem; color:#fff;">Seçiminiz:</span>
            <span style="font-size:1.5rem; font-weight:bold; color:#00c6ff;">{st.session_state.selected_movie_final}</span>
        </div>
        """, unsafe_allow_html=True)
        
        recs = get_recommendations_for_selected(
            st.session_state.movie_similarity_df,
            st.session_state.movie_metadata,
            st.session_state.selected_movie_final,
            5
        )
        
        if recs:
            display_interactive_cards(recs)
        else:
            st.warning("Bu film için yeterli veri bulunamadı.")

if __name__ == "__main__":
    main()
