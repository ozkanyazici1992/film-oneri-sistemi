import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
import gdown
import os
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI VE ULTRA MODERN CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CineAI | Film Keşif Asistanı",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# SİNEMA TEMASINA UYGUN ULTRA MODERN TASARIM
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700;900&display=swap');

    /* KOYU SİNEMA TEMASI */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        background-attachment: fixed;
    }
    
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }

    /* TİPOGRAFİ */
    .stApp, p, span, div, label {
        color: #e8e8e8 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif !important;
        color: #ffffff !important;
    }

    /* HERO SECTION */
    .hero-container {
        text-align: center;
        padding: 3rem 0 4rem 0;
        background: radial-gradient(ellipse at center, rgba(218,165,32,0.15) 0%, transparent 70%);
        position: relative;
    }
    
    .main-title {
        font-size: 4.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #DAA520 0%, #FFD700 50%, #FFA500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 40px rgba(218,165,32,0.5);
        margin-bottom: 1rem;
    }
    
    .subtitle {
        font-size: 1.2rem !important;
        color: #c0c0c0 !important;
        font-weight: 300 !important;
    }

    /* FİLM KARTLARI */
    div.movie-card {
        background: linear-gradient(145deg, rgba(26,26,46,0.8), rgba(22,33,62,0.6));
        border: 1px solid rgba(218,165,32,0.2);
        border-radius: 20px;
        padding: 25px;
        transition: all 0.4s;
        height: 100%;
        position: relative;
    }

    div.movie-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: #DAA520;
        box-shadow: 0 10px 30px rgba(218,165,32,0.2);
    }

    .card-title {
        color: #FFD700 !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        margin-bottom: 12px;
        height: 3.5em;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }

    .score-badge {
        background: linear-gradient(135deg, #DAA520, #FFD700);
        color: #000000 !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* ADAY BUTONLARI (Candidate Buttons) */
    .stButton button {
        background: rgba(40,40,60,0.8) !important;
        color: #fff !important;
        border: 1px solid rgba(218,165,32,0.3) !important;
        border-radius: 15px !important;
        transition: all 0.2s !important;
        height: auto !important;
        white-space: normal !important; /* Uzun isimler alt satıra geçsin */
        padding: 10px 20px !important;
    }

    .stButton button:hover {
        background: rgba(218,165,32,0.2) !important;
        border-color: #FFD700 !important;
        color: #FFD700 !important;
        transform: scale(1.02);
    }
    
    /* Input Alanı */
    .stTextInput input {
        border-radius: 30px !important;
        padding: 15px !important;
        border: 2px solid rgba(218,165,32,0.4) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid rgba(218,165,32,0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(218,165,32,0.15) !important;
        color: #FFD700 !important;
        border-bottom-color: #DAA520 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. VERİ İŞLEME VE FONKSİYONLAR
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
        
        # Temel temizlik
        df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")
        df["TIME"] = pd.to_datetime(df["TIME"], dayfirst=True, errors='coerce')
        df.dropna(subset=["TITLE", "YEAR", "TIME", "RATING"], inplace=True)
        df["YEAR"] = df["YEAR"].astype(int)
        df["RATING_10"] = df["RATING"] * 2
        
        # İstatistikler ve Skorlama
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
        
        df["IMDB_SCORE"] = df["TITLE"].map(movie_stats.set_index("TITLE")["IMDB_SCORE"])
        movie_metadata = movie_stats.set_index("TITLE")[["IMDB_SCORE", "YEAR", "GENRES"]].to_dict('index')

        # Similarity Matrix
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
        
        return df, df_filtered, movie_similarity_df, normalized_titles_dict, movie_metadata
        
    except Exception as e:
        return None, None, None, None, None

def find_movie_candidates(query, _normalized_titles_dict, top_n=5):
    """Girilen isme en yakın 5 filmi bulur."""
    normalized_input = normalize_title(query)
    close_matches = difflib.get_close_matches(normalized_input, _normalized_titles_dict.keys(), n=top_n, cutoff=0.3)
    real_titles = [_normalized_titles_dict[m] for m in close_matches]
    return real_titles

@st.cache_data
def get_recommendations_for_selected(_similarity_df, _movie_metadata, selected_movie, top_n):
    """Seçilen film için önerileri getirir."""
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

def display_movie_cards(movies_data, col_count=5):
    cols = st.columns(col_count)
    for idx, movie in enumerate(movies_data):
        with cols[idx % col_count]:
            html_content = f"""
            <div class="movie-card">
                <div style="font-size: 3.5rem; text-align: center; margin-bottom: 12px;">🎬</div>
                <div class="card-title" title="{movie['Film']}">{movie['Film']}</div>
                <div class="card-metric">
                    <span>📅 {movie['Yıl']}</span>
                    <span class="score-badge">★ {movie['IMDb']:.1f}</span>
                </div>
                <div style="font-size: 0.85rem; color: #888; margin-top: 8px; line-height: 1.4;">
                    {movie.get('Türler', 'Genel')[:35]}...
                </div>
                <div style="margin-top:12px; font-size:0.85rem; color:#DAA520; font-weight:600;">Eşleşme: %{int(movie["Benzerlik"]*100)}</div>
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. ANA UYGULAMA
# -----------------------------------------------------------------------------

def main():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Hero Section
    st.markdown("""
        <div class="hero-container">
            <h1 class="main-title">CineAI</h1>
            <p class="subtitle">Yapay Zeka Destekli Premium Film Keşif Platformu</p>
        </div>
    """, unsafe_allow_html=True)

    # Veri Yükleme
    if not st.session_state.data_loaded:
        with st.spinner('🎬 CineAI motoru başlatılıyor...'):
            FILE_ID = "1gl_iJXRyEaSzhHlgfBUdTzQZMer4gdsS"
            filepath = download_data_from_drive(FILE_ID)
            if filepath:
                result = prepare_data(filepath)
                if result[0] is not None:
                    (st.session_state.df, st.session_state.df_filtered, 
                     st.session_state.movie_similarity_df, st.session_state.normalized_titles_dict, 
                     st.session_state.movie_metadata) = result
                    st.session_state.data_loaded = True
                    st.rerun()
                else:
                    st.stop()
            else:
                st.stop()

    df = st.session_state.df
    movie_similarity_df = st.session_state.movie_similarity_df
    normalized_titles_dict = st.session_state.normalized_titles_dict
    movie_metadata = st.session_state.movie_metadata

    # State Yönetimi
    if 'candidates' not in st.session_state:
        st.session_state.candidates = []
    if 'selected_movie_final' not in st.session_state:
        st.session_state.selected_movie_final = None

    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 İstatistikler")
        st.metric("🎬 Toplam Film", f"{df['TITLE'].nunique()//1000}K+")
        st.metric("👥 Kullanıcı", f"{df['USERID'].nunique()//1000}K+")

    # TABS
    tab1, tab2 = st.tabs(["🔍 Film Önerisi", "📊 Veri Analizi"])

    # --- TAB 1: FILM ÖNERİSİ (GÜNCELLENMİŞ AKIŞ) ---
    with tab1:
        st.markdown("### 🎬 Film Arayın")
        
        # 1. ARAMA KISMI
        c1, c2 = st.columns([5, 1])
        with c1:
            query = st.text_input("Film adı", placeholder="Örn: batman, matrix, yüzüklerin...", label_visibility="collapsed", key="main_search")
        with c2:
            search_btn = st.button("🔎 Ara", type="primary")

        # Butona basınca adayları bul ve state'e kaydet
        if search_btn and query:
            st.session_state.candidates = find_movie_candidates(query, normalized_titles_dict, top_n=5)
            st.session_state.selected_movie_final = None # Yeni aramada eski sonuçları temizle

        # 2. ADAYLARI BUTON OLARAK GÖSTER (Eğer aday varsa)
        if st.session_state.candidates:
            st.markdown("---")
            st.info("👇 **Aşağıdakilerden hangisini kastettiniz? (Tıklayınca öneriler gelir)**")
            
            # Her film için bir kolon oluştur (Butonlar yan yana dursun)
            cols = st.columns(len(st.session_state.candidates))
            
            for i, movie_title in enumerate(st.session_state.candidates):
                # Her butona unique key veriyoruz
                if cols[i].button(movie_title, key=f"btn_{i}", use_container_width=True):
                    # BUTONA TIKLANDIĞI AN:
                    st.session_state.selected_movie_final = movie_title
                    # (Opsiyonel: Aday listesini temizleyip sadece sonucu gösterebiliriz ama kalsın ki fikrini değiştirebilsin)

        # 3. SONUÇ EKRANI (Seçim yapıldıysa)
        if st.session_state.selected_movie_final:
            with st.spinner("🧠 Yapay zeka analiz yapıyor..."):
                recs = get_recommendations_for_selected(
                    movie_similarity_df,
                    movie_metadata,
                    st.session_state.selected_movie_final,
                    5 # Standart 5 öneri
                )
            
            st.markdown("---")
            st.success(f"✨ **{st.session_state.selected_movie_final}** için seçtiğimiz filmler:")
            display_movie_cards(recs, col_count=5)

    # --- TAB 2: ANALİZ ---
    with tab2:
        st.markdown("### 📊 Veri İçgörüleri")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("**Popüler Türler**")
            genres = df['GENRES'].str.get_dummies(sep='|').sum().sort_values().tail(10)
            fig = px.bar(x=genres.values, y=genres.index, orientation='h')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            fig.update_traces(marker_color='#DAA520')
            st.plotly_chart(fig, use_container_width=True)
        with col_a2:
            st.markdown("**IMDb Dağılımı**")
            fig2 = px.histogram(st.session_state.df_filtered, x='IMDB_SCORE', nbins=20)
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            fig2.update_traces(marker_color='#FFD700')
            st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
