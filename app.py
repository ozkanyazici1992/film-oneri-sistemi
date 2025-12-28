import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
import gdown
import os
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI VE ULTRA CINEMATIC CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MovieMind AI",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800&family=Bebas+Neue&display=swap');

    /* --- ARKA PLAN VE GENEL --- */
    [data-testid="stAppViewContainer"] {
        background-color: #000000;
        background-image: radial-gradient(circle at 50% 0%, #2b0c0d 0%, #000000 70%);
        background-attachment: fixed;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* YAZI TİPLERİ - SEO DOSTU VE OKUNAKLI */
    .stApp, p, span, div, label {
        color: #e0e0e0 !important;
        font-family: 'Montserrat', sans-serif !important;
        font-size: 0.95rem !important;
    }
    
    /* BAŞLIKLAR - SİNEMATİK */
    h1, h2, h3, h4 {
        font-family: 'Bebas Neue', cursive !important;
        color: #FFD700 !important; /* Oscar Gold */
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    /* HERO SECTION - RED CARPET STYLE */
    .hero-container {
        text-align: center;
        padding: 3rem 0 2rem 0;
        border-bottom: 2px solid linear-gradient(90deg, transparent, #E50914, transparent);
    }
    
    .main-title {
        font-size: 4.5rem !important;
        background: linear-gradient(to bottom, #FFD700 0%, #FDB931 50%, #9f7928 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 15px rgba(253, 185, 49, 0.4));
        margin-bottom: 10px;
    }
    
    .subtitle {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 300 !important;
        letter-spacing: 3px;
        text-transform: uppercase;
        opacity: 0.9;
    }

    /* SEÇİLEN FİLM PANELİ - BLOCKBUSTER DETAY */
    .selected-movie-info {
        background: rgba(20, 20, 20, 0.8);
        border: 1px solid #333;
        border-top: 4px solid #E50914; /* Netflix Red Line */
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.8);
        text-align: center;
        backdrop-filter: blur(10px);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .info-label {
        color: #E50914 !important; /* Kırmızı vurgu */
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 2px;
        margin-bottom: 5px;
        text-transform: uppercase;
    }
    
    .info-title {
        font-family: 'Bebas Neue', cursive !important;
        font-size: 2.2rem !important;
        color: #ffffff !important;
        margin-bottom: 10px;
        line-height: 1.1;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }
    
    .info-meta {
        display: flex;
        gap: 20px;
        justify-content: center;
        align-items: center;
        font-size: 1rem;
        color: #cccccc !important;
        margin-bottom: 15px;
    }
    
    .highlight-box {
        background: rgba(255, 215, 0, 0.1);
        border: 1px solid rgba(255, 215, 0, 0.3);
        color: #FFD700 !important;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 600;
    }

    /* ÖNERİ KARTLARI - AFİŞ GÖRÜNÜMÜ */
    div.movie-card {
        background: linear-gradient(180deg, #1a1a1a 0%, #0d0d0d 100%);
        border: 1px solid #333;
        border-radius: 8px 8px 0 0;
        padding: 15px; 
        height: 190px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }

    div.movie-card:hover {
        transform: translateY(-5px);
        border-color: #E50914; /* Hoverda kırmızı sınır */
        box-shadow: 0 10px 30px rgba(229, 9, 20, 0.2);
    }
    
    /* Kartın üzerine gelince hafif parlaması için */
    div.movie-card::after {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: linear-gradient(to bottom, transparent 60%, rgba(0,0,0,0.8) 100%);
        pointer-events: none;
    }

    .card-icon {
        font-size: 1.8rem;
        text-align: center;
        margin-bottom: 8px;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
    }

    .card-title {
        color: #ffffff !important; /* Tam beyaz */
        font-family: 'Montserrat', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        margin-bottom: 4px;
        height: 2.6em;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        line-height: 1.3;
        text-transform: capitalize;
    }

    .card-meta {
        font-size: 0.8rem !important;
        color: #999 !important;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .star-rating {
        color: #FFD700 !important;
        font-weight: 800;
    }

    .card-genre {
        font-size: 0.7rem !important;
        color: #bbb !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-top: 4px;
        font-style: normal;
    }

    .match-badge {
        position: absolute;
        top: 10px;
        right: 10px;
        background: #E50914;
        color: white !important;
        font-size: 0.65rem;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0,0,0,0.5);
    }

    /* ETKİLEŞİM BUTONLARI - CALL TO ACTION */
    div[data-testid="column"] button {
        background: #1f1f1f !important;
        border: 1px solid #333 !important;
        color: #E50914 !important; /* Kırmızı yazı */
        border-radius: 0 0 8px 8px !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        padding: 6px 0px !important;
        margin-top: -8px !important;
        transition: 0.2s !important;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="column"] button:hover {
        background: #E50914 !important;
        color: #ffffff !important;
        border-color: #E50914 !important;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.4);
    }

    /* INPUT ALANI - GOOGLE SEARCH TARZI AMA DARK */
    .stTextInput input {
        background-color: #1a1a1a !important;
        border: 2px solid #333 !important;
        color: #ffffff !important;
        border-radius: 50px !important; /* Tam yuvarlak */
        padding: 12px 20px !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    .stTextInput input:focus {
        border-color: #E50914 !important;
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.3) !important;
    }

    /* ADAY BUTONLARI */
    .element-container button {
        background-color: transparent !important;
        border: 1px solid #444 !important;
        color: #ccc !important;
        height: auto !important;
        min-height: 45px;
        white-space: normal !important;
        word-wrap: break-word !important;
    }
    .element-container button:hover {
        border-color: #FFD700 !important;
        color: #FFD700 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. VERİ MOTORU
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
        return None

@st.cache_data
def normalize_title(title):
    return ''.join(
        c for c in unicodedata.normalize('NFD', title)
        if unicodedata.category(c) != 'Mn'
    ).lower().strip()

@st.cache_resource(ttl=3600, show_spinner=False)
def prepare_data(filepath, min_votes=2500):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")
        df.dropna(subset=["TITLE", "YEAR", "RATING"], inplace=True)
        df["YEAR"] = df["YEAR"].astype(int)
        
        # Puan dönüşümü (10 üzerinden)
        df["RATING_10"] = df["RATING"] * 2
        
        movie_stats = df.groupby("TITLE", sort=False).agg({
            "RATING_10": "mean",
            "YEAR": "first",
            "GENRES": "first"
        }).reset_index()
        
        movie_metadata = movie_stats.set_index("TITLE").rename(columns={"RATING_10": "RATING"}).to_dict('index')

        vote_counts = df.groupby("TITLE")["RATING"].count()
        popular = vote_counts[vote_counts >= 1000].index
        df_filtered = df[df["TITLE"].isin(popular)]
        
        user_movie_matrix = df_filtered.pivot_table(index="USERID", columns="TITLE", values="RATING_10").fillna(0)
        movie_similarity_df = pd.DataFrame(cosine_similarity(user_movie_matrix.T), index=user_movie_matrix.columns, columns=user_movie_matrix.columns)
        normalized_titles_dict = {normalize_title(t): t for t in movie_similarity_df.columns}
        
        return movie_similarity_df, normalized_titles_dict, movie_metadata
    except:
        return None, None, None

def find_candidates(query, titles_dict):
    norm = normalize_title(query)
    matches = difflib.get_close_matches(norm, titles_dict.keys(), n=5, cutoff=0.3)
    return [titles_dict[m] for m in matches]

def get_recs(sim_df, meta, title):
    if title not in sim_df.columns: return []
    scores = sim_df[title].drop(title).nlargest(5)
    return [{"Title": m, "Score": s, **meta.get(m, {})} for m, s in scores.items()]

# -----------------------------------------------------------------------------
# 3. KART GÖRÜNÜMÜ
# -----------------------------------------------------------------------------
def display_cards(movies):
    cols = st.columns(5)
    for i, m in enumerate(movies):
        with cols[i]:
            rating_val = m.get('RATING', 0)
            
            # Kart Yapısı (HTML)
            st.markdown(f"""
            <div class="movie-card">
                <div class="match-badge">%{int(m['Score']*100)} UYUM</div>
                <div class="card-icon">🎬</div>
                <div class="card-title" title="{m['Title']}">{m['Title']}</div>
                <div class="card-meta">
                    <span>{int(m['YEAR']) if 'YEAR' in m else '-'}</span>
                    <span class="star-rating">★ {rating_val:.1f}</span>
                </div>
                <div class="card-genre">
                    {m.get('GENRES', '').replace('|', ', ')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Aksiyon Butonu
            if st.button(f"İNCELE & BENZERLERİ", key=f"btn_{i}", use_container_width=True):
                st.session_state.selected_movie_final = m['Title']
                st.session_state.candidates = []
                st.rerun()

# -----------------------------------------------------------------------------
# 4. ANA AKIŞ
# -----------------------------------------------------------------------------
def main():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'selected_movie_final' not in st.session_state:
        st.session_state.selected_movie_final = None
    if 'candidates' not in st.session_state:
        st.session_state.candidates = []

    # HERO ALANI
    st.markdown("""
        <div class="hero-container">
            <h1 class="main-title">MovieMind AI</h1>
            <p class="subtitle">Size Özel Sinema Film Tavsiye Platformu</p>
        </div>
    """, unsafe_allow_html=True)

    # DATA LOADING
    if not st.session_state.data_loaded:
        with st.spinner('Sinema arşivi yükleniyor...'):
            path = download_data_from_drive("1gl_iJXRyEaSzhHlgfBUdTzQZMer4gdsS")
            if path:
                sim, titles, meta = prepare_data(path)
                if sim is not None:
                    st.session_state.sim = sim
                    st.session_state.titles = titles
                    st.session_state.meta = meta
                    st.session_state.data_loaded = True
                    st.rerun()

    if not st.session_state.data_loaded: st.stop()

    # ARAMA MOTORU
    c1, c2, c3 = st.columns([1, 6, 1])
    with c2:
        query = st.text_input("ara", placeholder="Hangi filmi beğendiniz? (Örn: The Dark Knight)", label_visibility="collapsed")
        if query and ('last_q' not in st.session_state or st.session_state.last_q != query):
            st.session_state.candidates = find_candidates(query, st.session_state.titles)
            st.session_state.last_q = query
            st.session_state.selected_movie_final = None

    # ADAYLAR
    if st.session_state.candidates and not st.session_state.selected_movie_final:
        st.markdown("<div style='text-align:center; margin-top:20px; color:#999; font-size:0.9rem;'>BUNU MU DEMEK İSTEDİNİZ?</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(len(st.session_state.candidates))
        for i, cand in enumerate(st.session_state.candidates):
            if cols[i].button(cand, key=f"cand_{i}", use_container_width=True):
                st.session_state.selected_movie_final = cand
                st.session_state.candidates = []
                st.rerun()

    # SONUÇ EKRANI
    if st.session_state.selected_movie_final:
        sel_movie = st.session_state.selected_movie_final
        info = st.session_state.meta.get(sel_movie, {})
        current_rating = info.get('RATING', 0)
        
        # SEÇİLEN FİLM PANELİ
        st.markdown(f"""
        <div class="selected-movie-info">
            <div class="info-label">ŞU AN İNCELENEN YAPIM</div>
            <div class="info-title">{sel_movie}</div>
            <div class="info-meta">
                <span class="highlight-box">{int(info.get('YEAR', 0))}</span>
                <span>{info.get('GENRES', '').replace('|', ' • ')}</span>
                <span class="highlight-box">IMDb: {current_rating:.1f}</span>
            </div>
            <div style="color:#aaa; font-size:0.9rem; max-width:700px; margin:0 auto;">
                Yapay zeka algoritmamız, <b>{sel_movie}</b> filminin genetik kodlarını analiz etti. 
                Senaryo yapısı, tür özellikleri ve izleyici davranışlarına göre sizin için en iyi 5 alternatifi belirledi.
            </div>
        </div>
        """, unsafe_allow_html=True)

        recs = get_recs(st.session_state.sim, st.session_state.meta, sel_movie)
        if recs:
            display_cards(recs)
        else:
            st.warning("Veri bulunamadı.")

if __name__ == "__main__":
    main()
