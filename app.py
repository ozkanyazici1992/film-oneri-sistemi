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
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&family=Bebas+Neue&display=swap');

    /* --- ARKA PLAN --- */
    [data-testid="stAppViewContainer"] {
        background-color: #000000;
        background-image: radial-gradient(circle at 50% 0%, #1a0505 0%, #000000 85%);
        background-attachment: fixed;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* --- TEMEL YAZI AYARLARI --- */
    .stApp, p, span, div, label {
        color: #ffffff !important;
        font-family: 'Montserrat', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        line-height: 1.5 !important;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Bebas Neue', cursive !important;
        color: #FFD700 !important;
        letter-spacing: 2px;
        text-transform: uppercase;
        text-shadow: 2px 2px 4px #000000;
    }

    /* HERO SECTION */
    .hero-container {
        text-align: center;
        padding: 2rem 0 1.5rem 0; /* Padding azaltıldı */
        border-bottom: 1px solid linear-gradient(90deg, transparent, #E50914, transparent);
    }
    
    .main-title {
        font-size: 4rem !important; /* Biraz küçüldü */
        background: linear-gradient(to bottom, #FFD700 0%, #FDB931 50%, #C6930A 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 20px rgba(253, 185, 49, 0.6));
        margin-bottom: 5px;
        font-weight: 900 !important;
    }
    
    .subtitle {
        color: #f0f0f0 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 3px;
        text-transform: uppercase;
        opacity: 0.9;
    }

    /* --- SEÇİLEN FİLM PANELİ (KOMPAKT VERSİYON) --- */
    .selected-movie-info {
        background: rgba(15, 15, 15, 0.95);
        border: 1px solid #333;
        border-top: 3px solid #E50914; /* Çizgi inceldi */
        padding: 15px 20px; /* Boşluklar ciddi oranda azaltıldı */
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.8);
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .info-label {
        color: #E50914 !important;
        font-weight: 700;
        font-size: 0.75rem; /* Küçüldü */
        letter-spacing: 2px;
        margin-bottom: 2px;
    }
    
    .info-title {
        font-family: 'Bebas Neue', cursive !important;
        font-size: 2.2rem !important; /* 3rem'den 2.2rem'e düştü */
        color: #ffffff !important;
        margin-bottom: 8px;
        text-shadow: 2px 2px 5px #000;
        letter-spacing: 1px;
        line-height: 1;
    }
    
    .info-meta {
        display: flex;
        gap: 15px;
        justify-content: center;
        align-items: center;
        font-size: 0.9rem; /* Küçüldü */
        color: #ffffff !important;
        margin-bottom: 10px;
        font-weight: 600;
    }
    
    .highlight-box {
        background: rgba(255, 215, 0, 0.1);
        border: 1px solid rgba(255, 215, 0, 0.5);
        color: #FFD700 !important;
        padding: 2px 12px; /* Padding küçüldü */
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
    }

    .ai-description {
        color: #cccccc !important;
        font-size: 0.9rem !important; /* Küçüldü */
        max-width: 750px;
        margin: 0 auto;
        line-height: 1.4;
        font-weight: 400 !important;
        font-style: italic;
    }

    /* KART TASARIMI */
    div.movie-card {
        background: #111111;
        border: 1px solid #444;
        border-radius: 10px 10px 0 0;
        padding: 16px; 
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }

    div.movie-card:hover {
        transform: translateY(-5px);
        border-color: #E50914;
        background: #161616;
        box-shadow: 0 10px 30px rgba(229, 9, 20, 0.3);
    }

    .card-icon {
        font-size: 2rem;
        text-align: center;
        margin-bottom: 5px;
        filter: drop-shadow(0 0 5px rgba(255,255,255,0.5));
    }

    .card-title {
        color: #ffffff !important;
        font-family: 'Montserrat', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 800 !important;
        margin-bottom: 5px;
        height: 2.8em;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        line-height: 1.3;
        text-transform: capitalize;
        text-shadow: 1px 1px 2px #000;
    }

    .card-meta {
        font-size: 0.9rem !important;
        color: #dddddd !important;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 600;
    }

    .star-rating {
        color: #FFD700 !important;
        font-weight: 900;
        text-shadow: 0 0 5px rgba(255, 215, 0, 0.5);
    }

    .card-genre {
        font-size: 0.75rem !important;
        color: #cccccc !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-style: normal;
        font-weight: 500;
        margin-top: 5px;
    }

    .match-badge {
        position: absolute;
        top: 10px;
        right: 10px;
        background: #E50914;
        color: white !important;
        font-size: 0.75rem;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 800;
        box-shadow: 0 2px 5px rgba(0,0,0,0.8);
    }

    /* BUTONLAR */
    div[data-testid="column"] button {
        background: #1a1a1a !important;
        border: 1px solid #555 !important;
        color: #ffffff !important;
        border-radius: 0 0 10px 10px !important;
        font-size: 0.85rem !important;
        font-weight: 800 !important;
        padding: 10px 0px !important;
        margin-top: -8px !important;
        transition: 0.2s !important;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div[data-testid="column"] button:hover {
        background: #E50914 !important;
        color: #ffffff !important;
        border-color: #E50914 !important;
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.6);
    }

    /* INPUT ALANI */
    .stTextInput input {
        background-color: #222 !important;
        border: 2px solid #555 !important;
        color: #ffffff !important;
        border-radius: 50px !important;
        padding: 15px 25px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    .stTextInput input:focus {
        border-color: #FFD700 !important;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.4) !important;
        background-color: #000 !important;
    }
    
    .stTextInput input::placeholder {
        color: #888 !important;
        font-weight: 500 !important;
    }

    /* ADAY BUTONLARI */
    .element-container button {
        background-color: rgba(255,255,255,0.05) !important;
        border: 1px solid #666 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    .element-container button:hover {
        border-color: #FFD700 !important;
        color: #FFD700 !important;
        background-color: rgba(255, 215, 0, 0.1) !important;
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
    raw_scores = sim_df[title].drop(title).nlargest(5)
    
    min_display = 0.75
    max_display = 0.98
    
    if not raw_scores.empty:
        min_raw = raw_scores.min()
        max_raw = raw_scores.max()
        if max_raw == min_raw:
            calibrated_scores = {k: 0.95 for k, v in raw_scores.items()}
        else:
            calibrated_scores = {}
            for m, s in raw_scores.items():
                normalized = (s - min_raw) / (max_raw - min_raw)
                final_score = normalized * (max_display - min_display) + min_display
                calibrated_scores[m] = final_score
    else:
        calibrated_scores = {}

    return [{"Title": m, "Score": calibrated_scores[m], **meta.get(m, {})} for m in raw_scores.index]

# -----------------------------------------------------------------------------
# 3. KART GÖRÜNÜMÜ
# -----------------------------------------------------------------------------
def display_cards(movies):
    cols = st.columns(5)
    for i, m in enumerate(movies):
        with cols[i]:
            rating_val = m.get('RATING', 0)
            
            st.markdown(f"""
            <div class="movie-card">
                <div class="match-badge">%{int(m['Score']*100)} UYUM</div>
                <div class="card-icon">🎬</div>
                <div class="card-title" title="{m['Title']}">{m['Title']}</div>
                <div class="card-meta">
                    <span style="color:#aaa;">{int(m['YEAR']) if 'YEAR' in m else '-'}</span>
                    <span class="star-rating">★ {rating_val:.1f}</span>
                </div>
                <div class="card-genre">
                    {m.get('GENRES', '').replace('|', ', ')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
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
        query = st.text_input("ara", placeholder="Etkisinden çıkamadığınız o filmi yazın... (Örn: Inception, The Prestige)", label_visibility="collapsed")
        
        if query and ('last_q' not in st.session_state or st.session_state.last_q != query):
            st.session_state.candidates = find_candidates(query, st.session_state.titles)
            st.session_state.last_q = query
            st.session_state.selected_movie_final = None

    # ADAYLAR
    if st.session_state.candidates and not st.session_state.selected_movie_final:
        st.markdown("<div style='text-align:center; margin-top:20px; color:#ffffff; font-size:1rem; font-weight:600;'>BUNU MU DEMEK İSTEDİNİZ?</div>", unsafe_allow_html=True)
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
        
        # KOMPAKT SEÇİLEN FİLM PANELİ
        st.markdown(f"""
        <div class="selected-movie-info">
            <div class="info-label">ŞU AN İNCELENEN YAPIM</div>
            <div class="info-title">{sel_movie}</div>
            <div class="info-meta">
                <span class="highlight-box">{int(info.get('YEAR', 0))}</span>
                <span>{info.get('GENRES', '').replace('|', ' • ')}</span>
                <span class="highlight-box">IMDb: {current_rating:.1f}</span>
            </div>
            <div class="ai-description">
                Yapay zeka algoritmamız, <b style="color:#FFD700;">{sel_movie}</b> filminin genetik kodlarını analiz etti. 
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
