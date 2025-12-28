import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
import gdown
import os
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI VE KOMPAKT PREMIUM CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MovieMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Cinzel:wght@700&display=swap');

    /* --- GENEL TEMALAR --- */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 10%, #1e1e24 0%, #0b0b0f 100%);
        background-attachment: fixed;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* YAZI TİPLERİ VE RENKLERİ */
    .stApp, p, span, div, label {
        color: #b0b0b0 !important; /* Gümüş Grisi */
        font-family: 'Poppins', sans-serif !important;
        font-size: 0.9rem !important; 
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Cinzel', serif !important;
        color: #d4af37 !important; /* Metalik Altın */
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* HERO SECTION */
    .hero-container {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(212, 175, 55, 0.1);
    }
    
    .main-title {
        font-size: 3rem !important;
        background: linear-gradient(to bottom, #d4af37, #aa8c2c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 4px 10px rgba(0,0,0,0.5);
        margin-bottom: 5px;
    }
    
    .subtitle {
        color: #888 !important;
        font-size: 0.9rem !important;
        letter-spacing: 1px;
        font-weight: 300 !important;
    }

    /* SEÇİLEN FİLM BİLGİ PANELİ (KOMPAKT) */
    .selected-movie-info {
        background: linear-gradient(90deg, rgba(20,20,30,0.9), rgba(40,40,50,0.9));
        border-left: 4px solid #d4af37;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 25px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.4);
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    
    .info-title {
        font-size: 1.6rem !important;
        color: #f0f0f0 !important;
        margin-bottom: 8px;
        line-height: 1.2;
    }
    
    .info-meta {
        display: flex;
        gap: 15px;
        justify-content: center;
        font-size: 0.9rem;
        color: #d4af37 !important;
    }

    /* ÖNERİ KARTLARI */
    div.movie-card {
        background: rgba(30, 30, 40, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0 0;
        padding: 12px; 
        height: 180px;
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    div.movie-card:hover {
        background: rgba(40, 40, 55, 0.8);
        border-color: #d4af37;
        transform: translateY(-3px);
    }

    .card-title {
        color: #e0e0e0 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        margin-bottom: 5px;
        height: 2.8em;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        line-height: 1.3;
        font-family: 'Poppins', sans-serif !important;
        text-transform: none;
    }

    .card-meta {
        font-size: 0.75rem !important;
        color: #888 !important;
        display: flex;
        justify-content: space-between;
        margin-bottom: 3px;
    }

    .card-genre {
        font-size: 0.7rem !important;
        color: #666 !important;
        font-style: italic;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .match-badge {
        font-size: 0.7rem;
        color: #d4af37 !important;
        font-weight: bold;
        margin-top: 5px;
        text-align: right;
    }

    /* ETKİLEŞİM BUTONLARI */
    div[data-testid="column"] button {
        background: rgba(212, 175, 55, 0.1) !important;
        border: 1px solid rgba(212, 175, 55, 0.2) !important;
        color: #d4af37 !important;
        border-radius: 0 0 10px 10px !important;
        font-size: 0.75rem !important;
        padding: 4px 0px !important;
        margin-top: -8px !important;
        transition: 0.3s !important;
        width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    div[data-testid="column"] button:hover {
        background: #d4af37 !important;
        color: #000 !important;
    }

    /* INPUT ALANI */
    .stTextInput input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid #444 !important;
        color: #d4af37 !important;
        border-radius: 8px !important;
        padding: 10px 15px !important;
        font-size: 0.95rem !important;
    }

    /* ADAY FİLM BUTONLARI */
    .element-container button {
        height: auto !important;
        min-height: 50px;
        white-space: normal !important;
        word-wrap: break-word !important;
        padding: 5px 10px !important;
        line-height: 1.2 !important;
        font-size: 0.8rem !important;
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
        
        # Sözlüğe RATING anahtarı ile 10'luk puanı atıyoruz
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
            
            st.markdown(f"""
            <div class="movie-card">
                <div style="text-align:center; font-size:1.5rem; margin-bottom:5px;">🎥</div>
                <div class="card-title" title="{m['Title']}">{m['Title']}</div>
                <div class="card-meta">
                    <span>{int(m['YEAR']) if 'YEAR' in m else '-'}</span>
                    <span style="color:#d4af37; font-weight:bold;">★ {rating_val:.1f}</span>
                </div>
                <div class="card-genre">
                    {m.get('GENRES', '').replace('|', ', ')}
                </div>
                <div class="match-badge">
                    %{int(m['Score']*100)} UYUM
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Analiz Et ⚡", key=f"btn_{i}", use_container_width=True):
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

    # Hero
    st.markdown("""
        <div class="hero-container">
            <h1 class="main-title">MovieMind AI</h1>
            <p class="subtitle">Size Özel Sinema Film Tavsiye Platformu</p>
        </div>
    """, unsafe_allow_html=True)

    # Load Data
    if not st.session_state.data_loaded:
        with st.spinner('Veritabanı işleniyor...'):
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

    # Arama
    c1, c2, c3 = st.columns([1, 6, 1])
    with c2:
        query = st.text_input("ara", placeholder="Film ara... (örn: Godfather, Matrix)", label_visibility="collapsed")
        if query and ('last_q' not in st.session_state or st.session_state.last_q != query):
            st.session_state.candidates = find_candidates(query, st.session_state.titles)
            st.session_state.last_q = query
            st.session_state.selected_movie_final = None

    # Adaylar
    if st.session_state.candidates and not st.session_state.selected_movie_final:
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
        
        st.markdown(f"""
        <div class="selected-movie-info">
            <div style="color:#d4af37; font-size:0.75rem; letter-spacing:2px; margin-bottom:5px;">ANALİZ EDİLEN FİLM</div>
            <div class="info-title">{sel_movie}</div>
            <div class="info-meta">
                <span>📅 {int(info.get('YEAR', 0))}</span>
                <span>•</span>
                <span>🎭 {info.get('GENRES', '').replace('|', ', ')}</span>
                <span>•</span>
                <span>⭐ {current_rating:.1f} / 10</span>
            </div>
            <div style="margin-top:15px; color:#888; font-size:0.8rem; max-width:600px;">
                Yapay zeka bu filmin özniteliklerini taradı ve en uygun eşleşmeleri aşağıda listeledi.
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
