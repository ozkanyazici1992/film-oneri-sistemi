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

    /* HERO SECTION - SİNEMATİK */
    .hero-container {
        text-align: center;
        padding: 3rem 0 4rem 0;
        background: radial-gradient(ellipse at center, rgba(218,165,32,0.15) 0%, transparent 70%);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(218,165,32,0.03) 2px,
            rgba(218,165,32,0.03) 4px
        );
        animation: scan 8s linear infinite;
        pointer-events: none;
    }
    
    @keyframes scan {
        0% { transform: translateY(0); }
        100% { transform: translateY(50px); }
    }
    
    .main-title {
        font-size: 4.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #DAA520 0%, #FFD700 50%, #FFA500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 40px rgba(218,165,32,0.5);
        letter-spacing: 2px;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .subtitle {
        font-size: 1.2rem !important;
        color: #c0c0c0 !important;
        font-weight: 300 !important;
        letter-spacing: 1px;
        position: relative;
        z-index: 1;
    }

    /* FİLM KARTLARI - PREMİUM */
    div.movie-card {
        background: linear-gradient(145deg, rgba(26,26,46,0.8), rgba(22,33,62,0.6));
        border: 1px solid rgba(218,165,32,0.2);
        border-radius: 20px;
        padding: 25px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    
    div.movie-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #DAA520, #FFD700, #DAA520);
        opacity: 0;
        transition: opacity 0.4s;
    }

    div.movie-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(218,165,32,0.3), 0 0 0 1px rgba(218,165,32,0.5);
        border-color: #DAA520;
    }
    
    div.movie-card:hover::before {
        opacity: 1;
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
        line-height: 1.4;
    }

    .card-metric {
        font-size: 0.95rem;
        color: #b8b8b8 !important;
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        align-items: center;
    }

    .score-badge {
        background: linear-gradient(135deg, #DAA520, #FFD700);
        color: #000000 !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        box-shadow: 0 2px 8px rgba(218,165,32,0.4);
    }

    /* INPUT ALANLARI - MODERN */
    .stTextInput > div > div > input {
        background: rgba(40,40,60,0.9) !important;
        color: #FFFFFF !important;
        border: 2px solid rgba(218,165,32,0.4) !important;
        border-radius: 30px;
        padding: 14px 24px;
        font-size: 1rem;
        transition: all 0.3s;
        font-weight: 500 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #DAA520 !important;
        box-shadow: 0 0 0 3px rgba(218,165,32,0.3) !important;
        background: rgba(50,50,70,1) !important;
    }
    
    .stTextInput > label {
        display: none !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #b8b8b8 !important;
        font-weight: 400 !important;
    }

    /* BUTONLAR - GOLD THEME */
    .stButton > button {
        background: linear-gradient(135deg, #DAA520 0%, #FFD700 100%) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.05rem;
        width: 100%;
        padding: 14px 28px;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(218,165,32,0.4);
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(218,165,32,0.6);
        background: linear-gradient(135deg, #FFD700 0%, #DAA520 100%) !important;
    }

    /* SELECTBOX - PREMIUM */
    .stSelectbox > div > div {
        background: rgba(40,40,60,0.95) !important;
        border: 2px solid rgba(218,165,32,0.4) !important;
        border-radius: 20px;
        padding: 10px 16px;
        transition: all 0.3s;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #DAA520 !important;
        background: rgba(50,50,70,1) !important;
    }
    
    .stSelectbox > div > div > div {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Seçili değer */
    .stSelectbox [data-baseweb="select"] > div {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    .stSelectbox label {
        color: #DAA520 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 10px;
    }
    
    /* Dropdown menü kutusu */
    [data-baseweb="popover"] {
        background: rgba(30,30,50,0.98) !important;
    }
    
    /* Dropdown menü içeriği */
    [data-baseweb="menu"] {
        background: rgba(30,30,50,0.98) !important;
    }
    
    [data-baseweb="menu"] > ul {
        background: rgba(30,30,50,0.98) !important;
    }
    
    /* Seçenekler */
    [role="option"] {
        background: rgba(30,30,50,0.98) !important;
        color: #FFFFFF !important;
        font-weight: 500 !important;
        padding: 12px 16px !important;
    }
    
    [role="option"]:hover {
        background: rgba(218,165,32,0.4) !important;
        color: #FFD700 !important;
    }
    
    [aria-selected="true"][role="option"] {
        background: rgba(218,165,32,0.3) !important;
        color: #FFD700 !important;
    }

    /* TABS - SİNEMATİK */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 2px solid rgba(218,165,32,0.2);
        padding-bottom: 0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 55px;
        border-radius: 15px 15px 0 0;
        background: rgba(26,26,46,0.4);
        color: #a0a0a0 !important;
        border: none;
        border-bottom: 3px solid transparent;
        padding: 0 28px;
        font-size: 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(218,165,32,0.15) !important;
        color: #FFD700 !important;
        border-bottom-color: #DAA520 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(218,165,32,0.1);
        color: #FFD700 !important;
    }

    /* SIDEBAR - DARK GOLD */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a2e 100%) !important;
        border-right: 1px solid rgba(218,165,32,0.2);
    }
    
    /* METRİKLER */
    [data-testid="stMetricValue"] {
        color: #FFD700 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b8b8b8 !important;
    }
    
    /* SCROLLBAR */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(26,26,46,0.3);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #DAA520, #FFD700);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #FFD700, #DAA520);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HIZ OPTİMİZASYONU - VERİ İŞLEME
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
        # Hızlı okuma
        df = pd.read_csv(filepath, low_memory=False)
        
        # Vektörize işlemler
        df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")
        df["TIME"] = pd.to_datetime(df["TIME"], dayfirst=True, errors='coerce')
        df.dropna(subset=["TITLE", "YEAR", "TIME", "RATING"], inplace=True)
        df["YEAR"] = df["YEAR"].astype(int)
        df["RATING_10"] = df["RATING"] * 2
        
        # Grup işlemleri
        vote_counts = df.groupby("TITLE", sort=False)["RATING"].count()
        df["NUM_VOTES"] = df["TITLE"].map(vote_counts)
        mean_rating = df["RATING_10"].mean()
        
        # İstatistikler - Metadata için
        movie_stats = df.groupby("TITLE", sort=False).agg({
            "RATING_10": "mean",
            "NUM_VOTES": "max",
            "YEAR": "first",
            "GENRES": "first"
        }).reset_index()
        
        # Vektörize weighted rating
        movie_stats["IMDB_SCORE"] = (
            (movie_stats["NUM_VOTES"] / (movie_stats["NUM_VOTES"] + min_votes)) * movie_stats["RATING_10"] +
            (min_votes / (movie_stats["NUM_VOTES"] + min_votes)) * mean_rating
        )
        
        df["IMDB_SCORE"] = df["TITLE"].map(movie_stats.set_index("TITLE")["IMDB_SCORE"])
        
        # --- OPTİMİZASYON: HIZLI ERİŞİM İÇİN SÖZLÜK ---
        movie_metadata = movie_stats.set_index("TITLE")[["IMDB_SCORE", "YEAR", "GENRES"]].to_dict('index')

        # Filtreleme
        popular_titles = vote_counts[vote_counts >= vote_threshold].index
        df_filtered = df[df["TITLE"].isin(popular_titles)].copy()
        
        # Similarity matrix
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
        st.error(f"❌ Veri işleme hatası: {str(e)}")
        return None, None, None, None, None

def find_movie_candidates(query, _normalized_titles_dict, top_n=5):
    """Kullanıcı girdisine en yakın 5 filmi bulur ve listeler."""
    normalized_input = normalize_title(query)
    # Sadece fuzzy match yap, veriyi çekme
    close_matches = difflib.get_close_matches(normalized_input, _normalized_titles_dict.keys(), n=top_n, cutoff=0.3)
    # Normalize edilmiş key'lerden gerçek film isimlerini bul
    real_titles = [_normalized_titles_dict[m] for m in close_matches]
    return real_titles

@st.cache_data
def get_recommendations_for_selected(_similarity_df, _movie_metadata, selected_movie, top_n):
    """Kullanıcının seçtiği KESİN film ismine göre öneri getirir."""
    
    # Seçilen film veritabanında var mı? (Garantilemek için)
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
# 3. GÖRSEL KARTLAR
# -----------------------------------------------------------------------------

def display_movie_cards(movies_data, col_count=5):
    """Premium film kartları - Standart 5 kolon"""
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
                {'<div style="margin-top:12px; font-size:0.85rem; color:#DAA520; font-weight:600;">Eşleşme: %' + str(int(movie["Benzerlik"]*100)) + '</div>' if "Benzerlik" in movie else ''}
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. ANA UYGULAMA
# -----------------------------------------------------------------------------

def main():
    # Session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.df = None
        st.session_state.df_filtered = None
        st.session_state.movie_similarity_df = None
        st.session_state.normalized_titles_dict = None
        st.session_state.movie_metadata = None

    # Arama state yönetimi (YENİLENDİ)
    if 'search_candidates' not in st.session_state:
        st.session_state.search_candidates = [] # Bulunan potansiyel filmler
    
    if 'search_term' not in st.session_state:
        st.session_state.search_term = ""

    if 'results_ready' not in st.session_state:
        st.session_state.results_ready = False

    # Hero Section
    st.markdown("""
        <div class="hero-container">
            <h1 class="main-title">CineAI</h1>
            <p class="subtitle">Yapay Zeka Destekli Premium Film Keşif Platformu</p>
        </div>
    """, unsafe_allow_html=True)

    # Veri yükleme
    if not st.session_state.data_loaded:
        FILE_ID = "1gl_iJXRyEaSzhHlgfBUdTzQZMer4gdsS"
        filepath = download_data_from_drive(FILE_ID)
        
        if filepath:
            with st.spinner('🎬 CineAI motoru başlatılıyor...'):
                result = prepare_data(filepath)
                
                if result[0] is not None:
                    st.session_state.df = result[0]
                    st.session_state.df_filtered = result[1]
                    st.session_state.movie_similarity_df = result[2]
                    st.session_state.normalized_titles_dict = result[3]
                    st.session_state.movie_metadata = result[4]
                    st.session_state.data_loaded = True
                    st.rerun()
                else:
                    st.stop()
        else:
            st.stop()
    
    # Veriler
    df = st.session_state.df
    df_filtered = st.session_state.df_filtered
    movie_similarity_df = st.session_state.movie_similarity_df
    normalized_titles_dict = st.session_state.normalized_titles_dict
    movie_metadata = st.session_state.movie_metadata

    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Veritabanı İstatistikleri")
        st.markdown("---")
        
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("🎬 Film", f"{df['TITLE'].nunique()//1000}K+")
        col_s2.metric("👥 Kullanıcı", f"{df['USERID'].nunique()//1000}K+")
        
        st.markdown("---")
        st.markdown("### 📈 Yıllık Trend")
        year_counts = df.groupby('YEAR', sort=False)['TITLE'].nunique().reset_index()
        fig_mini = px.area(year_counts, x='YEAR', y='TITLE')
        fig_mini.update_layout(
            height=140,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            showlegend=False
        )
        fig_mini.update_traces(line_color='#DAA520', fillcolor='rgba(218,165,32,0.3)')
        st.plotly_chart(fig_mini, width='stretch', key="sidebar_chart")

    # Tabs
    tab1, tab2 = st.tabs([
        "🔍 Film Önerisi",
        "📊 Veri Analizi"
    ])

    # TAB 1: ÖNERİ (SADELEŞTİRİLMİŞ AKIŞ)
    with tab1:
        st.markdown("### 🎬 Film Arayın")
        
        # 1. Adım: Arama Çubuğu
        col_search_inp, col_search_btn = st.columns([4, 1])
        
        with col_search_inp:
            query = st.text_input("film_search", 
                                placeholder="🔍 Film adı yazın... (örn: lor of the ring, batmn)",
                                label_visibility="collapsed", 
                                key="query_input")
        
        with col_search_btn:
            search_clicked = st.button("🔎 Ara", type="primary")

        # Butona basılınca adayları bul
        if search_clicked and query:
            candidates = find_movie_candidates(query, normalized_titles_dict, top_n=5)
            st.session_state.search_candidates = candidates
            st.session_state.search_term = query
            st.session_state.results_ready = False # Yeni arama yapıldı, sonuçlar sıfırlandı

        # 2. Adım: Aday Listesi ve Seçim (Eğer aday varsa göster)
        if st.session_state.search_candidates:
            st.markdown("---")
            st.info("🤔 **Bunu mu demek istediniz?** Lütfen aşağıdaki listeden doğru filmi seçin:")
            
            selected_candidate = st.selectbox(
                "Bulunan Filmler", 
                st.session_state.search_candidates,
                label_visibility="collapsed"
            )
            
            # 3. Adım: Önerileri Getir Butonu
            if st.button(f"🎯 '{selected_candidate}' için Önerileri Getir", type="primary"):
                with st.spinner('🎬 Analiz yapılıyor...'):
                    recommendations = get_recommendations_for_selected(
                        movie_similarity_df,
                        movie_metadata,
                        selected_candidate,
                        5 # Sabit 5 öneri
                    )
                    st.session_state.recommendations = recommendations
                    st.session_state.results_ready = True
                    st.session_state.selected_movie_name = selected_candidate

        # 4. Adım: Sonuç Ekranı
        if st.session_state.get('results_ready') and st.session_state.get('recommendations'):
            st.markdown("---")
            st.success(f"✨ **{st.session_state.selected_movie_name}** filmine benzer 5 önerimiz:")
            display_movie_cards(st.session_state.recommendations, col_count=5)
            
        elif st.session_state.get('results_ready') and not st.session_state.get('recommendations'):
             st.error("⚠️ Bir hata oluştu veya veri bulunamadı.")
             
        elif search_clicked and not st.session_state.search_candidates:
            st.warning("😔 Aradığınız isme yakın bir film bulamadık. Lütfen tekrar deneyin.")

    # TAB 2: ANALİZ
    with tab2:
        st.markdown("### 📊 Veri Seti İçgörüleri")
        
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            st.markdown("**🎬 En Popüler Türler**")
            genres_count = df['GENRES'].str.get_dummies(sep='|').sum().sort_values(ascending=True).tail(10)
            fig_bar = px.bar(x=genres_count.values, y=genres_count.index, orientation='h')
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title="Film Sayısı",
                yaxis_title=None,
                height=400
            )
            fig_bar.update_traces(marker_color='#DAA520')
            st.plotly_chart(fig_bar, width='stretch', key="genre_bar")
            
        with col_a2:
            st.markdown("**⭐ IMDb Puan Dağılımı**")
            fig_hist = px.histogram(df_filtered, x='IMDB_SCORE', nbins=25)
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title="IMDb Puanı",
                yaxis_title="Film Sayısı",
                bargap=0.1,
                height=400
            )
            fig_hist.update_traces(marker_color='#FFD700')
            st.plotly_chart(fig_hist, width='stretch', key="score_hist")

if __name__ == "__main__":
    main()
