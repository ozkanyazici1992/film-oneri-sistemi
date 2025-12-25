import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
import gdown
import os
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# -----------------------------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI VE CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CineAI | Film Keşif Asistanı",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ZORUNLU KOYU TEMA VE HIZLI YÜKLEME İÇİN CSS
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* EN ÖNEMLİ KISIM: Arka Planı Zorla Siyah Yap */
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117 !important;
        background-image: radial-gradient(circle at top left, #1a2a3a, #000000) !important;
    }
    
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }

    /* Genel Yazı Rengi */
    .stApp, p, h1, h2, h3, h4, h5, h6, span, div {
        color: #e0e0e0 !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* Başlık Alanı (Hero Section) */
    .hero-container {
        text-align: center;
        padding: 2rem 0 3rem 0;
        background: linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,255,255,0.05) 50%, rgba(0,0,0,0) 100%);
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #40E0D0, #00CED1, #FFFFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 20px rgba(64, 224, 208, 0.3);
        margin-bottom: 0.5rem;
    }

    /* Kart Tasarımı */
    div.movie-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        backdrop-filter: blur(10px);
    }

    div.movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 206, 209, 0.2);
        border-color: #40E0D0;
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

    .card-metric {
        font-size: 0.9rem;
        color: #cccccc !important;
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
    }

    .score-badge {
        background-color: #40E0D0;
        color: #000 !important;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.8rem;
    }

    /* Input Alanları */
    .stTextInput > div > div > input {
        background-color: rgba(255,255,255,0.05) !important;
        color: white !important;
        border: 1px solid #40E0D0 !important;
        border-radius: 25px;
        padding: 10px 20px;
    }
    
    /* Input placeholder ve label gizleme */
    .stTextInput > label {
        display: none !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #888888 !important;
    }

    /* Butonlar */
    .stButton > button {
        background: linear-gradient(45deg, #40E0D0, #008B8B) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 0 15px rgba(64, 224, 208, 0.5);
    }

    /* Tablar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 25px;
        background-color: rgba(255,255,255,0.05);
        color: #888888 !important;
        border: 1px solid rgba(255,255,255,0.1);
        padding: 0 20px;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #40E0D0 !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border: 2px solid #40E0D0;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(64, 224, 208, 0.1);
        color: #40E0D0 !important;
        border-color: #40E0D0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Metrik Rengi */
    [data-testid="stMetricValue"] {
        color: #40E0D0 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. VERİ YÜKLEME VE İŞLEME (HIZ İÇİN OPTİMİZE EDİLDİ)
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

@st.cache_resource(ttl=3600, show_spinner="Veriler işleniyor...")
def prepare_data(filepath, vote_threshold=1000, min_votes=2500):
    try:
        df = pd.read_csv(filepath)
        
        # Veri Temizleme
        df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")
        df["TIME"] = pd.to_datetime(df["TIME"], dayfirst=True, errors='coerce')
        df.dropna(subset=["TITLE", "YEAR", "TIME", "RATING"], inplace=True)
        df["YEAR"] = df["YEAR"].astype(int)
        df["RATING_10"] = df["RATING"] * 2
        
        # İstatistikler
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
        
        # Filtreleme
        popular_titles = vote_counts[vote_counts >= vote_threshold].index
        df_filtered = df[df["TITLE"].isin(popular_titles)].copy()
        
        # Matrisler
        user_movie_matrix = df_filtered.pivot_table(
            index="USERID", columns="TITLE", values="RATING_10", aggfunc='mean'
        ).fillna(0)
        
        movie_similarity_df = pd.DataFrame(
            cosine_similarity(user_movie_matrix.T),
            index=user_movie_matrix.columns,
            columns=user_movie_matrix.columns
        )
        
        normalized_titles_dict = {normalize_title(t): t for t in movie_similarity_df.columns}
        
        # ÖN HESAPLAMA: Yıl ve tür bazlı en iyi filmleri cache'le
        year_best = {}
        for year in df_filtered['YEAR'].unique():
            year_data = df_filtered[df_filtered['YEAR'] == year].groupby(['TITLE', 'GENRES'])['IMDB_SCORE'].mean().reset_index()
            year_best[year] = year_data.sort_values('IMDB_SCORE', ascending=False).head(8)
        
        # Tür listesi ve en iyi filmleri
        all_genres = sorted(list(set([g for sublist in df['GENRES'].dropna().str.split('|') for g in sublist])))
        genre_best = {}
        for genre in all_genres:
            genre_data = df_filtered[df_filtered["GENRES"].str.contains(genre, na=False)]
            genre_top = genre_data.groupby(['TITLE', 'YEAR'])['IMDB_SCORE'].mean().reset_index()
            genre_best[genre] = genre_top.sort_values('IMDB_SCORE', ascending=False).head(8)
        
        return df, df_filtered, user_movie_matrix, movie_similarity_df, normalized_titles_dict, year_best, genre_best, all_genres
        
    except Exception as e:
        st.error(f"❌ Veri işleme hatası: {str(e)}")
        return None, None, None, None, None, None, None, None

def recommend_by_title(title, similarity_df, df, top_n=5, normalized_titles_dict=None):
    normalized_input = normalize_title(title)
    close_matches = difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=1)
    
    if not close_matches:
        alternatives = [normalized_titles_dict[t] for t in difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=3)]
        return None, alternatives
    
    match = normalized_titles_dict[close_matches[0]]
    scores = similarity_df[match].drop(labels=[match], errors="ignore")
    recommendations = scores.sort_values(ascending=False).head(top_n)
    
    rec_data = []
    for movie, similarity_score in recommendations.items():
        movie_info = df[df["TITLE"] == movie].iloc[0]
        rec_data.append({
            "Film": movie,
            "Benzerlik": float(similarity_score),
            "IMDb": float(movie_info['IMDB_SCORE']),
            "Yıl": int(movie_info["YEAR"]),
            "Türler": movie_info["GENRES"].replace("|", ", ")
        })
    
    return rec_data, match

# -----------------------------------------------------------------------------
# 3. YARDIMCI GÖRSELLEŞTİRME FONKSİYONLARI
# -----------------------------------------------------------------------------

def display_movie_cards(movies_data, col_count=4):
    """Filmleri modern kartlar halinde gösterir"""
    cols = st.columns(col_count)
    
    for idx, movie in enumerate(movies_data):
        with cols[idx % col_count]:
            html_content = f"""
            <div class="movie-card">
                <div style="font-size: 3rem; text-align: center; margin-bottom: 10px;">🎬</div>
                <div class="card-title" title="{movie['Film']}">{movie['Film']}</div>
                <div class="card-metric">
                    <span>📅 {movie['Yıl']}</span>
                    <span class="score-badge">★ {movie['IMDb']:.1f}</span>
                </div>
                <div style="font-size: 0.8rem; color: #888; margin-top: 5px;">
                    {movie.get('Türler', 'Genel')[:30]}...
                </div>
                {'<div style="margin-top:10px; font-size:0.8rem; color:#40E0D0;">Match: %' + str(int(movie["Benzerlik"]*100)) + '</div>' if "Benzerlik" in movie else ''}
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. ANA UYGULAMA
# -----------------------------------------------------------------------------

def main():
    # Session state başlatma - ÖNEMLI: Menü çoğalmasını önler
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.df = None
        st.session_state.df_filtered = None
        st.session_state.movie_similarity_df = None
        st.session_state.normalized_titles_dict = None
        st.session_state.year_best = None
        st.session_state.genre_best = None
        st.session_state.all_genres = None

    # --- HERO SECTION ---
    st.markdown("""
        <div class="hero-container">
            <h1 class="main-title">CineAI</h1>
            <p class="subtitle">Yapay zeka destekli yeni nesil film keşif platformu.</p>
        </div>
    """, unsafe_allow_html=True)

    # Veri Yükleme - Sadece bir kez
    if not st.session_state.data_loaded:
        FILE_ID = "1gl_iJXRyEaSzhHlgfBUdTzQZMer4gdsS"
        filepath = download_data_from_drive(FILE_ID)
        
        if filepath:
            with st.spinner('🚀 CineAI motoru ve veri seti yükleniyor...'):
                df, df_filtered, _, movie_similarity_df, normalized_titles_dict, year_best, genre_best, all_genres = prepare_data(filepath)
                
                if df is not None:
                    st.session_state.df = df
                    st.session_state.df_filtered = df_filtered
                    st.session_state.movie_similarity_df = movie_similarity_df
                    st.session_state.normalized_titles_dict = normalized_titles_dict
                    st.session_state.year_best = year_best
                    st.session_state.genre_best = genre_best
                    st.session_state.all_genres = all_genres
                    st.session_state.data_loaded = True
                    st.rerun()
                else:
                    st.stop()
        else:
            st.stop()
    
    # Session state'ten verileri al
    df = st.session_state.df
    df_filtered = st.session_state.df_filtered
    movie_similarity_df = st.session_state.movie_similarity_df
    normalized_titles_dict = st.session_state.normalized_titles_dict
    year_best = st.session_state.year_best
    genre_best = st.session_state.genre_best
    all_genres = st.session_state.all_genres

    # --- SIDEBAR (İSTATİSTİKLER) ---
    with st.sidebar:
        st.header("📊 Veritabanı")
        st.markdown("---")
        
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("Film", f"{df['TITLE'].nunique()//1000}K+")
        col_s2.metric("Kullanıcı", f"{df['USERID'].nunique()//1000}K+")
        
        st.markdown("### 📈 Trend")
        year_counts = df.groupby('YEAR')['TITLE'].nunique().reset_index()
        fig_mini = px.area(year_counts, x='YEAR', y='TITLE')
        fig_mini.update_layout(
            height=150, 
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            showlegend=False
        )
        fig_mini.update_traces(line_color='#40E0D0', fillcolor='rgba(64, 224, 208, 0.2)')
        st.plotly_chart(fig_mini, use_container_width=True, key="sidebar_chart")

    # --- ANA MENÜ (TABS) - Key parametresi eklendi ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Film Önerisi", 
        "🏆 Yılın En İyileri", 
        "🎭 Tür Keşfi", 
        "📊 Veri Analizi"
    ])

    # 1. SEKME: ÖNERİ SİSTEMİ
    with tab1:
        st.markdown("### 🎬 Ne izledin ve beğendin?")
        
        col_search, col_count = st.columns([3, 1])
        with col_search:
            movie_input = st.text_input("film_search", placeholder="🔍 Film adı yazın... (örn: Inception, Matrix)", 
                                       label_visibility="collapsed", key="movie_search_input")
        with col_count:
            num_rec = st.selectbox("", [4, 8, 12], index=0, label_visibility="collapsed", 
                                  format_func=lambda x: f"{x} Öneri", key="num_rec_select")

        if st.button("Bana Benzerlerini Bul", type="primary", key="search_button"):
            if movie_input:
                with st.spinner('🔍 Benzer filmler aranıyor...'):
                    recommendations, match = recommend_by_title(
                        movie_input, movie_similarity_df, df, num_rec, normalized_titles_dict
                    )
                
                if recommendations:
                    st.success(f"✅ **{match}** filmine dayalı önerilerimiz:")
                    st.markdown("---")
                    display_movie_cards(recommendations, col_count=4)
                else:
                    st.warning("Bu filmi bulamadık. Şunları mı demek istediniz?")
                    for alt in match:
                        st.info(f"👉 {alt}")
            else:
                st.error("Lütfen bir film adı girin.")

    # 2. SEKME: YILA GÖRE EN İYİLER (ÖN HESAPLI - ÇOK HIZLI)
    with tab2:
        col_y1, col_y2 = st.columns([1, 3])
        with col_y1:
            years = sorted(df['YEAR'].unique(), reverse=True)
            sel_year = st.selectbox("Yıl Seçin", years, key="year_select")
        
        # Cache'den direkt çek - ANLIK!
        top_year = year_best.get(sel_year, pd.DataFrame())
        
        top_year_list = []
        for _, row in top_year.iterrows():
            top_year_list.append({
                "Film": row['TITLE'],
                "IMDb": row['IMDB_SCORE'],
                "Yıl": sel_year,
                "Türler": row['GENRES'].replace("|", ", ")
            })
            
        st.markdown(f"### 🏆 {sel_year} Yılının Efsaneleri")
        if top_year_list:
            display_movie_cards(top_year_list, col_count=4)
        else:
            st.info("Bu yıl için yeterli veri bulunamadı.")

    # 3. SEKME: TÜR KEŞFİ (ÖN HESAPLI - ÇOK HIZLI)
    with tab3:
        sel_genre = st.selectbox("Hangi türde film arıyorsun?", all_genres, key="genre_select")
        
        # Cache'den direkt çek - ANLIK!
        top_genre = genre_best.get(sel_genre, pd.DataFrame())
        
        top_genre_list = []
        genre_display_name = sel_genre if sel_genre != "(no genres listed)" else "Türsüz"
        
        for _, row in top_genre.iterrows():
            top_genre_list.append({
                "Film": row['TITLE'],
                "IMDb": row['IMDB_SCORE'],
                "Yıl": int(row['YEAR']),
                "Türler": genre_display_name
            })
            
        st.markdown(f"### 🎭 En İyi **{genre_display_name}** Filmleri")
        if top_genre_list:
            display_movie_cards(top_genre_list, col_count=4)
        else:
            st.info("Bu tür için yeterli veri bulunamadı.")

    # 4. SEKME: ANALİZ
    with tab4:
        st.markdown("### 📊 Veri Seti İçgörüleri")
        
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            st.markdown("**En Popüler Türler**")
            genres_count = df['GENRES'].str.get_dummies(sep='|').sum().sort_values(ascending=True).tail(10)
            fig_bar = px.bar(x=genres_count.values, y=genres_count.index, orientation='h')
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title="Film Sayısı",
                yaxis_title=None
            )
            fig_bar.update_traces(marker_color='#40E0D0')
            st.plotly_chart(fig_bar, use_container_width=True, key="genre_bar_chart")
            
        with col_a2:
            st.markdown("**IMDb Puan Dağılımı**")
            fig_hist = px.histogram(df_filtered, x='IMDB_SCORE', nbins=20)
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title="Puan",
                yaxis_title="Film Sayısı",
                bargap=0.1
            )
            fig_hist.update_traces(marker_color='#008B8B')
            st.plotly_chart(fig_hist, use_container_width=True, key="score_histogram")

if __name__ == "__main__":
    main()
