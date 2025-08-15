import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
import requests
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Streamlit sayfa yapılandırması
st.set_page_config(
    page_title="🎬 Film Öneri Sistemi",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #7D8590;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem;
    }
    
    .recommendation-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def download_data_from_drive(file_id):
    """Google Drive'dan veri setini indir ve önbelleğe al"""
    try:
        # Google Drive doğrudan indirme URL'si
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        
        with st.spinner('📥 Veri seti indiriliyor... Bu işlem birkaç dakika sürebilir.'):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # BytesIO kullanarak veriyi belleğe yükle
            data_bytes = BytesIO(response.content)
            
        st.success("✅ Veri seti başarıyla indirildi!")
        return data_bytes
    
    except Exception as e:
        st.error(f"❌ Veri seti indirilemedi: {str(e)}")
        return None

def weighted_rating(rating, votes, min_votes, mean_rating):
    """Ağırlıklı derecelendirme hesapla (IMDb formatında)"""
    denominator = votes + min_votes
    if denominator == 0:
        return 0
    return (votes / denominator) * rating + (min_votes / denominator) * mean_rating

def normalize_title(title):
    """Film başlıklarını normalleştir"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', title)
        if unicodedata.category(c) != 'Mn'
    ).lower().strip()

@st.cache_data(ttl=3600)
def prepare_data(data_bytes, vote_threshold=1000, min_votes=2500):
    """Veri setini hazırla ve öneri sistemi için işle"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text('📊 Veri seti yükleniyor...')
        progress_bar.progress(10)
        
        # CSV'yi oku
        df = pd.read_csv(data_bytes)
        
        status_text.text('🔧 Veri temizleniyor...')
        progress_bar.progress(30)
        
        # Başlık ve yıl bilgisini ayır
        df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")
        
        # Zaman bilgisini datetime'a çevir
        df["TIME"] = pd.to_datetime(df["TIME"], dayfirst=True, errors='coerce')
        
        # Eksik verileri temizle
        df.dropna(subset=["TITLE", "YEAR", "TIME", "RATING"], inplace=True)
        
        # Yılı integer'a çevir
        df["YEAR"] = df["YEAR"].astype(int)
        
        # Derecelendirmeyi 10'luk sisteme çevir
        df["RATING_10"] = df["RATING"] * 2
        
        status_text.text('📈 İstatistikler hesaplanıyor...')
        progress_bar.progress(50)
        
        # Film başına oy sayılarını hesapla
        vote_counts = df.groupby("TITLE")["RATING"].count()
        df["NUM_VOTES"] = df["TITLE"].map(vote_counts)
        
        # Ortalama derecelendirme
        mean_rating = df["RATING_10"].mean()
        
        # Film istatistiklerini topla
        movie_stats = df.groupby("TITLE").agg({
            "RATING_10": "mean",
            "NUM_VOTES": "max"
        }).reset_index()
        
        # Ağırlıklı IMDb skorlarını hesapla
        movie_stats["IMDB_SCORE"] = movie_stats.apply(
            lambda x: weighted_rating(x["RATING_10"], x["NUM_VOTES"], min_votes, mean_rating),
            axis=1
        )
        
        # Skorları ana veri çerçevesine ekle
        df["IMDB_SCORE"] = df["TITLE"].map(movie_stats.set_index("TITLE")["IMDB_SCORE"])
        
        status_text.text('🎯 Öneri matrisi oluşturuluyor...')
        progress_bar.progress(70)
        
        # Popüler filmleri filtrele
        popular_titles = vote_counts[vote_counts >= vote_threshold].index
        df_filtered = df[df["TITLE"].isin(popular_titles)].copy()
        
        # Kullanıcı-film derecelendirme matrisi
        user_movie_matrix = df_filtered.pivot_table(
            index="USERID",
            columns="TITLE",
            values="RATING_10",
            aggfunc='mean'
        ).fillna(0)
        
        status_text.text('🔄 Benzerlik matrisi hesaplanıyor...')
        progress_bar.progress(90)
        
        # Film benzerlik matrisi (cosine similarity)
        movie_similarity_df = pd.DataFrame(
            cosine_similarity(user_movie_matrix.T),
            index=user_movie_matrix.columns,
            columns=user_movie_matrix.columns
        )
        
        # Normalleştirilmiş başlık sözlüğü
        normalized_titles_dict = {normalize_title(t): t for t in movie_similarity_df.columns}
        
        progress_bar.progress(100)
        status_text.text('✅ Veri hazırlama tamamlandı!')
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return df, df_filtered, user_movie_matrix, movie_similarity_df, normalized_titles_dict
        
    except Exception as e:
        st.error(f"❌ Veri hazırlanırken hata oluştu: {str(e)}")
        return None, None, None, None, None

def find_best_match(input_title, normalized_titles_dict):
    """En iyi eşleşen film başlığını bul"""
    normalized_input = normalize_title(input_title)
    close_matches = difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=1)
    return normalized_titles_dict[close_matches[0]] if close_matches else None

def suggest_alternatives(input_title, normalized_titles_dict, n=3):
    """Alternatif film önerileri"""
    normalized_input = normalize_title(input_title)
    return [normalized_titles_dict[t] for t in difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=n)]

def recommend_by_title(title, similarity_df, df, top_n=5, normalized_titles_dict=None):
    """Film başlığına göre öneri yap"""
    match = find_best_match(title, normalized_titles_dict)
    
    if not match:
        alternatives = suggest_alternatives(title, normalized_titles_dict)
        return None, alternatives
    
    scores = similarity_df[match].drop(labels=[match], errors="ignore")
    recommendations = scores.sort_values(ascending=False).head(top_n)
    
    # Film bilgilerini ekle
    rec_data = []
    for movie, similarity_score in recommendations.items():
        movie_info = df[df["TITLE"] == movie].iloc[0]
        rec_data.append({
            "Film": movie,
            "Benzerlik Skoru": f"{similarity_score:.3f}",
            "IMDb Skoru": f"{movie_info['IMDB_SCORE']:.2f}",
            "Yıl": int(movie_info["YEAR"]),
            "Türler": movie_info["GENRES"]
        })
    
    return rec_data, match

def recommend_by_user(user_id, user_matrix, similarity_df, df, top_n=5):
    """Kullanıcı geçmişine göre öneri yap"""
    if user_id not in user_matrix.index:
        return None
    
    user_ratings = user_matrix.loc[user_id]
    watched = user_ratings[user_ratings > 0]
    
    if watched.empty:
        return []
    
    scores = similarity_df[watched.index].dot(watched)
    scores = scores.drop(watched.index, errors='ignore')
    recommendations = scores.sort_values(ascending=False).head(top_n)
    
    # Film bilgilerini ekle
    rec_data = []
    for movie, score in recommendations.items():
        movie_info = df[df["TITLE"] == movie].iloc[0]
        rec_data.append({
            "Film": movie,
            "Öneri Skoru": f"{score:.2f}",
            "IMDb Skoru": f"{movie_info['IMDB_SCORE']:.2f}",
            "Yıl": int(movie_info["YEAR"]),
            "Türler": movie_info["GENRES"]
        })
    
    return rec_data

def get_top_movies_by_year(df, year, top_n=10):
    """Yıla göre en iyi filmleri getir"""
    year_movies = df[df['YEAR'] == year]
    if year_movies.empty:
        return []
    
    top = year_movies.groupby(['TITLE', 'GENRES'])['IMDB_SCORE'].mean().reset_index()
    top = top.sort_values('IMDB_SCORE', ascending=False).head(top_n)
    
    return top.to_dict('records')

def get_top_movies_by_genre(df, genre, top_n=10):
    """Türe göre en iyi filmleri getir"""
    genre_movies = df[df["GENRES"].str.contains(genre, case=False, na=False)]
    if genre_movies.empty:
        return []
    
    top = genre_movies.groupby(['TITLE', 'YEAR'])['IMDB_SCORE'].mean().reset_index()
    top = top.sort_values('IMDB_SCORE', ascending=False).head(top_n)
    
    return top.to_dict('records')

def main():
    # Ana başlık
    st.markdown('<h1 class="main-header">🎬 Film Öneri Sistemi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">26M+ film verisi ile kişiselleştirilmiş öneriler</p>', unsafe_allow_html=True)
    
    # Google Drive dosya ID'si
    FILE_ID = "1mdXIj3yZWd6cNV8hc5T2rTMjjAEbfgfL"
    
    # Veri setini indir ve hazırla
    if 'data_loaded' not in st.session_state:
        data_bytes = download_data_from_drive(FILE_ID)
        
        if data_bytes is not None:
            df, df_filtered, user_movie_matrix, movie_similarity_df, normalized_titles_dict = prepare_data(data_bytes)
            
            if df is not None:
                st.session_state.df = df
                st.session_state.df_filtered = df_filtered
                st.session_state.user_movie_matrix = user_movie_matrix
                st.session_state.movie_similarity_df = movie_similarity_df
                st.session_state.normalized_titles_dict = normalized_titles_dict
                st.session_state.data_loaded = True
            else:
                st.error("❌ Veri hazırlanamadı. Lütfen sayfayı yenileyin.")
                return
        else:
            st.error("❌ Veri indirilemedi. Lütfen internet bağlantınızı kontrol edin.")
            return
    
    # Veri setini session state'den al
    df = st.session_state.df
    df_filtered = st.session_state.df_filtered
    user_movie_matrix = st.session_state.user_movie_matrix
    movie_similarity_df = st.session_state.movie_similarity_df
    normalized_titles_dict = st.session_state.normalized_titles_dict
    
    # Sidebar - İstatistikler
    with st.sidebar:
        st.header("📊 Veri Seti İstatistikleri")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Toplam Film", f"{df['TITLE'].nunique():,}")
            st.metric("Toplam Kullanıcı", f"{df['USERID'].nunique():,}")
        with col2:
            st.metric("Toplam Değerlendirme", f"{len(df):,}")
            st.metric("Ortalama IMDb Skoru", f"{df['IMDB_SCORE'].mean():.2f}")
        
        # Yıl dağılımı grafiği
        st.subheader("📅 Yıllara Göre Film Sayısı")
        year_counts = df.groupby('YEAR')['TITLE'].nunique().reset_index()
        fig = px.line(year_counts, x='YEAR', y='TITLE', 
                     title='Yıllara Göre Film Sayısı')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Ana içerik
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Film Bazlı Öneriler", 
        "👤 Kullanıcı Bazlı Öneriler", 
        "📅 Yıla Göre En İyiler",
        "🎭 Türe Göre En İyiler",
        "🔍 Veri Keşfi"
    ])
    
    with tab1:
        st.header("🎯 Film Bazlı Öneriler")
        st.write("Sevdiğiniz bir filmi yazın, benzer filmleri önereceğiz!")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            movie_input = st.text_input("Film Adı:", placeholder="Örnek: The Shawshank Redemption")
        with col2:
            num_recommendations = st.selectbox("Öneri Sayısı:", [5, 10, 15, 20], index=0)
        
        if st.button("🎬 Öneri Al", type="primary"):
            if movie_input:
                recommendations, match_or_alternatives = recommend_by_title(
                    movie_input, movie_similarity_df, df, num_recommendations, normalized_titles_dict
                )
                
                if recommendations is None:
                    st.error("❌ Film bulunamadı. Şunları kastetmiş olabilir misiniz?")
                    for alt in match_or_alternatives:
                        st.write(f"• {alt}")
                else:
                    st.success(f"✅ '{match_or_alternatives}' filmine göre öneriler:")
                    
                    # Önerileri tablo olarak göster
                    rec_df = pd.DataFrame(recommendations)
                    st.dataframe(rec_df, use_container_width=True)
                    
                    # Benzerlik skoru grafiği
                    fig = px.bar(rec_df, x='Film', y='Benzerlik Skoru', 
                               title=f'{match_or_alternatives} - Benzerlik Skorları',
                               color='IMDb Skoru', color_continuous_scale='viridis')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Lütfen bir film adı girin.")
    
    with tab2:
        st.header("👤 Kullanıcı Bazlı Öneriler")
        st.write("Kullanıcı ID'sine göre kişiselleştirilmiş öneriler!")
        
        # En aktif kullanıcıları göster
        top_users = df["USERID"].value_counts().head(20).index.tolist()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            user_input = st.selectbox("Kullanıcı ID seçin:", [""] + top_users)
        with col2:
            num_user_rec = st.selectbox("Öneri Sayısı:", [5, 10, 15, 20], index=1)
        
        if st.button("👤 Kullanıcı Önerileri Al", type="primary"):
            if user_input:
                user_id = int(user_input)
                recommendations = recommend_by_user(
                    user_id, user_movie_matrix, movie_similarity_df, df, num_user_rec
                )
                
                if recommendations is None:
                    st.error("❌ Kullanıcı bulunamadı.")
                elif not recommendations:
                    st.warning("⚠️ Bu kullanıcı için izlenmiş film bulunamadı.")
                else:
                    st.success(f"✅ Kullanıcı {user_id} için öneriler:")
                    
                    # Kullanıcının izlediği filmler
                    user_movies = df[df['USERID'] == user_id]['TITLE'].unique()
                    st.write(f"**İzlediği film sayısı:** {len(user_movies)}")
                    
                    with st.expander("İzlediği filmlerden bazıları"):
                        for movie in user_movies[:10]:
                            st.write(f"• {movie}")
                    
                    # Önerileri göster
                    rec_df = pd.DataFrame(recommendations)
                    st.dataframe(rec_df, use_container_width=True)
                    
                    # Öneri skoru grafiği
                    fig = px.bar(rec_df, x='Film', y='Öneri Skoru', 
                               title=f'Kullanıcı {user_id} - Öneri Skorları',
                               color='IMDb Skoru', color_continuous_scale='plasma')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Lütfen bir kullanıcı ID seçin.")
    
    with tab3:
        st.header("📅 Yıla Göre En İyi Filmler")
        
        years = sorted(df['YEAR'].unique(), reverse=True)
        selected_year = st.selectbox("Yıl seçin:", years)
        
        if st.button("📅 Yılın En İyilerini Göster", type="primary"):
            top_movies = get_top_movies_by_year(df_filtered, selected_year)
            
            if not top_movies:
                st.error(f"❌ {selected_year} yılı için film bulunamadı.")
            else:
                st.success(f"✅ {selected_year} yılının en iyi filmleri:")
                
                # Tablo olarak göster
                movies_df = pd.DataFrame(top_movies)
                movies_df = movies_df.rename(columns={
                    'TITLE': 'Film',
                    'IMDB_SCORE': 'IMDb Skoru',
                    'GENRES': 'Türler'
                })
                st.dataframe(movies_df, use_container_width=True)
                
                # Grafik
                fig = px.bar(movies_df, x='Film', y='IMDb Skoru', 
                           title=f'{selected_year} Yılının En İyi Filmleri',
                           color='IMDb Skoru', color_continuous_scale='blues')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("🎭 Türe Göre En İyi Filmler")
        
        # Mevcut türleri al
        all_genres = set()
        for genres in df['GENRES'].dropna():
            all_genres.update([g.strip() for g in genres.split('|')])
        
        popular_genres = ['Action', 'Comedy', 'Drama', 'Romance', 'Thriller', 
                         'Horror', 'Adventure', 'Animation', 'Crime', 'Mystery']
        
        available_popular = [g for g in popular_genres if g in all_genres]
        other_genres = sorted([g for g in all_genres if g not in popular_genres])
        
        genre_options = available_popular + other_genres
        
        selected_genre = st.selectbox("Tür seçin:", genre_options)
        
        if st.button("🎭 Türün En İyilerini Göster", type="primary"):
            top_movies = get_top_movies_by_genre(df_filtered, selected_genre)
            
            if not top_movies:
                st.error(f"❌ {selected_genre} türü için film bulunamadı.")
            else:
                st.success(f"✅ {selected_genre} türünün en iyi filmleri:")
                
                # Tablo olarak göster
                movies_df = pd.DataFrame(top_movies)
                movies_df = movies_df.rename(columns={
                    'TITLE': 'Film',
                    'YEAR': 'Yıl',
                    'IMDB_SCORE': 'IMDb Skoru'
                })
                st.dataframe(movies_df, use_container_width=True)
                
                # Grafik
                fig = px.scatter(movies_df, x='Yıl', y='IMDb Skoru', 
                               size='IMDb Skoru', hover_name='Film',
                               title=f'{selected_genre} Türü - Yıl ve IMDb Skoru Dağılımı',
                               color='IMDb Skoru', color_continuous_scale='reds')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("🔍 Veri Keşfi ve Analiz")
        
        # Genel istatistikler
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam Film", f"{df['TITLE'].nunique():,}")
        with col2:
            st.metric("Toplam Değerlendirme", f"{len(df):,}")
        with col3:
            st.metric("Ortalama Rating", f"{df['RATING'].mean():.2f}")
        with col4:
            st.metric("En Son Yıl", f"{df['YEAR'].max()}")
        
        # Grafik seçenekleri
        chart_type = st.selectbox("Grafik türü seçin:", [
            "En Çok Değerlendirilen Filmler",
            "Yıllara Göre Film Sayısı", 
            "En Popüler Türler",
            "Rating Dağılımı"
        ])
        
        if chart_type == "En Çok Değerlendirilen Filmler":
            top_rated = df['TITLE'].value_counts().head(20)
            fig = px.bar(x=top_rated.values, y=top_rated.index, 
                        title='En Çok Değerlendirilen 20 Film',
                        labels={'x': 'Değerlendirme Sayısı', 'y': 'Film'},
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Yıllara Göre Film Sayısı":
            year_counts = df.groupby('YEAR')['TITLE'].nunique().reset_index()
            fig = px.area(year_counts, x='YEAR', y='TITLE', 
                         title='Yıllara Göre Film Sayısı Trend')
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "En Popüler Türler":
            # Türleri ayır ve say
            genre_counts = {}
            for genres in df['GENRES'].dropna():
                for genre in genres.split('|'):
                    genre = genre.strip()
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Tür', 'Sayı'])
            genre_df = genre_df.sort_values('Sayı', ascending=False).head(15)
            
            fig = px.bar(genre_df, x='Sayı', y='Tür', 
                        title='En Popüler 15 Film Türü',
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Rating Dağılımı":
            fig = px.histogram(df, x='RATING', nbins=50, 
                             title='Rating Dağılımı (1-5 Skala)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Veri seti örneği
        st.subheader("📋 Veri Seti Örneği")
        st.dataframe(df.head(100), use_container_width=True)

if __name__ == "__main__":
    main()
