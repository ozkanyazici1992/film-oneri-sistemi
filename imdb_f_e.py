import pandas as pd
import numpy as np
import unicodedata
import difflib
import logging
from sklearn.metrics.pairwise import cosine_similarity
from colorama import init, Fore, Style
import gdown

init(autoreset=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.2f}'.format

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

def prepare_data(drive_file_id, vote_threshold=10, min_votes=250):
    """
    Prepare dataset directly from Google Drive.
    """
    # Google Drive'dan indir
    url = f"https://drive.google.com/uc?id={drive_file_id}"
    output = "movies_imdb_2.csv"
    gdown.download(url, output, quiet=False)

    df = pd.read_csv(output)

    # TITLE ve YEAR çıkarma
    df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")

    # TIME datetime
    df["TIME"] = pd.to_datetime(df["TIME"], dayfirst=True, errors='coerce')

    # Eksik değerleri sil
    df.dropna(subset=["TITLE", "YEAR", "TIME", "RATING"], inplace=True)

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

    logging.info("Data preparation completed successfully.")
    return df, df_filtered, user_movie_matrix, movie_similarity_df, normalized_titles_dict

# Buradan sonrası kod aynen önceki haliyle çalışır
def find_best_match(input_title, normalized_titles_dict):
    normalized_input = normalize_title(input_title)
    close_matches = difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=1)
    return normalized_titles_dict[close_matches[0]] if close_matches else None

def suggest_alternatives(input_title, normalized_titles_dict):
    normalized_input = normalize_title(input_title)
    return [normalized_titles_dict[t] for t in difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=3)]

def recommend_by_title(title, similarity_df, top_n=5, watched=None, normalized_titles_dict=None):
    watched = watched or set()
    match = find_best_match(title, normalized_titles_dict)

    if not match:
        print(Fore.RED + "❌ Movie not found. Did you mean:")
        for alternative in suggest_alternatives(title, normalized_titles_dict):
            print(Fore.YELLOW + f"- {alternative}")
        return []

    print(Fore.CYAN + f"\n🎯 Recommendations based on '{match}':")
    scores = similarity_df[match].drop(labels=watched.union({match}), errors="ignore")
    return scores.sort_values(ascending=False).head(top_n).index.tolist()

def recommend_by_user(user_id, user_matrix, similarity_df, top_n=5):
    if user_id not in user_matrix.index:
        print(Fore.RED + f"❌ User ID {user_id} not found.")
        return []

    user_ratings = user_matrix.loc[user_id]
    watched = user_ratings[user_ratings > 0]

    if watched.empty:
        print(Fore.YELLOW + "ℹ️ No watch history found for user.")
        return []

    scores = similarity_df[watched.index].dot(watched)
    scores = scores.drop(watched.index, errors='ignore')
    return scores.sort_values(ascending=False).head(top_n).index.tolist()

def top_movies_by_year(df, year, top_n=5):
    try:
        year = int(year)
        year_movies = df[df['YEAR'] == year]
        if year_movies.empty:
            print(Fore.RED + f"⚠️ No movies found for year {year}.")
            return []
        top = year_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(top_n)
        print(Fore.CYAN + f"\n🗓️ Top IMDb scored movies for year {year}:")
        for i, (title, score) in enumerate(top.items(), 1):
            print(f"{i}. {title} - IMDb Score: {score:.2f}")
        return top.index.tolist()
    except ValueError:
        print(Fore.RED + "⚠️ Invalid year input.")
        return []

def recommend_by_genre(df, genre, top_n=5):
    genre = genre.strip().title()
    genre_movies = df[df["GENRES"].str.contains(genre, case=False, na=False)]
    if genre_movies.empty:
        print(Fore.RED + f"⚠️ No movies found in genre '{genre}'.")
        return []
    top = genre_movies.groupby('TITLE')['IMDB_SCORE'].mean().sort_values(ascending=False).head(top_n)
    print(Fore.CYAN + f"\n🎬 Top IMDb scored movies in '{genre}':")
    for i, (title, score) in enumerate(top.items(), 1):
        print(f"{i}. {title} - IMDb Score: {score:.2f}")
    return top.index.tolist()

def main():
    drive_file_id = "1gl_iJXRyEaSzhHlgfBUdTzQZMer4gdsS"
    df, df_filtered, user_movie_matrix, similarity_df, norm_titles = prepare_data(drive_file_id=drive_file_id)
    watched_movies = set()

    while True:
        print(Fore.BLUE + Style.BRIGHT + "\n🎞️ Ready for KodBlessYou Movie Recommendations?\n")
        print(Fore.CYAN + "🔍 The choice is yours, movie lover!")

        print(Fore.LIGHTGREEN_EX + "\n🎥 [1] Movie Recommendations by Title")
        print(Fore.WHITE + "   → Enter a movie you've watched and get similar masterpieces!")

        print(Fore.LIGHTYELLOW_EX + "\n🧑‍💻 [2] Recommendations by User History")
        print(Fore.WHITE + "   → Based on your watch history, we pick movies you'll love!")

        print(Fore.LIGHTMAGENTA_EX + "\n📅 [3] Top Movies by Year")
        print(Fore.WHITE + "   → Pick a year and discover the best movies released then!")

        print(Fore.LIGHTCYAN_EX + "\n🎭 [4] Recommendations by Genre")
        print(Fore.WHITE + "   → From comedy to sci-fi, choose a genre and get recommendations!")

        print(Fore.LIGHTRED_EX + "\n❌ [q] Exit")
        print(Fore.WHITE + "   → Press 'q' to quit. But the movies will always be here...")

        print(Style.RESET_ALL)
        choice = input("Your choice: ").strip().lower()
        if choice == 'q':
            print(Fore.GREEN + "👋 Exiting program.")
            break

        watched_movies.clear()

        if choice == '1':
            movie = input(Fore.LIGHTBLUE_EX + "🎬 Enter your favorite movie to get recommendations: " + Fore.RESET).strip()
            if not movie:
                print(Fore.RED + "⚠️ Movie title cannot be empty.")
                continue
            recommendations = recommend_by_title(movie, similarity_df, top_n=5, watched=watched_movies, normalized_titles_dict=norm_titles)
            if recommendations:
                print(Fore.GREEN + "\n✅ Recommended Movies:")
                for i, rec_movie in enumerate(recommendations, 1):
                    score = df[df["TITLE"] == rec_movie]["IMDB_SCORE"].mean()
                    print(f"{i}. {rec_movie} - IMDb Score: {score:.2f}")
                    watched_movies.add(rec_movie)
            else:
                print(Fore.YELLOW + "🔍 No recommendations found.")

        elif choice == '2':
            try:
                top_users = df["USERID"].value_counts().head(10).index.tolist()
                user_input = input(f"\nTop active User IDs: {', '.join(map(str, top_users))}\nEnter one: ").strip()
                user_id = int(user_input)
                recommendations = recommend_by_user(user_id, user_movie_matrix, similarity_df)
                if recommendations:
                    print(Fore.GREEN + "\n✅ Recommended Movies:")
                    for i, rec_movie in enumerate(recommendations, 1):
                        score = df[df["TITLE"] == rec_movie]["IMDB_SCORE"].mean()
                        print(f"{i}. {rec_movie} - IMDb Score: {score:.2f}")
                else:
                    print(Fore.YELLOW + "🔍 No recommendations found.")
            except ValueError:
                print(Fore.RED + "⚠️ Invalid User ID input.")

        elif choice == '3':
            year_input = input(Fore.LIGHTMAGENTA_EX + "📅 Enter a year (e.g., 2015) to explore top movies: " + Fore.RESET).strip()
            top_movies_by_year(df_filtered, year_input)

        elif choice == '4':
            print(Fore.LIGHTCYAN_EX + "\n🎞️ Genre-based Recommendations:\n"
                                      " Action      | Comedy      | Drama       | Romance\n"
                                      " Thriller    | Biography   | Horror      | Adventure\n"
                                      " Animation   | Crime       | Mystery     | Fantasy\n"
                                      " War         | Western     | Documentary | Musical\n" + Fore.RESET)

            genre_input = input(Fore.LIGHTBLUE_EX + "🎬 Choose a genre for recommendations: " + Fore.RESET).strip()
            if not genre_input:
                print(Fore.RED + "⚠️ Genre cannot be empty.")
                continue
            recommend_by_genre(df_filtered, genre_input)

        else:
            print(Fore.RED + "⚠️ Invalid choice.")

if __name__ == "__main__":
    main()
