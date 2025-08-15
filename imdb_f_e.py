import pandas as pd
import numpy as np
import unicodedata
import difflib
import logging
from sklearn.metrics.pairwise import cosine_similarity
from colorama import init, Fore, Style

init(autoreset=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.2f}'.format

def weighted_rating(rating, votes, min_votes, mean_rating):
    """
    Calculate the weighted rating.

    rating: Average rating of the movie
    votes: Number of votes for the movie
    min_votes: Minimum votes required to be listed
    mean_rating: Overall mean rating across all movies

    This formula balances the rating by considering vote counts,
    similar to IMDb's weighted rating system.
    """
    denominator = votes + min_votes
    if denominator == 0:
        return 0
    return (votes / denominator) * rating + (min_votes / denominator) * mean_rating

def normalize_title(title):
    """
    Normalize movie titles by removing accents,
    converting to lowercase, and trimming spaces.

    This helps in matching titles accurately.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', title)
        if unicodedata.category(c) != 'Mn'
    ).lower().strip()

def prepare_data(filepath="others/movies_imdb.csv", vote_threshold=10, min_votes=2500):
    """
    Prepare the dataset for recommendations:

    - Load CSV file.
    - Extract title and year from the TITLE column.
    - Convert TIME column to datetime.
    - Remove rows with missing critical data.
    - Compute vote counts and mean ratings.
    - Calculate weighted IMDb scores.
    - Filter movies exceeding vote threshold.
    - Create user-movie rating matrix.
    - Compute movie similarity matrix using cosine similarity.
    - Create normalized title dictionary for quick lookup.
    """
    df = pd.read_csv(filepath)

    # Extract title and year using regex
    df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")

    # Convert TIME to datetime format
    df["TIME"] = pd.to_datetime(df["TIME"], dayfirst=True, errors='coerce')

    # Drop rows with missing TITLE, YEAR, TIME, or RATING
    df.dropna(subset=["TITLE", "YEAR", "TIME", "RATING"], inplace=True)

    # Convert YEAR to integer
    df["YEAR"] = df["YEAR"].astype(int)

    # Normalize ratings to a 10-point scale
    df["RATING_10"] = df["RATING"] * 2

    # Count votes per movie
    vote_counts = df.groupby("TITLE")["RATING"].count()
    df["NUM_VOTES"] = df["TITLE"].map(vote_counts)

    # Overall mean rating across all movies
    mean_rating = df["RATING_10"].mean()

    # Aggregate mean rating and max votes per movie
    movie_stats = df.groupby("TITLE").agg({
        "RATING_10": "mean",
        "NUM_VOTES": "max"
    }).reset_index()

    # Calculate weighted IMDb scores
    movie_stats["IMDB_SCORE"] = movie_stats.apply(
        lambda x: weighted_rating(x["RATING_10"], x["NUM_VOTES"], min_votes, mean_rating),
        axis=1
    )

    # Map weighted scores back to the main dataframe
    df["IMDB_SCORE"] = df["TITLE"].map(movie_stats.set_index("TITLE")["IMDB_SCORE"])

    # Filter popular movies based on vote threshold
    popular_titles = vote_counts[vote_counts >= vote_threshold].index
    df_filtered = df[df["TITLE"].isin(popular_titles)].copy()

    # Create user-movie rating matrix
    user_movie_matrix = df_filtered.pivot_table(
        index="USERID",
        columns="TITLE",
        values="RATING_10",
        aggfunc='mean'
    ).fillna(0)

    # Compute movie similarity matrix using cosine similarity
    movie_similarity_df = pd.DataFrame(
        cosine_similarity(user_movie_matrix.T),
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns
    )

    # Create a normalized title dictionary: normalized_title -> original_title
    normalized_titles_dict = {normalize_title(t): t for t in movie_similarity_df.columns}

    logging.info("Data preparation completed successfully.")
    return df, df_filtered, user_movie_matrix, movie_similarity_df, normalized_titles_dict

def find_best_match(input_title, normalized_titles_dict):
    """
    Find the best matching movie title from normalized titles.

    Uses difflib to get close matches and returns the closest one.
    Returns None if no match is found.
    """
    normalized_input = normalize_title(input_title)
    close_matches = difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=1)
    return normalized_titles_dict[close_matches[0]] if close_matches else None

def suggest_alternatives(input_title, normalized_titles_dict):
    """
    Suggest alternative movie titles if exact match not found.

    Returns up to 3 closest matches.
    """
    normalized_input = normalize_title(input_title)
    return [normalized_titles_dict[t] for t in difflib.get_close_matches(normalized_input, normalized_titles_dict.keys(), n=3)]

def recommend_by_title(title, similarity_df, top_n=5, watched=None, normalized_titles_dict=None):
    """
    Recommend movies based on a given movie title.

    - Finds the closest matching movie.
    - If none found, suggests alternatives.
    - Otherwise, returns top_n most similar movies excluding watched ones and the movie itself.
    """
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
    """
    Recommend movies based on a user's watching history.

    - Checks if user exists and has watched movies.
    - Computes weighted similarity scores for unseen movies.
    - Returns top_n recommendations.
    """
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
    """
    List top rated movies for a specific year.

    - Filters movies by year.
    - Handles invalid input or no movies for the year.
    - Prints and returns top_n movies by IMDb score.
    """
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
    """
    Recommend top movies based on genre.

    - Filters movies containing the genre.
    - Returns top_n movies sorted by IMDb score.
    """
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
    """
    Main application loop:

    - Prepares data.
    - Offers menu to user for different recommendation types.
    - Handles user input and prints recommendations.
    - Exits on 'q'.
    """
    df, df_filtered, user_movie_matrix, similarity_df, norm_titles = prepare_data()
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
"""
1. **Pandas:** Used as the main data processing library in the project. 
Movie and user rating data were loaded, cleaned, and processed from the IMDb dataset. 
Specifically, I utilized Pandas functions like `groupby` and `pivot_table` to extract 
title and year information, remove missing values, and create group-based summary statistics. 
Additionally, I worked with pivot tables to build the user-movie matrix, which forms the foundation 
of the recommendation algorithm.

2. **NumPy:** Used for the numerical operations underlying Pandas and especially to provide data 
to the `cosine_similarity` function. The NumPy array structure was employed to efficiently store 
large data matrices and to perform mathematical operations quickly.

3. **unicodedata:** Since movie titles can contain various characters and accents, Unicode normalization 
was applied to ensure consistency in title comparisons. This made it easier to match titles like 
“Amélie” containing special characters with their simplified versions such as “Amelie”.

4. **difflib:** Movie names entered by users may not exactly match those in the dataset. Therefore, 
I used the `get_close_matches` function to find the closest matching movie titles for the input. 
This approach helped find the correct movie even if there were missing letters or minor spelling errors, 
thus improving user experience.

5. **scikit-learn (cosine_similarity):** Cosine similarity was used to calculate similarity between vectors 
representing user ratings of movies. This method numerically expresses whether two movies were rated similarly 
by users and forms the core of the recommendation algorithm. Consequently, movies with high similarity scores 
are recommended to each other.

6. **colorama:** Used to produce colored outputs during command-line user interaction. Success messages, 
warnings, error messages, and informational notes were displayed in different colors to enhance readability 
and user experience. This was especially helpful for users to easily distinguish recommendation lists and error messages.

"""
"""
* **Data Source:** The dataset was obtained from the GroupLens website.
* **Story:** We researched all CSV-format IMDb and MovieLens datasets and finally **merged 6 different datasets** into a single CSV file.
* **Project Objective:** By processing user and movie data, calculate **IMDb weighted scores (`IMDB_SCORE`)** and, based on these scores:

  * Recommend similar movies by title
  * Provide recommendations based on a user’s watch history
  * List the top movies of a specific year
  * Recommend the best movies by genre
    This allows for **score-based year and genre recommendations**.
* **Number of Records:** \~26,000,000
* **Size:** \~1.8 GB
* **Data Content:** User–movie ratings and movie metadata
* **Columns:**

  * **USERID:** User ID
  * **TITLE:** Movie title and year
  * **YEAR:** Release year
  * **GENRES:** Movie genres
  * **RATING:** Rating out of 5
  * **RATING\_10:** Rating normalized to a 10-point scale
  * **TIME:** Rating date
  * **NUM\_VOTES:** Total number of votes
  * **IMDB\_SCORE:** Weighted IMDb score

"""
