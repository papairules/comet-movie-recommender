import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Step 1: Load only top 1000 active users and top 300 movies using chunking
ratings_path = "ratings.csv"
movies_df = pd.read_csv("movies.csv")

# Chunk read ratings
chunk_iter = pd.read_csv(ratings_path, chunksize=500_000)
ratings_filtered_list = []

# We'll track the most frequent movieId and userId
movie_counter = {}
user_counter = {}

for chunk in chunk_iter:
    movie_counts = chunk['movieId'].value_counts().to_dict()
    user_counts = chunk['userId'].value_counts().to_dict()

    for k, v in movie_counts.items():
        movie_counter[k] = movie_counter.get(k, 0) + v
    for k, v in user_counts.items():
        user_counter[k] = user_counter.get(k, 0) + v

# Get top 300 popular movies & top 1000 active users
top_movies = sorted(movie_counter.items(), key=lambda x: x[1], reverse=True)[:300]
top_users = sorted(user_counter.items(), key=lambda x: x[1], reverse=True)[:1000]
top_movie_ids = set([m[0] for m in top_movies])
top_user_ids = set([u[0] for u in top_users])

# Reload chunks and filter data
chunk_iter = pd.read_csv(ratings_path, chunksize=500_000)
filtered_chunks = []

for chunk in chunk_iter:
    filtered = chunk[(chunk['movieId'].isin(top_movie_ids)) & (chunk['userId'].isin(top_user_ids))]
    filtered_chunks.append(filtered)

ratings_df = pd.concat(filtered_chunks)

# Merge titles
df = pd.merge(ratings_df, movies_df, on='movieId')

# Pivot matrix
user_movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')
user_movie_matrix_filled = user_movie_matrix.fillna(0)

# Similarity
movie_ratings_matrix = user_movie_matrix_filled.T
movie_similarity = cosine_similarity(movie_ratings_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=movie_ratings_matrix.index, columns=movie_ratings_matrix.index)

# -----------------------------
# Sentiment model setup
# -----------------------------
imdb_df = pd.read_csv("IMDB Dataset.csv")
imdb_df['clean_review'] = imdb_df['review'].apply(lambda x: re.sub(r'<.*?>', '', x.lower()))

label_encoder = LabelEncoder()
imdb_df['sentiment_encoded'] = label_encoder.fit_transform(imdb_df['sentiment'])

vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(imdb_df['clean_review'])
y = imdb_df['sentiment_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Recommendation function
def recommend_movies(movie_title, similarity_df, top_n=5):
    if movie_title not in similarity_df.columns:
        return f"❌ Movie '{movie_title}' not found in similarity matrix."
    scores = similarity_df[movie_title].sort_values(ascending=False)
    return scores.iloc[1:top_n+1]

# Sentiment prediction
def predict_sentiment(review_text):
    cleaned = re.sub(r'<.*?>', '', review_text.lower())
    vector = vectorizer.transform([cleaned])
    pred = nb_model.predict(vector)[0]
    return label_encoder.inverse_transform([pred])[0]

# Mock reviews
mock_reviews = {
    "Toy Story 2 (1999)": "Even better than the first! Heartwarming and fun.",
    "Forrest Gump (1994)": "A timeless classic. Forrest is such a lovable character.",
    "Back to the Future (1985)": "An overrated mess. I couldn’t finish it.",
    "Jurassic Park (1993)": "Scary, thrilling, and visually stunning.",
    "Star Wars: Episode IV - A New Hope (1977)": "Way too slow and outdated for today's audience."
}

__all__ = [
    "movie_similarity_df",
    "recommend_movies",
    "predict_sentiment",
    "mock_reviews"
]
