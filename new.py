# Core Libraries
import pandas as pd
import numpy as np
import re

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score

# Visualization (Optional)
import matplotlib.pyplot as plt
import seaborn as sns
# Load files
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Sample to balance memory/performance
ratings_sampled = ratings.sample(n=30000, random_state=42)

# Merge ratings with movie metadata
ratings_sampled = ratings_sampled.merge(movies, on='movieId', how='left')

# Create binary label: liked if rating >= 4
ratings_sampled['liked'] = (ratings_sampled['rating'] >= 4).astype(int)
# Extract year from movie title
def extract_year(title):
    match = re.search(r'\((\d{4})\)', str(title))
    return int(match.group(1)) if match else np.nan

ratings_sampled['year'] = ratings_sampled['title'].apply(extract_year)

# User-level features
ratings_sampled['user_activity'] = ratings_sampled.groupby('userId')['movieId'].transform('count')
ratings_sampled['user_avg_rating'] = ratings_sampled.groupby('userId')['rating'].transform('mean')

# One-hot encode genres
ratings_sampled['genres'] = ratings_sampled['genres'].apply(lambda x: str(x).split('|'))
mlb = MultiLabelBinarizer()
genre_dummies = mlb.fit_transform(ratings_sampled['genres'])
genre_df = pd.DataFrame(genre_dummies, columns=mlb.classes_)

# Combine genre features with main DataFrame
ratings_sampled = pd.concat([ratings_sampled.reset_index(drop=True), genre_df], axis=1)
# Define final features
feature_cols = mlb.classes_.tolist() + ['year', 'user_activity', 'user_avg_rating']

# Drop rows with missing year
ratings_sampled.dropna(subset=feature_cols, inplace=True)

# Create training data
X = ratings_sampled[feature_cols]
y = ratings_sampled['liked']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Improved Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# Load only a sample to avoid memory issues
imdb_reviews = pd.read_csv("IMDB Dataset.csv", nrows=20000)

# Check data
imdb_reviews.head()
# Convert sentiment labels to binary
imdb_reviews['sentiment'] = imdb_reviews['sentiment'].map({'positive': 1, 'negative': 0})

# TF-IDF vectorization of review text
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(imdb_reviews['review'])
y = imdb_reviews['sentiment']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Predict and evaluate
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

print("🎯 Logistic Regression Accuracy (Sentiment):", round(lr_accuracy * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, lr_pred))
