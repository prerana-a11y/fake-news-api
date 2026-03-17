import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

# Combine datasets
df = pd.concat([fake, real])

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Remove empty rows
df = df.dropna()


# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)


# Input / Output
X = df["text"]
y = df["label"]


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Vectorize
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)


# Accuracy
accuracy = model.score(X_test_vec, y_test)
print("Accuracy:", accuracy * 100)


# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model training completed")
