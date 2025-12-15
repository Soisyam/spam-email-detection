import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD DATA (CSV VERSION)
# --------------------------------------------------

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, encoding="latin-1")

# Keep only required columns
df = df[["v1", "v2"]]
df.columns = ["label", "text"]

# Encode labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Drop empty rows
df.dropna(inplace=True)

print("Dataset loaded successfully")
print("Total samples:", len(df))
print(df.head())

# --------------------------------------------------
# TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print(report)

# --------------------------------------------------
# SAVE MODEL & VECTORIZER
# --------------------------------------------------

joblib.dump(model, os.path.join(RESULTS_DIR, "best_model.pkl"))
joblib.dump(vectorizer, os.path.join(RESULTS_DIR, "vectorizer.pkl"))

with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(report)

print("\nTraining completed successfully.")
print("Model and vectorizer saved in /results")
