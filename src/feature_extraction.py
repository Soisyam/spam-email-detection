from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_tfidf(corpus):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=3000,
        ngram_range=(1,2)
    )
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer


def extract_metadata_features(texts):
    features = []

    for text in texts:
        features.append([
            len(text),                    # length of email
            sum(1 for c in text if c.isupper()),  # capital letters
            text.count('!'),              # exclamation marks
            text.count('http'),           # links
            sum(1 for c in text if c.isdigit())   # digits
        ])

    return np.array(features)
