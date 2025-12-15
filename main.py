

from preprocessing import clean_text
from feature_extraction import extract_metadata_features
from scipy.sparse import hstack
import numpy as np
import joblib

# Load saved components
model = joblib.load("results/best_model.pkl")
vectorizer = joblib.load("results/vectorizer.pkl")

email = input("Enter email content: ")

cleaned = clean_text(email)
X_text = vectorizer.transform([cleaned])
X_meta = extract_metadata_features([cleaned])

X_final = hstack([X_text, X_meta])

prediction = model.predict(X_final)[0]

if hasattr(model, "predict_proba"):
    confidence = np.max(model.predict_proba(X_final))
    print("Spam" if prediction == 1 else "Not Spam", f"({confidence*100:.2f}% confidence)")
else:
    print("Spam" if prediction == 1 else "Not Spam")
