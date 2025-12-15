import os
import joblib
import streamlit as st

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "results", "best_model.pkl")
VECT_PATH = os.path.join(BASE_DIR, "results", "vectorizer.pkl")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    st.error("Model files not found. Please train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.set_page_config(page_title="Spam Email Detection", layout="centered")

st.title("Spam Message Detection System")
st.write("Enter a message to check whether it is **Spam** or **Ham**.")

user_input = st.text_area("Message text", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        if prediction == 1:
            st.error("🚫 This message is SPAM")
        else:
            st.success("✅ This message is NOT spam (Ham)")
