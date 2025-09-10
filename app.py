import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# ------------------------------
# Load Model
# ------------------------------
model = load_model("imdb_sentiment_model.h5")

# ------------------------------
# Load Tokenizer
# ------------------------------
# Save your tokenizer in notebook like this after training:
# with open("tokenizer.pkl", "wb") as f:
#     pickle.dump(tokenizer, f)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 200   # same as used during training

# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="Sentiment Analysis", page_icon="🎭", layout="centered")

st.title("🎭 Sentiment Analysis on IMDB Reviews")
st.write("Enter a movie review below and find out if the sentiment is **Positive** or **Negative**.")

# Input text box
user_input = st.text_area("Movie Review", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review first!")
    else:
        # Preprocess
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

        # Predict
        prediction = model.predict(padded)
        sentiment = "Positive 😀" if prediction[0][0] > 0.5 else "Negative 😞"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

        # Show result
        st.subheader(f"Prediction: {sentiment}")
        st.progress(float(confidence))
        st.write(f"Confidence: {confidence:.2f}")
