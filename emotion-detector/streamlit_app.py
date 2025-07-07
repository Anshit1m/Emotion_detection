import streamlit as st
import joblib

label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

model = joblib.load("models/emotion_model.pkl")

st.title("😃 Emotion Detection from Text")
text_input = st.text_input("Enter a sentence:")

if st.button("Detect Emotion"):
    prediction = model.predict([text_input])[0]
    emotion = label_map.get(prediction, "Unknown")
    st.success(f"Predicted Emotion: **{emotion}** (label: {prediction})") 