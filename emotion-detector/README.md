# Emotion Detector

Detect user emotions (happy, sad, angry, etc.) from social media or chat text using machine learning.

## Features
- Loads and preprocesses a public emotion-labeled dataset
- Trains a text classification model (Logistic Regression)
- Simple Streamlit web app for real-time emotion prediction

## Project Structure
```
emotion-detector/
├── data_loader.py            # Loads and prepares the dataset
├── train_model.py            # Trains and saves the model
├── streamlit_app.py          # Streamlit web app
├── models/
│   └── emotion_model.pkl     # Saved model
├── requirements.txt          # Dependencies
└── README.md                 # Project overview
```

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the model:**
   ```bash
   python train_model.py
   ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage
- Enter a sentence in the app to detect its emotion.
- The model is trained on the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset.

## Notes
- You can further improve the model or UI as needed.
- Make sure to retrain the model if you change the data or model code. 