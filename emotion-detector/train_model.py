from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from data_loader import get_data
import os

try:
    print("Loading data...")
    X_train, X_test, y_train, y_test = get_data()
    print("Data loaded. Training model...")
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    print("Model trained. Evaluating...")
    print("Classification Report:\n")
    print(classification_report(y_test, model.predict(X_test)))
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/emotion_model.pkl")
    print("Model saved to models/emotion_model.pkl")
except Exception as e:
    print(f"An error occurred: {e}") 