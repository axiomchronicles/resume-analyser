import joblib
import sys
import numpy as np


model = joblib.load("models/resume_classifier_model.joblib")
mlb = joblib.load("models/resume_label_binarizer.joblib")


def classify(text, threshold=0.5):
    proba = model.predict_proba([text])[0]
    mask = proba >= threshold
    labels = mlb.classes_[mask]
    return list(labels)