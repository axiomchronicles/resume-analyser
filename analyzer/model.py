# train.py
import pandas as pd
import ast
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def parse_labels(lbl):
    if isinstance(lbl, str) and lbl.startswith("["):
        return ast.literal_eval(lbl)  
    return [lbl]                       


train_df = pd.read_csv("dataset/resume_suggestions_train.csv")
test_df = pd.read_csv("dataset/resume_suggestions_test.csv")

X_train = train_df["suggestion"]
y_train = train_df["label"].apply(parse_labels)

X_test = test_df["suggestion"]
y_test = test_df["label"].apply(parse_labels)


mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train)
y_test_bin = mlb.transform(y_test)


model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", OneVsRestClassifier(LogisticRegression(max_iter=300)))
])


print("\nTraining classifier...")
model.fit(X_train, y_train_bin)


preds = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test_bin, preds, target_names=mlb.classes_))


joblib.dump(model, "resume_classifier_model.joblib")
joblib.dump(mlb, "resume_label_binarizer.joblib")

print("\nModel and label binarizer saved:")
print(" - resume_classifier_model.joblib")
print(" - resume_label_binarizer.joblib")
