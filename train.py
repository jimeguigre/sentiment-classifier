"""
train.py
Entrena y compara varios algoritmos (SVM, Random Forest, Naive Bayes)
para clasificación de sentimientos, usando TF-IDF como representación.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

from preprocess import preprocess_series

DATA_PATH = "data/reviews_large.csv"
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["clean_text"] = preprocess_series(df["text"])
    return df


def build_models():
    return {
        "SVM": LinearSVC(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "NaiveBayes": MultinomialNB(),
    }


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    df = load_data()
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.25,
        random_state=RANDOM_STATE, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    results = []
    trained_models = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        results.append(evaluate(model, X_test, y_test, name))

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("\n=== Comparativa final de modelos ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_name]
    print(f"\nMejor modelo: {best_name} -> guardado como model.pkl")

    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(best_model, "model.pkl")
    results_df.to_csv("model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
