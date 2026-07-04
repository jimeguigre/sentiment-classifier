"""
predict.py
Uso: python predict.py "texto a analizar"
Carga el modelo y vectorizador entrenados y predice el sentimiento.
"""
import sys
import joblib
from preprocess import clean_text


def load_artifacts():
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("model.pkl")
    return vectorizer, model


def predict(text: str) -> str:
    vectorizer, model = load_artifacts()
    clean = clean_text(text)
    X = vectorizer.transform([clean])
    return model.predict(X)[0]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Uso: python predict.py "texto a analizar"')
        sys.exit(1)
    texto = " ".join(sys.argv[1:])
    resultado = predict(texto)
    print(f"Texto: {texto}")
    print(f"Sentimiento predicho: {resultado}")
