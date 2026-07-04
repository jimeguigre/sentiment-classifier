# Clasificador de Sentimientos con NLP

Modelo de clasificación de texto para análisis de sentimientos (positivo/negativo)
usando técnicas clásicas de aprendizaje automático: TF-IDF + SVM / Random Forest / Naive Bayes.

## Estructura del proyecto

```
sentiment-classifier/
├── data/
│   ├── reviews.csv          # dataset pequeño de ejemplo
│   └── reviews_large.csv    # dataset ampliado (generado)
├── preprocess.py            # limpieza y normalización de texto
├── generate_dataset.py      # generador de dataset sintético
├── train.py                 # entrenamiento y evaluación de modelos
├── predict.py                # inferencia por línea de comandos
├── requirements.txt
└── README.md
```

## 1. Instalación

```bash
pip install -r requirements.txt
```

## 2. Preprocesamiento

`preprocess.py` implementa un pipeline de limpieza de texto:
- Conversión a minúsculas
- Eliminación de URLs y puntuación
- Eliminación de stopwords en inglés
- Tokenización simple

```bash
python preprocess.py
```

## 3. Dataset

Se incluye un dataset de ejemplo (`data/reviews.csv`) y un generador
(`generate_dataset.py`) que crea un dataset sintético más amplio
(`data/reviews_large.csv`, ~900 filas balanceadas) combinando plantillas
y vocabulario positivo/negativo. **En un proyecto real**, sustituye esto
por un dataset público como IMDB Reviews, Sentiment140 o Amazon Reviews
(descargable con `pip install datasets` + Hugging Face, o Kaggle).

```bash
python generate_dataset.py
```

## 4. Entrenamiento y evaluación

`train.py` hace lo siguiente:
1. Carga y limpia los datos
2. Vectoriza el texto con `TfidfVectorizer` (unigramas + bigramas)
3. Entrena tres algoritmos: **SVM (LinearSVC)**, **Random Forest** y **Naive Bayes**
4. Evalúa cada modelo con accuracy, precision, recall, F1-score y matriz de confusión
5. Guarda el mejor modelo (`model.pkl`) y el vectorizador (`vectorizer.pkl`)

```bash
python train.py
```

Salida esperada:

```
=== Comparativa final de modelos ===
       model  accuracy  precision  recall   f1
         SVM       1.0        1.0     1.0  1.0
RandomForest       1.0        1.0     1.0  1.0
  NaiveBayes       1.0        1.0     1.0  1.0
```

(Con el dataset sintético incluido los modelos alcanzan 100% porque las
plantillas son muy regulares; con un dataset real de reseñas la accuracy
típica ronda 80-90% y ahí es donde se aprecian mejor las diferencias entre
SVM, Random Forest y Naive Bayes.)

## 5. Predicción

```bash
python predict.py "This app was absolutely amazing, I loved using it"
# Sentimiento predicho: positive

python predict.py "The service was terrible and I want a refund"
# Sentimiento predicho: negative
```
