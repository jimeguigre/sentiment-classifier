"""
generate_dataset.py
Genera un dataset sintético más amplio y variado combinando plantillas
y vocabulario positivo/negativo, para tener suficientes ejemplos de
entrenamiento (uso educativo/demo). En un proyecto real se sustituiría
por un dataset público como IMDB, Sentiment140 o Amazon Reviews.
"""
import random
import csv

random.seed(42)

subjects = ["movie", "product", "restaurant", "book", "app", "service",
            "hotel", "phone", "laptop", "trip", "show", "album", "game", "course"]

positive_adjs = ["amazing", "fantastic", "excellent", "wonderful", "brilliant",
                  "outstanding", "delightful", "impressive", "great", "superb",
                  "enjoyable", "fabulous", "incredible", "charming", "solid"]

negative_adjs = ["terrible", "awful", "disappointing", "boring", "poor",
                  "dreadful", "mediocre", "frustrating", "unpleasant", "bad",
                  "annoying", "horrible", "weak", "dull", "useless"]

positive_templates = [
    "This {s} was {a}, I really loved it",
    "What an {a} {s}, highly recommend it to everyone",
    "The {s} exceeded my expectations, truly {a}",
    "I am so happy with this {s}, it was {a}",
    "An {a} experience with this {s}, will come back again",
    "Absolutely {a} {s}, one of the best I've seen",
    "The {s} was {a} and worth every penny",
    "Such a {a} {s}, I can't stop recommending it",
]

negative_templates = [
    "This {s} was {a}, I really regret it",
    "What a {a} {s}, would not recommend it to anyone",
    "The {s} fell far short of expectations, truly {a}",
    "I am so disappointed with this {s}, it was {a}",
    "A {a} experience with this {s}, will not come back",
    "Absolutely {a} {s}, one of the worst I've seen",
    "The {s} was {a} and a waste of money",
    "Such a {a} {s}, I can't stop complaining about it",
]

rows = []
for s in subjects:
    for t in positive_templates:
        for a in random.sample(positive_adjs, 4):
            rows.append((t.format(s=s, a=a), "positive"))
    for t in negative_templates:
        for a in random.sample(negative_adjs, 4):
            rows.append((t.format(s=s, a=a), "negative"))

random.shuffle(rows)

with open("data/reviews_large.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(rows)

print(f"Generadas {len(rows)} filas -> data/reviews_large.csv")
