"""
preprocess.py
Pipeline de preprocesamiento de texto para el clasificador de sentimientos.
"""
import re
import string

STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "s", "same", "she", "should",
    "so", "some", "such", "t", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "you", "your", "yours", "yourself", "yourselves",
}


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Limpia y normaliza un texto: minúsculas, sin puntuación, sin stopwords."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"[^a-záéíóúñü\s]", " ", text)          # solo letras
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def preprocess_series(texts):
    """Aplica clean_text a una lista/serie de textos."""
    return [clean_text(t) for t in texts]


if __name__ == "__main__":
    ejemplo = "This Movie was AMAZING!!! I loved it, 10/10 https://example.com"
    print("Original :", ejemplo)
    print("Limpio   :", clean_text(ejemplo))
