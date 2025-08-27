import re
import numpy as np

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|<.*?>", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize_basic(text: str) -> str:
    rules = [
        (r"(ing|ed|ly|ness|able|ible|ment|tion|s)$", ""),
    ]
    for pat, rep in rules:
        text = re.sub(pat, rep, text)
    return text

def explain_top_terms(text, cleaned, vectorizer, y_index, class_names, lr_explainer=None, k=6):
    x = vectorizer.transform([cleaned])
    feature_names = np.array(vectorizer.get_feature_names_out())

    if lr_explainer is not None and hasattr(lr_explainer, "coef_"):
        coef = lr_explainer.coef_[y_index]
        contrib = x.toarray().ravel() * coef
        top_idx = np.argsort(contrib)[::-1][:k]
        terms = []
        for i in top_idx:
            if x[0, i] > 0:
                terms.append({"term": feature_names[i], "score": float(contrib[i])})
        return terms[:k]

    arr = x.toarray().ravel()
    top_idx = np.argsort(arr)[::-1][:k]
    terms = []
    for i in top_idx:
        if arr[i] > 0:
            terms.append({"term": feature_names[i], "score": float(arr[i])})
    return terms[:k]
