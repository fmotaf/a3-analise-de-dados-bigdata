import re
from collections import Counter
from typing import List, Tuple

import nltk
from nltk.corpus import stopwords


def ensure_nltk():
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


def limpar_texto(texto: str) -> List[str]:
    texto = texto.lower()
    palavras = re.findall(r"[a-zA-Z']+", texto)
    ensure_nltk()
    stop_en = set(stopwords.words("english"))
    tokens = [p for p in palavras if p not in stop_en and len(p) > 2]
    return tokens


def termos_mais_frequentes(textos, top_n: int = 30) -> List[Tuple[str, int]]:
    todos_tokens: List[str] = []
    for t in textos:
        if not isinstance(t, str):
            continue
        todos_tokens.extend(limpar_texto(t))
    contagem = Counter(todos_tokens)
    return contagem.most_common(top_n)
