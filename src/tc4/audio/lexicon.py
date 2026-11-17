"""Léxico local de termos críticos (pt-BR e inglês), complementar ao Azure Text Analytics.

O Azure identifica entidades clínicas e sentimento; o léxico garante que expressões que a
equipe considera de alerta (ex.: "falta de ar") sejam sempre sinalizadas e explicáveis.
"""
from __future__ import annotations

import re
import unicodedata

from ..models import ATENCAO, CRITICO, INFO

# termo -> (grupo, severidade)
CRITICAL_TERMS: dict[str, tuple[str, int]] = {
    "falta de ar": ("respiratorio", CRITICO),
    "nao consigo respirar": ("respiratorio", CRITICO),
    "dificuldade para respirar": ("respiratorio", CRITICO),
    "cansaco para respirar": ("respiratorio", CRITICO),
    "chiado": ("respiratorio", ATENCAO),
    "tosse": ("respiratorio", INFO),
    "dor no peito": ("cardiovascular", CRITICO),
    "aperto no peito": ("cardiovascular", CRITICO),
    "palpitacao": ("cardiovascular", ATENCAO),
    "coracao acelerado": ("cardiovascular", ATENCAO),
    "desmaio": ("neurologico", CRITICO),
    "desmaiei": ("neurologico", CRITICO),
    "tontura": ("neurologico", ATENCAO),
    "confusao": ("neurologico", ATENCAO),
    "fala enrolada": ("neurologico", CRITICO),
    "formigamento": ("neurologico", ATENCAO),
    "cansaco": ("fadiga", ATENCAO),
    "cansada": ("fadiga", ATENCAO),
    "cansado": ("fadiga", ATENCAO),
    "fraqueza": ("fadiga", ATENCAO),
    "exausta": ("fadiga", ATENCAO),
    "exausto": ("fadiga", ATENCAO),
    "sem forca": ("fadiga", ATENCAO),
    "febre": ("infeccioso", ATENCAO),
    "calafrio": ("infeccioso", ATENCAO),
    "vermelhidao": ("ferida", ATENCAO),
    "secrecao": ("ferida", ATENCAO),
    "sangramento": ("hemorragico", CRITICO),
    "inchaco no braco": ("linfedema", ATENCAO),
    "braco inchado": ("linfedema", ATENCAO),
    "inchado": ("linfedema", ATENCAO),
    "inchada": ("linfedema", ATENCAO),
    "inchaco": ("linfedema", ATENCAO),
    "vermelho": ("ferida", ATENCAO),
    "vermelha": ("ferida", ATENCAO),
    "dor forte": ("dor", ATENCAO),
    "muita dor": ("dor", ATENCAO),
    "eu cai": ("queda", CRITICO),
    "levei um tombo": ("queda", CRITICO),
    "cai no banheiro": ("queda", CRITICO),
    "queda": ("queda", ATENCAO),
    # inglês (áudios públicos)
    "shortness of breath": ("respiratorio", CRITICO),
    "can't breathe": ("respiratorio", CRITICO),
    "cannot breathe": ("respiratorio", CRITICO),
    "trouble breathing": ("respiratorio", CRITICO),
    "wheezing": ("respiratorio", ATENCAO),
    "cough": ("respiratorio", INFO),
    "chest pain": ("cardiovascular", CRITICO),
    "palpitations": ("cardiovascular", ATENCAO),
    "dizzy": ("neurologico", ATENCAO),
    "fainted": ("neurologico", CRITICO),
    "slurred": ("neurologico", CRITICO),
    "tired": ("fadiga", ATENCAO),
    "fatigue": ("fadiga", ATENCAO),
    "exhausted": ("fadiga", ATENCAO),
    "weakness": ("fadiga", ATENCAO),
    "fever": ("infeccioso", ATENCAO),
    "bleeding": ("hemorragico", CRITICO),
    "swelling": ("linfedema", ATENCAO),
    "severe pain": ("dor", ATENCAO),
    "i fell": ("queda", CRITICO),
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text)


_NORMALIZED = {normalize(k): (k, *v) for k, v in CRITICAL_TERMS.items()}


def find_critical_terms(text: str) -> list[dict]:
    norm = normalize(text or "")
    hits = []
    for term, (original, group, severity) in _NORMALIZED.items():
        pattern = r"(?<![\w])" + re.escape(term) + r"(?![\w])"
        count = len(re.findall(pattern, norm))
        if count:
            hits.append({"term": original, "group": group, "severity": severity,
                         "count": count})
    # evita contar duas vezes o mesmo trecho ("braço inchado" já cobre "inchado")
    norm_terms = {h["term"]: normalize(h["term"]) for h in hits}
    hits = [h for h in hits if not any(
        o is not h and o["group"] == h["group"] and norm_terms[h["term"]] != norm_terms[o["term"]]
        and norm_terms[h["term"]] in norm_terms[o["term"]] for o in hits)]
    return sorted(hits, key=lambda h: (-h["severity"], h["term"]))
