"""Separação conservadora de alegações compostas para pesquisa independente."""

from __future__ import annotations

import re


_VERBS = (
    "morreu", "faleceu", "nasceu", "puxou", "disse", "afirmou", "aprovou",
    "cancelou", "ganhou", "perdeu", "foi", "será", "sera", "vai", "está",
    "esta", "tem", "teve", "fez", "matou", "comprou", "vendeu", "recebeu",
)
_VERB_PATTERN = re.compile(r"\b(?:" + "|".join(_VERBS) + r")\b", re.IGNORECASE)
_INTRO = re.compile(r"^\s*(?:é verdade que|e verdade que|será que|sera que)\s+", re.IGNORECASE)


def split_claims(message: str) -> list[str]:
    """Divide só duas orações explícitas; casos ambíguos permanecem inteiros."""
    plain = _INTRO.sub("", message.strip()).rstrip(" ?.!")
    if plain.count(" e ") != 1:
        return [message.strip()]
    left, right = (part.strip() for part in plain.split(" e ", 1))
    first_verb = _VERB_PATTERN.search(left)
    second_verb = _VERB_PATTERN.match(right)
    if not first_verb or not second_verb:
        return [message.strip()]
    subject = left[:first_verb.start()].strip()
    if len(subject.split()) < 2 or len(subject.split()) > 14:
        return [message.strip()]
    return [left + ".", f"{subject} {right}."]


def ambiguous_compound(message: str) -> bool:
    """Reconhece duas ações aparentes que não conseguimos dividir com segurança."""
    return bool(re.search(r"\be\b", message, re.IGNORECASE) and
                len(_VERB_PATTERN.findall(message)) >= 2 and
                len(split_claims(message)) == 1)
