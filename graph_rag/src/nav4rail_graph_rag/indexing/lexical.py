from __future__ import annotations

import math
import re
from collections import Counter

from nav4rail_graph_rag.domain import BTRecord

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(tok.lower() for tok in _TOKEN.findall(text))


class LexicalIndex:
    def __init__(self, records: tuple[BTRecord, ...]) -> None:
        self.records = records
        self.doc_terms: dict[int, Counter[str]] = {}
        df: Counter[str] = Counter()
        for record in records:
            terms = Counter(tokenize(record.mission + " " + " ".join(record.tags)))
            self.doc_terms[record.record_id] = terms
            df.update(terms.keys())
        n_docs = max(len(records), 1)
        self.idf = {term: math.log((n_docs + 1) / (freq + 0.5)) + 1 for term, freq in df.items()}

    def search(self, query: str, top_k: int = 5) -> tuple[tuple[BTRecord, float], ...]:
        q_terms = Counter(tokenize(query))
        scores: list[tuple[BTRecord, float]] = []
        for record in self.records:
            if not record.parsed:
                continue
            terms = self.doc_terms.get(record.record_id, Counter())
            score = sum(freq * terms.get(term, 0) * self.idf.get(term, 0.0) for term, freq in q_terms.items())
            if score > 0:
                scores.append((record, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return tuple(scores[:top_k])
