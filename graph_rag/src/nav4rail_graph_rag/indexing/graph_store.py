from __future__ import annotations

from collections import Counter, defaultdict

from nav4rail_graph_rag.catalog import CONTROL_NODES, DEFAULT_SKILLS
from nav4rail_graph_rag.domain import BTEdge, BTNode, BTPattern


class GraphStore:
    def __init__(self, nodes: tuple[BTNode, ...], edges: tuple[BTEdge, ...], patterns: tuple[BTPattern, ...]) -> None:
        self.nodes = nodes
        self.edges = edges
        self.patterns = patterns
        self.records_by_skill: dict[str, set[int]] = defaultdict(set)
        self.skills_by_record: dict[int, set[str]] = defaultdict(set)
        self.patterns_by_skill: dict[str, Counter[str]] = defaultdict(Counter)
        self.cooccurrence: dict[str, Counter[str]] = defaultdict(Counter)
        self._build()

    def _build(self) -> None:
        structural = CONTROL_NODES | {"root", "BehaviorTree", "TreeNodesModel"}
        for node in self.nodes:
            if node.tag in DEFAULT_SKILLS or node.tag not in structural:
                self.records_by_skill[node.tag].add(node.record_id)
                self.skills_by_record[node.record_id].add(node.tag)
        for skills in self.skills_by_record.values():
            for skill in skills:
                for other in skills:
                    if other != skill:
                        self.cooccurrence[skill][other] += 1
        for pattern in self.patterns:
            for skill in DEFAULT_SKILLS:
                if skill in pattern.signature:
                    self.patterns_by_skill[skill][pattern.signature] += pattern.count

    def expand_skills(self, seeds: tuple[str, ...], top_k: int = 8) -> tuple[str, ...]:
        scores: Counter[str] = Counter()
        for seed in seeds:
            scores.update(self.cooccurrence.get(seed, Counter()))
        for seed in seeds:
            scores.pop(seed, None)
        return tuple(skill for skill, _ in scores.most_common(top_k))

    def related_patterns(self, seeds: tuple[str, ...], top_k: int = 5) -> tuple[BTPattern, ...]:
        scores: Counter[str] = Counter()
        for seed in seeds:
            scores.update(self.patterns_by_skill.get(seed, Counter()))
        return tuple(BTPattern(signature, count, "graph", 2) for signature, count in scores.most_common(top_k))
