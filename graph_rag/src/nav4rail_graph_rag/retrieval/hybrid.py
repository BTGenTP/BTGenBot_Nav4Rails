from __future__ import annotations

import re

from nav4rail_graph_rag.catalog import DEFAULT_SKILLS
from nav4rail_graph_rag.domain import BTPattern, RetrievalBundle
from nav4rail_graph_rag.indexing import GraphStore, LexicalIndex

_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ComputePathThroughPoses", ("waypoint", "waypoints", "sequence of poses", "through")),
    ("ComputePathToPose", ("path", "planner", "goal", "pose", "aller", "navigate", "naviguer")),
    ("FollowPath", ("follow", "suivre", "path", "chemin")),
    ("NavigateToPose", ("dock", "depot", "dépôt", "delivery", "livraison")),
    ("Spin", ("spin", "rotate", "inspect", "inspection", "tourner")),
    ("Wait", ("wait", "attendre", "seconds", "secondes")),
    ("BackUp", ("backup", "back up", "reculer")),
    ("ClearEntireCostmap", ("clear", "costmap", "obstacle", "stuck", "bloque")),
    ("RecoveryNode", ("recover", "recovery", "retry", "fallback", "obstacle")),
    ("DistanceController", ("every meter", "1 meter", "mètre", "metre")),
)


def infer_skills(text: str) -> tuple[str, ...]:
    low = text.lower()
    found: list[str] = []
    for skill, needles in _KEYWORDS:
        if any(needle in low for needle in needles) and skill not in found:
            found.append(skill)
    for skill in DEFAULT_SKILLS:
        if re.search(rf"\b{re.escape(skill.lower())}\b", low) and skill not in found:
            found.append(skill)
    if not found:
        found.extend(["ComputePathToPose", "FollowPath"])
    if "ComputePathToPose" in found and "FollowPath" not in found and "NavigateToPose" not in found:
        found.append("FollowPath")
    return tuple(found)


class HybridRetriever:
    def __init__(self, lexical: LexicalIndex, graph: GraphStore | None = None) -> None:
        self.lexical = lexical
        self.graph = graph

    def retrieve(self, mission: str, top_k: int = 5) -> RetrievalBundle:
        lexical_hits = self.lexical.search(mission, top_k=top_k)
        examples = tuple(record for record, _ in lexical_hits)
        seeds = infer_skills(mission)
        graph_skills: tuple[str, ...] = ()
        patterns: tuple[BTPattern, ...] = ()
        if self.graph is not None:
            graph_skills = self.graph.expand_skills(seeds)
            patterns = self.graph.related_patterns(seeds)
        skills = tuple(dict.fromkeys([*seeds, *graph_skills]))
        return RetrievalBundle(examples, skills, patterns, graph_skills, sum(score for _, score in lexical_hits))
