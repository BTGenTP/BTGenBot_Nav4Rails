from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Mission:
    text: str
    mission_id: str = "ad-hoc"
    locale: str = "auto"


@dataclass(frozen=True)
class PortSpec:
    name: str
    direction: Literal["input", "output", "inout"] = "input"
    required: bool = True
    default: str | None = None


@dataclass(frozen=True)
class SkillSpec:
    name: str
    category: str = "action"
    ports: tuple[PortSpec, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class BTNode:
    record_id: int
    node_id: str
    tag: str
    depth: int
    attrs: dict[str, str] = field(default_factory=dict)

    @property
    def skill_name(self) -> str:
        if self.tag in {"Action", "Condition", "SubTree"} and "ID" in self.attrs:
            return self.attrs["ID"]
        return self.tag


@dataclass(frozen=True)
class BTEdge:
    record_id: int
    source: str
    target: str
    relation: str = "child"


@dataclass(frozen=True)
class BTPattern:
    signature: str
    count: int = 1
    scope: str = "structural"
    depth: int = 2


@dataclass(frozen=True)
class BTRecord:
    record_id: int
    instruction: str
    mission: str
    xml: str
    parse_status: str
    parse_error: str | None = None
    n_nodes: int = 0
    max_depth: int = 0
    tags: tuple[str, ...] = ()

    @property
    def parsed(self) -> bool:
        return self.parse_error is None and self.parse_status.startswith("parsed")


@dataclass(frozen=True)
class Step:
    skill: str
    params: dict[str, str] = field(default_factory=dict)
    comment: str | None = None


@dataclass(frozen=True)
class RetrievalBundle:
    examples: tuple[BTRecord, ...] = ()
    candidate_skills: tuple[str, ...] = ()
    patterns: tuple[BTPattern, ...] = ()
    graph_skills: tuple[str, ...] = ()
    score: float = 0.0


@dataclass(frozen=True)
class ValidationIssue:
    level: Literal["L1", "L2", "L3"]
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    l1_ok: bool
    l2_ok: bool
    l3_ok: bool
    issues: tuple[ValidationIssue, ...] = ()
    n_nodes: int = 0

    @property
    def ok(self) -> bool:
        return self.l1_ok and self.l2_ok and self.l3_ok


@dataclass(frozen=True)
class GenerationResult:
    mission: Mission
    steps: tuple[Step, ...]
    xml: str
    validation: ValidationReport
    retrieval: RetrievalBundle


@dataclass(frozen=True)
class PipelineTrace:
    run_id: str
    mission: Mission
    events: tuple[dict[str, Any], ...] = ()
