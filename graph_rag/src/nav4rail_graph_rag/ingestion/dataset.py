from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from nav4rail_graph_rag.domain import BTEdge, BTNode, BTPattern, BTRecord
from nav4rail_graph_rag.ingestion.xml_parser import extract_graph, parse_xml


@dataclass(frozen=True)
class IngestionArtifacts:
    records: tuple[BTRecord, ...]
    nodes: tuple[BTNode, ...]
    edges: tuple[BTEdge, ...]
    patterns: tuple[BTPattern, ...]


def load_dataset(path: Path | str) -> IngestionArtifacts:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records: list[BTRecord] = []
    nodes: list[BTNode] = []
    edges: list[BTEdge] = []
    pattern_counts: dict[tuple[str, str, int], int] = {}
    for record_id, raw in enumerate(data):
        root, status, error = parse_xml(raw.get("output", ""))
        record_nodes: list[BTNode] = []
        tags: tuple[str, ...] = ()
        max_depth = 0
        if root is not None:
            record_nodes, record_edges, record_patterns = extract_graph(record_id, root)
            nodes.extend(record_nodes)
            edges.extend(record_edges)
            tags = tuple(node.tag for node in record_nodes)
            max_depth = max((node.depth for node in record_nodes), default=0)
            for pattern in record_patterns:
                key = (pattern.signature, pattern.scope, pattern.depth)
                pattern_counts[key] = pattern_counts.get(key, 0) + pattern.count
        records.append(BTRecord(record_id, raw.get("instruction", ""), raw.get("input", ""), raw.get("output", ""), status, error, len(record_nodes), max_depth, tags))
    patterns = tuple(BTPattern(sig, count, scope, depth) for (sig, scope, depth), count in pattern_counts.items())
    return IngestionArtifacts(tuple(records), tuple(nodes), tuple(edges), patterns)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def save_artifacts(artifacts: IngestionArtifacts, out_dir: Path | str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "records.jsonl", [asdict(row) for row in artifacts.records])
    _write_jsonl(out / "nodes.jsonl", [asdict(row) for row in artifacts.nodes])
    _write_jsonl(out / "edges.jsonl", [asdict(row) for row in artifacts.edges])
    _write_jsonl(out / "patterns.jsonl", [asdict(row) for row in artifacts.patterns])
