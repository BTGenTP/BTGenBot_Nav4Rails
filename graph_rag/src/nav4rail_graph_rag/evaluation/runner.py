from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from nav4rail_graph_rag.evaluation.metrics import score_generation
from nav4rail_graph_rag.orchestration import GraphRAGPipeline


def run_eval(pipeline: GraphRAGPipeline, missions: list[dict], out_dir: Path | str) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path(out_dir) / run_id
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, item in enumerate(missions):
        result = pipeline.run(item["mission"])
        expected = tuple(item.get("expected_skills", ()))
        rows.append({
            "run_id": run_id,
            "mission_id": item.get("mission_id", f"mission-{index:03d}"),
            "mission": item["mission"],
            "metrics": score_generation(result, expected),
            "steps": [asdict(step) for step in result.steps],
            "issues": [asdict(issue) for issue in result.validation.issues],
            "xml": result.xml,
        })
    (out / "eval.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"run_id": run_id, "n": len(rows)}, indent=2), encoding="utf-8")
    return out
