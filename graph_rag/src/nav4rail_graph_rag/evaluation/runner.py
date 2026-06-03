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
    aggregate = {
        "n": 0,
        "l1_ok": 0,
        "l2_ok": 0,
        "l3_ok": 0,
        "ok": 0,
        "skill_recall_sum": 0.0,
        "hallucinated_skills": 0,
    }
    for index, item in enumerate(missions):
        result = pipeline.run(item["mission"])
        expected = tuple(item.get("expected_skills", ()))
        metrics = score_generation(result, expected)
        aggregate["n"] += 1
        aggregate["l1_ok"] += int(metrics["l1_ok"])
        aggregate["l2_ok"] += int(metrics["l2_ok"])
        aggregate["l3_ok"] += int(metrics["l3_ok"])
        aggregate["ok"] += int(metrics["ok"])
        aggregate["skill_recall_sum"] += float(metrics["skill_recall"])
        aggregate["hallucinated_skills"] += int(metrics["hallucinated_skills"])
        rows.append({
            "run_id": run_id,
            "mission_id": item.get("mission_id", f"mission-{index:03d}"),
            "mission": item["mission"],
            "metrics": metrics,
            "steps": [asdict(step) for step in result.steps],
            "issues": [asdict(issue) for issue in result.validation.issues],
            "xml": result.xml,
        })
    n = max(int(aggregate["n"]), 1)
    summary = {
        "run_id": run_id,
        "n": aggregate["n"],
        "l1_rate": round(aggregate["l1_ok"] / n, 4),
        "l2_rate": round(aggregate["l2_ok"] / n, 4),
        "l3_rate": round(aggregate["l3_ok"] / n, 4),
        "ok_rate": round(aggregate["ok"] / n, 4),
        "skill_recall_mean": round(aggregate["skill_recall_sum"] / n, 4),
        "hallucinated_skills": aggregate["hallucinated_skills"],
    }
    (out / "eval.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pipeline.tracker.log_metrics(summary, prefix="eval")
    return out
