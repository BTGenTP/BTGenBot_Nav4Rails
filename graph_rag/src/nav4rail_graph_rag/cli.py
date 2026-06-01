from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nav4rail_graph_rag.config import Settings
from nav4rail_graph_rag.evaluation import run_eval
from nav4rail_graph_rag.generation import build_llm_client
from nav4rail_graph_rag.ingestion import load_dataset, save_artifacts
from nav4rail_graph_rag.observability import WandbTracker
from nav4rail_graph_rag.orchestration import build_pipeline

DEFAULT_EVAL_MISSIONS = [
    {
        "mission_id": "nav-01",
        "mission": "Navigate to the goal pose and follow the computed path.",
        "expected_skills": ["ComputePathToPose", "FollowPath"],
    },
    {
        "mission_id": "nav-02",
        "mission": "Spin in place to inspect the area, then back up a little.",
        "expected_skills": ["Spin", "BackUp"],
    },
]


def _build_from_dataset(dataset: Path, settings: Settings):
    dataset = _resolve_path(dataset)
    artifacts = load_dataset(dataset)
    return artifacts, build_pipeline(
        artifacts.records,
        artifacts.nodes,
        artifacts.edges,
        artifacts.patterns,
        llm=build_llm_client(settings.llm),
        tracker=WandbTracker.from_settings(settings),
    )


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(prog="nav4rail-graph-rag")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Build local JSONL artifacts from bt_dataset.json.")
    ingest.add_argument("--dataset", type=Path, default=settings.dataset_path)
    ingest.add_argument("--out", type=Path, default=settings.artifact_dir)
    gen = sub.add_parser("generate", help="Generate XML for one mission with the offline pipeline.")
    gen.add_argument("mission")
    gen.add_argument("--dataset", type=Path, default=settings.dataset_path)
    eval_cmd = sub.add_parser("eval", help="Run a small immutable evaluation batch.")
    eval_cmd.add_argument("--dataset", type=Path, default=settings.dataset_path)
    eval_cmd.add_argument("--missions", type=Path, help="JSON file containing mission dicts.")
    eval_cmd.add_argument("--out", type=Path, default=settings.run_dir)
    for command in (gen, eval_cmd):
        command.add_argument("--provider", choices=["echo", "litellm", "mistral", "anthropic", "openai"])
        command.add_argument("--model", help="LiteLLM model string, e.g. mistral/mistral-large-latest.")
    args = parser.parse_args(argv)
    if args.command == "ingest":
        artifacts = load_dataset(_resolve_path(args.dataset))
        save_artifacts(artifacts, args.out)
        print(json.dumps({"records": len(artifacts.records), "nodes": len(artifacts.nodes), "patterns": len(artifacts.patterns)}))
        return 0
    if args.command == "generate":
        settings = _settings_with_cli_overrides(settings, args)
        _, pipeline = _build_from_dataset(args.dataset, settings)
        result = pipeline.run(args.mission, top_k=settings.retrieval.top_k)
        metadata = getattr(pipeline.llm, "last_metadata", None)
        print(json.dumps({
            "provider": settings.llm.provider,
            "model": settings.llm.model,
            "llm": asdict(metadata) if metadata else None,
            "validation": asdict(result.validation),
            "steps": [asdict(step) for step in result.steps],
            "xml": result.xml,
        }, ensure_ascii=False, indent=2))
        pipeline.tracker.finish()
        return 0
    if args.command == "eval":
        settings = _settings_with_cli_overrides(settings, args)
        _, pipeline = _build_from_dataset(args.dataset, settings)
        missions = DEFAULT_EVAL_MISSIONS
        if args.missions:
            missions = json.loads(args.missions.read_text(encoding="utf-8"))
        run_path = run_eval(pipeline, missions, args.out)
        print(json.dumps({"run_path": str(run_path), "missions": len(missions)}))
        pipeline.tracker.finish()
        return 0
    return 2


def _resolve_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / path
    if candidate.exists():
        return candidate
    return path


def _settings_with_cli_overrides(settings: Settings, args) -> Settings:
    if not getattr(args, "provider", None) and not getattr(args, "model", None):
        return settings
    llm = settings.llm
    provider = args.provider or llm.provider
    model = args.model or ("echo/offline" if provider == "echo" else llm.model)
    return Settings(
        profile=settings.profile,
        dataset_path=settings.dataset_path,
        artifact_dir=settings.artifact_dir,
        run_dir=settings.run_dir,
        llm=type(llm)(
            provider=provider,
            model=model,
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
        ),
        retrieval=settings.retrieval,
        wandb=settings.wandb,
        repair_max_attempts=settings.repair_max_attempts,
        log_json=settings.log_json,
        trace=settings.trace,
    )
