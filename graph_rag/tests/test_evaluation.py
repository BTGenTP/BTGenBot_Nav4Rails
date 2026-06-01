from nav4rail_graph_rag.domain import BTRecord
from nav4rail_graph_rag.evaluation import run_eval
from nav4rail_graph_rag.orchestration import build_pipeline


def test_eval_runner_writes_immutable_run(tmp_path) -> None:
    record = BTRecord(
        0,
        "",
        "Navigate to goal and follow path",
        "<root />",
        "parsed",
        tags=("ComputePathToPose", "FollowPath"),
    )
    pipeline = build_pipeline((record,))
    out = run_eval(
        pipeline,
        [{"mission": "Navigate to goal and follow path", "expected_skills": ["ComputePathToPose"]}],
        tmp_path,
    )
    assert (out / "eval.jsonl").exists()
    assert (out / "metrics.json").exists()
