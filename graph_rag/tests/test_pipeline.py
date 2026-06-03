from nav4rail_graph_rag.domain import BTRecord
from nav4rail_graph_rag.orchestration import build_pipeline


def test_offline_pipeline_generates_valid_xml() -> None:
    record = BTRecord(0, "", "Navigate to goal and follow path", "<root />", "parsed", tags=("ComputePathToPose", "FollowPath"))
    pipeline = build_pipeline((record,))
    result = pipeline.run("Navigate to the goal pose and follow the path")
    assert result.validation.ok, result.validation.issues
    assert "ComputePathToPose" in result.xml
