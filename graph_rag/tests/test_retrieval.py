from nav4rail_graph_rag.domain import BTRecord
from nav4rail_graph_rag.indexing import GraphStore, LexicalIndex
from nav4rail_graph_rag.retrieval import HybridRetriever, infer_skills


def test_infer_navigation_skills() -> None:
    skills = infer_skills("Navigate to a goal and follow the computed path")
    assert "ComputePathToPose" in skills
    assert "FollowPath" in skills


def test_hybrid_retriever_returns_examples() -> None:
    record = BTRecord(0, "", "Navigate and follow path", "<root />", "parsed", tags=("ComputePathToPose", "FollowPath"))
    retriever = HybridRetriever(LexicalIndex((record,)), GraphStore((), (), ()))
    bundle = retriever.retrieve("follow path", top_k=1)
    assert bundle.examples[0].record_id == 0
    assert "FollowPath" in bundle.candidate_skills
