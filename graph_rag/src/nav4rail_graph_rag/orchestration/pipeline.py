from __future__ import annotations

from nav4rail_graph_rag.domain import GenerationResult, Mission
from nav4rail_graph_rag.generation import EchoLLMClient, LLMClient, render_bt_xml
from nav4rail_graph_rag.retrieval import HybridRetriever
from nav4rail_graph_rag.validation import validate_bt_xml


class GraphRAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm: LLMClient | None = None) -> None:
        self.retriever = retriever
        self.llm = llm or EchoLLMClient()

    def run(self, mission: str | Mission, top_k: int = 5) -> GenerationResult:
        mission_obj = mission if isinstance(mission, Mission) else Mission(str(mission))
        retrieval = self.retriever.retrieve(mission_obj.text, top_k=top_k)
        steps = self.llm.plan_steps(mission_obj, retrieval)
        tree_id = mission_obj.mission_id if mission_obj.mission_id != "ad-hoc" else "MainTree"
        xml = render_bt_xml(steps, tree_id=tree_id)
        validation = validate_bt_xml(xml)
        return GenerationResult(mission_obj, steps, xml, validation, retrieval)


def build_pipeline(records, nodes=(), edges=(), patterns=()) -> GraphRAGPipeline:
    from nav4rail_graph_rag.indexing import GraphStore, LexicalIndex

    lexical = LexicalIndex(tuple(records))
    graph = GraphStore(tuple(nodes), tuple(edges), tuple(patterns)) if nodes else None
    return GraphRAGPipeline(HybridRetriever(lexical, graph))
