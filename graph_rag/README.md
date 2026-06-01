# NAV4RAIL GraphRAG

Training-free GraphRAG scaffold for generating Nav2 / BehaviorTree.CPP v4 XML from operator missions.

The default path is intentionally conservative:

1. Extract a strict intent / step plan from a natural-language mission.
2. Retrieve validated BT examples, skills, blackboard variables and recurring patterns.
3. Render XML deterministically from a closed skill catalogue.
4. Validate with L1 XML syntax, L2 BT structure and L3 semantic checks.
5. Write immutable run artifacts for evaluation.

This package does not require cloud credentials for the offline MVP. Cloud LLM calls can be added behind the `LLMClient` interface.

## Quick Start

```bash
cd repositories/BTGenBot_Nav4Rails/graph_rag
python -m pytest -q
python -m nav4rail_graph_rag ingest --dataset ../dataset/bt_dataset.json --out artifacts
python -m nav4rail_graph_rag generate "Patrouiller la section 4, inspecter, puis revenir au depot."
```

## Project Shape

- `src/nav4rail_graph_rag/domain`: typed domain models.
- `src/nav4rail_graph_rag/ingestion`: dataset loading, XML parsing and graph extraction.
- `src/nav4rail_graph_rag/indexing`: lexical and local graph indexes.
- `src/nav4rail_graph_rag/retrieval`: hybrid retrieval.
- `src/nav4rail_graph_rag/orchestration`: training-free pipeline.
- `src/nav4rail_graph_rag/generation`: provider abstraction and deterministic XML renderer.
- `src/nav4rail_graph_rag/validation`: L1/L2/L3 validators.
- `src/nav4rail_graph_rag/evaluation`: benchmark runner and metrics.
- `docs/architecture.md`: detailed technical architecture and audit.

## Security

Do not place real API keys in this repository. Use `.env.example` as documentation only and provide secrets through the process environment or a secret manager.
