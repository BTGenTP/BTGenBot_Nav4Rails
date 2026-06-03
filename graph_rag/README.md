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
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
python -m nav4rail_graph_rag ingest --dataset ../dataset/bt_dataset.json --out artifacts
python -m nav4rail_graph_rag generate "Patrouiller la section 4, inspecter, puis revenir au depot."
```

If you do not install the package, use `PYTHONPATH=src` from this directory:

```bash
cd repositories/BTGenBot_Nav4Rails/graph_rag
PYTHONPATH=src python3 -m nav4rail_graph_rag ingest --dataset ../dataset/bt_dataset.json --out artifacts
```

## Online Inference

Install the optional online dependencies and expose provider keys through the environment:

```bash
python3 -m pip install -e ".[llm,observability]"
export MISTRAL_API_KEY=...
export NAV4RAIL_GRAPHRAG_LLM_PROVIDER=mistral
export NAV4RAIL_GRAPHRAG_LLM_MODEL=mistral/mistral-large-latest
python3 -m nav4rail_graph_rag generate "Navigate to the goal pose and follow the path."
```

Supported provider values are `mistral`, `anthropic`, `openai`, and `litellm`. The model string is passed to LiteLLM, so OpenAI-compatible local/self-hosted backends can be added later by setting the usual LiteLLM environment variables. A local `.env` file is loaded automatically; real keys should stay out of git.

## W&B Tracking

```bash
export WANDB_API_KEY=...
export NAV4RAIL_GRAPHRAG_WANDB_ENABLED=true
export NAV4RAIL_GRAPHRAG_WANDB_PROJECT=nav4rail-graphrag
PYTHONPATH=src python3 -m nav4rail_graph_rag eval --dataset ../dataset/bt_dataset.json --out runs
```

W&B logs validation rates, retrieved skill counts, LLM latency, token usage and estimated cost when the provider exposes it.

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
