from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "echo"
    model: str = "echo/offline"
    temperature: float = 0.0
    max_tokens: int = 2048


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int = 5
    use_dense: bool = False
    use_graph: bool = True
    graph_top_k: int = 8


@dataclass(frozen=True)
class Settings:
    profile: str = "local"
    dataset_path: Path = Path("../dataset/bt_dataset.json")
    artifact_dir: Path = Path("artifacts")
    run_dir: Path = Path("runs")
    llm: LLMSettings = LLMSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    repair_max_attempts: int = 1
    log_json: bool = False
    trace: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        prefix = "NAV4RAIL_GRAPHRAG_"
        return cls(
            profile=os.getenv(prefix + "PROFILE", "local"),
            dataset_path=Path(os.getenv(prefix + "DATASET_PATH", "../dataset/bt_dataset.json")),
            artifact_dir=Path(os.getenv(prefix + "ARTIFACT_DIR", "artifacts")),
            run_dir=Path(os.getenv(prefix + "RUN_DIR", "runs")),
            llm=LLMSettings(
                provider=os.getenv(prefix + "LLM_PROVIDER", "echo"),
                model=os.getenv(prefix + "LLM_MODEL", "echo/offline"),
                temperature=float(os.getenv(prefix + "LLM_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv(prefix + "LLM_MAX_TOKENS", "2048")),
            ),
            retrieval=RetrievalSettings(
                top_k=int(os.getenv(prefix + "RETRIEVAL_TOP_K", "5")),
                use_dense=_bool(prefix + "RETRIEVAL_USE_DENSE", False),
                use_graph=_bool(prefix + "RETRIEVAL_USE_GRAPH", True),
                graph_top_k=int(os.getenv(prefix + "RETRIEVAL_GRAPH_TOP_K", "8")),
            ),
            repair_max_attempts=int(os.getenv(prefix + "REPAIR_MAX_ATTEMPTS", "1")),
            log_json=_bool(prefix + "LOG_JSON", False),
            trace=_bool(prefix + "TRACE", False),
        )
