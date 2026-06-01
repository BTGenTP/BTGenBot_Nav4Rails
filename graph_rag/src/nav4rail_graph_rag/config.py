from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(env_path: Path | None = None) -> None:
    """Load a local .env file without adding a runtime dependency.

    Existing environment variables win over .env values, which keeps shell and
    CI secret injection authoritative.
    """

    candidates: list[Path] = []
    if env_path is not None:
        candidates.append(env_path)
    cwd = Path.cwd().resolve()
    candidates.extend([cwd / ".env", *[parent / ".env" for parent in cwd.parents]])
    here = Path(__file__).resolve()
    candidates.extend(parent / ".env" for parent in here.parents)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            os.environ.setdefault(key, value)
        return


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
class WandbSettings:
    enabled: bool = False
    project: str = "nav4rail-graphrag"
    entity: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    mode: str | None = None


@dataclass(frozen=True)
class Settings:
    profile: str = "local"
    dataset_path: Path = Path("../dataset/bt_dataset.json")
    artifact_dir: Path = Path("artifacts")
    run_dir: Path = Path("runs")
    llm: LLMSettings = LLMSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    wandb: WandbSettings = WandbSettings()
    repair_max_attempts: int = 1
    log_json: bool = False
    trace: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        prefix = "NAV4RAIL_GRAPHRAG_"
        provider = os.getenv(prefix + "LLM_PROVIDER", "echo")
        model = os.getenv(prefix + "LLM_MODEL") or _default_model_for_provider(provider)
        return cls(
            profile=os.getenv(prefix + "PROFILE", "local"),
            dataset_path=Path(os.getenv(prefix + "DATASET_PATH", "../dataset/bt_dataset.json")),
            artifact_dir=Path(os.getenv(prefix + "ARTIFACT_DIR", "artifacts")),
            run_dir=Path(os.getenv(prefix + "RUN_DIR", "runs")),
            llm=LLMSettings(
                provider=provider,
                model=model,
                temperature=float(os.getenv(prefix + "LLM_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv(prefix + "LLM_MAX_TOKENS", "2048")),
            ),
            retrieval=RetrievalSettings(
                top_k=int(os.getenv(prefix + "RETRIEVAL_TOP_K", "5")),
                use_dense=_bool(prefix + "RETRIEVAL_USE_DENSE", False),
                use_graph=_bool(prefix + "RETRIEVAL_USE_GRAPH", True),
                graph_top_k=int(os.getenv(prefix + "RETRIEVAL_GRAPH_TOP_K", "8")),
            ),
            wandb=WandbSettings(
                enabled=_bool(prefix + "WANDB_ENABLED", False),
                project=os.getenv(prefix + "WANDB_PROJECT", "nav4rail-graphrag"),
                entity=os.getenv(prefix + "WANDB_ENTITY") or None,
                group=os.getenv(prefix + "WANDB_GROUP") or None,
                tags=tuple(
                    tag.strip()
                    for tag in os.getenv(prefix + "WANDB_TAGS", "").split(",")
                    if tag.strip()
                ),
                mode=os.getenv(prefix + "WANDB_MODE") or None,
            ),
            repair_max_attempts=int(os.getenv(prefix + "REPAIR_MAX_ATTEMPTS", "1")),
            log_json=_bool(prefix + "LOG_JSON", False),
            trace=_bool(prefix + "TRACE", False),
        )


def _default_model_for_provider(provider: str) -> str:
    match provider.lower():
        case "mistral":
            return "mistral/mistral-large-latest"
        case "anthropic":
            return "anthropic/claude-3-5-sonnet-latest"
        case "openai":
            return "openai/gpt-4o"
        case "litellm":
            return "openai/gpt-4o"
        case _:
            return "echo/offline"
