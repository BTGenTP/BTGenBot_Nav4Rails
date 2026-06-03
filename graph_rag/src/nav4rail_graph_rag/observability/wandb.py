from __future__ import annotations

from dataclasses import asdict
from typing import Any

from nav4rail_graph_rag.config import Settings, WandbSettings
from nav4rail_graph_rag.domain import GenerationResult


class WandbTracker:
    """Thin optional wrapper around W&B.

    Importing this module never requires wandb. The dependency is loaded only when
    tracking is enabled.
    """

    def __init__(self, settings: WandbSettings, config: dict[str, Any] | None = None) -> None:
        self.settings = settings
        self.enabled = settings.enabled
        self._wandb = None
        self._run = None
        if self.enabled:
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError(
                    "W&B tracking requires the optional 'observability' extra: "
                    "python3 -m pip install -e '.[observability]'"
                ) from exc
            self._wandb = wandb
            init_kwargs: dict[str, Any] = {
                "project": settings.project,
                "config": config or {},
                "tags": list(settings.tags),
            }
            if settings.entity:
                init_kwargs["entity"] = settings.entity
            if settings.group:
                init_kwargs["group"] = settings.group
            if settings.mode:
                init_kwargs["mode"] = settings.mode
            try:
                self._run = wandb.init(**init_kwargs)
            except Exception:
                # W&B should never make inference unusable. This commonly
                # happens on fresh machines when WANDB_API_KEY is absent.
                self.enabled = False
                self._wandb = None
                self._run = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "WandbTracker":
        return cls(
            settings.wandb,
            config={
                "profile": settings.profile,
                "llm": asdict(settings.llm),
                "retrieval": asdict(settings.retrieval),
                "repair_max_attempts": settings.repair_max_attempts,
            },
        )

    def log_generation(self, result: GenerationResult, prefix: str = "generation") -> None:
        if not self.enabled or self._wandb is None:
            return
        metrics = {
            f"{prefix}/l1_ok": int(result.validation.l1_ok),
            f"{prefix}/l2_ok": int(result.validation.l2_ok),
            f"{prefix}/l3_ok": int(result.validation.l3_ok),
            f"{prefix}/ok": int(result.validation.ok),
            f"{prefix}/n_nodes": result.validation.n_nodes,
            f"{prefix}/n_steps": len(result.steps),
            f"{prefix}/retrieved_skills": len(result.retrieval.candidate_skills),
            f"{prefix}/graph_skills": len(result.retrieval.graph_skills),
            f"{prefix}/issues": len(result.validation.issues),
        }
        self._wandb.log(metrics)

    def log_metrics(self, metrics: dict[str, Any], prefix: str = "eval") -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.log({f"{prefix}/{key}": value for key, value in metrics.items()})

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()


class NullTracker(WandbTracker):
    def __init__(self) -> None:
        self.enabled = False
        self.settings = WandbSettings()
        self._wandb = None
        self._run = None
