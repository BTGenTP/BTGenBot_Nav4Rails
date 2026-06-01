from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Protocol

from nav4rail_graph_rag.catalog import DEFAULT_SKILLS
from nav4rail_graph_rag.config import LLMSettings
from nav4rail_graph_rag.domain import Mission, RetrievalBundle, Step


@dataclass(frozen=True)
class LLMCallMetadata:
    provider: str
    model: str
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class LLMClient(Protocol):
    last_metadata: LLMCallMetadata | None

    def plan_steps(self, mission: Mission, retrieval: RetrievalBundle) -> tuple[Step, ...]: ...


class EchoLLMClient:
    """Deterministic offline planner used for tests and local development."""

    def __init__(self) -> None:
        self.last_metadata: LLMCallMetadata | None = LLMCallMetadata("echo", "echo/offline")

    def plan_steps(self, mission: Mission, retrieval: RetrievalBundle) -> tuple[Step, ...]:
        skills = list(retrieval.candidate_skills)
        text = mission.text.lower()
        steps: list[Step] = []
        if "wait" in text or "attendre" in text:
            steps.append(Step("Wait", {"wait_duration": "2.0"}, "explicit wait"))
        if "ComputePathThroughPoses" in skills or "waypoint" in text:
            steps.append(Step("ComputePathThroughPoses", {"goals": "{waypoints}", "path": "{path}"}))
            steps.append(Step("FollowPath", {"path": "{path}", "controller_id": "FollowPath"}))
        elif any(skill in skills for skill in ("ComputePathToPose", "NavigateToPose", "FollowPath")):
            steps.append(Step("ComputePathToPose", {"goal": "{goal}", "path": "{path}", "planner_id": "GridBased"}))
            steps.append(Step("FollowPath", {"path": "{path}", "controller_id": "FollowPath"}))
        if "spin" in text or "inspect" in text or "inspection" in text:
            steps.append(Step("Spin", {"spin_dist": "6.28"}, "inspection sweep"))
        if "backup" in text or "back up" in text or "recul" in text:
            steps.append(Step("BackUp", {"backup_dist": "0.3", "backup_speed": "0.15"}))
        if "recover" in text or "obstacle" in text or "stuck" in text or "bloque" in text:
            steps.append(Step("ClearEntireCostmap", {"service_name": "local_costmap/clear_entirely_local_costmap"}))
        seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        deduped: list[Step] = []
        for step in steps:
            key = (step.skill, tuple(sorted(step.params.items())))
            if key not in seen:
                seen.add(key)
                deduped.append(step)
        return tuple(deduped) if deduped else (Step("Wait", {"wait_duration": "1.0"}),)


class LiteLLMClient:
    """Provider-agnostic online client for Mistral, Anthropic, OpenAI and vLLM endpoints."""

    def __init__(self, settings: LLMSettings, completion_fn=None) -> None:
        self.settings = settings
        self._completion_fn = completion_fn
        self.last_metadata: LLMCallMetadata | None = None

    def plan_steps(self, mission: Mission, retrieval: RetrievalBundle) -> tuple[Step, ...]:
        completion = self._completion_fn or self._load_litellm_completion()
        messages = self._messages(mission, retrieval)
        start = time.perf_counter()
        response = completion(
            model=self.settings.model,
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )
        latency = time.perf_counter() - start
        content = _extract_content(response)
        steps = _parse_steps(content)
        self.last_metadata = _metadata_from_response(
            response=response,
            provider=self.settings.provider,
            model=self.settings.model,
            latency_s=latency,
        )
        return steps

    @staticmethod
    def _load_litellm_completion():
        try:
            from litellm import completion
        except ImportError as exc:
            raise RuntimeError(
                "Online inference requires the optional 'llm' extra: "
                "python3 -m pip install -e '.[llm]'"
            ) from exc
        return completion

    def _messages(self, mission: Mission, retrieval: RetrievalBundle) -> list[dict[str, str]]:
        allowed = [skill for skill in retrieval.candidate_skills if skill in DEFAULT_SKILLS]
        if not allowed:
            allowed = ["ComputePathToPose", "FollowPath", "Wait"]
        skill_docs = {
            name: [port.name for port in DEFAULT_SKILLS[name].ports]
            for name in allowed
            if name in DEFAULT_SKILLS
        }
        patterns = [pattern.signature for pattern in retrieval.patterns[:5]]
        system = (
            "You are a NAV4RAIL BehaviorTree planning component. "
            "Return only a JSON array of steps. Each step must be an object with "
            "'skill', 'params', and optional 'comment'. Do not return XML, markdown, "
            "or prose. Use only allowed skills and valid port names."
        )
        user = json.dumps(
            {
                "mission": mission.text,
                "allowed_skills": skill_docs,
                "retrieved_patterns": patterns,
                "output_schema": [{"skill": "SkillName", "params": {"port": "value"}, "comment": "optional"}],
            },
            ensure_ascii=False,
            indent=2,
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_llm_client(settings: LLMSettings) -> LLMClient:
    provider = settings.provider.lower()
    if provider == "echo":
        return EchoLLMClient()
    if provider in {"litellm", "mistral", "anthropic", "openai"}:
        return LiteLLMClient(settings)
    raise ValueError(f"Unsupported LLM provider: {settings.provider}")


def _extract_content(response) -> str:
    if isinstance(response, dict):
        return response["choices"][0]["message"]["content"]
    return response.choices[0].message.content


def _parse_steps(content: str) -> tuple[Step, ...]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    raw = json.loads(text)
    if not isinstance(raw, list):
        raise ValueError("LLM response must be a JSON array of steps.")
    steps: list[Step] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("skill"), str):
            raise ValueError("Each LLM step must contain a string 'skill'.")
        params = item.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("Step 'params' must be an object.")
        skill = item["skill"]
        if skill not in DEFAULT_SKILLS:
            raise ValueError(f"LLM returned a skill outside allowlist: {skill}")
        allowed_ports = {port.name for port in DEFAULT_SKILLS[skill].ports}
        unknown_ports = set(params) - allowed_ports
        if unknown_ports:
            raise ValueError(f"LLM returned unknown ports for {skill}: {sorted(unknown_ports)}")
        steps.append(Step(skill, {str(k): str(v) for k, v in params.items()}, item.get("comment")))
    return tuple(steps)


def _metadata_from_response(response, provider: str, model: str, latency_s: float) -> LLMCallMetadata:
    usage = getattr(response, "usage", None)
    if isinstance(response, dict):
        usage = response.get("usage", usage)
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens") or prompt_tokens + completion_tokens
    cost = getattr(response, "_hidden_params", {}).get("response_cost", 0.0)
    if isinstance(response, dict):
        cost = response.get("_hidden_params", {}).get("response_cost", cost)
    return LLMCallMetadata(
        provider=provider,
        model=model,
        latency_s=round(latency_s, 4),
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        total_tokens=int(total_tokens),
        cost_usd=float(cost or 0.0),
    )


def _usage_value(usage, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(key) or 0)
    return int(getattr(usage, key, 0) or 0)
