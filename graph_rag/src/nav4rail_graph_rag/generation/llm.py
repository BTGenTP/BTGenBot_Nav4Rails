from __future__ import annotations

from typing import Protocol

from nav4rail_graph_rag.domain import Mission, RetrievalBundle, Step


class LLMClient(Protocol):
    def plan_steps(self, mission: Mission, retrieval: RetrievalBundle) -> tuple[Step, ...]: ...


class EchoLLMClient:
    """Deterministic offline planner used for tests and local development."""

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
