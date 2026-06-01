from __future__ import annotations

from .domain import PortSpec, SkillSpec

CONTROL_NODES = {
    "AlwaysFailure", "AlwaysSuccess", "Fallback", "ForceFailure", "ForceSuccess",
    "Inverter", "KeepRunningUntilFailure", "Parallel", "PipelineSequence", "RateController",
    "ReactiveFallback", "ReactiveSequence", "RecoveryNode", "Repeat", "RetryUntilSuccessful",
    "RetryUntilSuccesful", "RoundRobin", "Sequence", "SequenceStar", "Timeout",
}

DEFAULT_SKILLS: dict[str, SkillSpec] = {
    "ComputePathToPose": SkillSpec("ComputePathToPose", "navigation", (PortSpec("goal"), PortSpec("path", "output"), PortSpec("planner_id", required=False, default="GridBased"))),
    "ComputePathThroughPoses": SkillSpec("ComputePathThroughPoses", "navigation", (PortSpec("goals"), PortSpec("path", "output"))),
    "FollowPath": SkillSpec("FollowPath", "navigation", (PortSpec("path"), PortSpec("controller_id", required=False, default="FollowPath"))),
    "NavigateToPose": SkillSpec("NavigateToPose", "navigation", (PortSpec("goal"),)),
    "NavigateThroughPoses": SkillSpec("NavigateThroughPoses", "navigation", (PortSpec("goals"),)),
    "ClearEntireCostmap": SkillSpec("ClearEntireCostmap", "recovery", (PortSpec("service_name"),)),
    "Spin": SkillSpec("Spin", "recovery", (PortSpec("spin_dist", required=False, default="1.57"),)),
    "Wait": SkillSpec("Wait", "utility", (PortSpec("wait_duration", required=False, default="1.0"),)),
    "BackUp": SkillSpec("BackUp", "recovery", (PortSpec("backup_dist", required=False, default="0.3"), PortSpec("backup_speed", required=False, default="0.15"))),
    "GoalUpdated": SkillSpec("GoalUpdated", "condition"),
    "GoalReached": SkillSpec("GoalReached", "condition"),
    "DistanceController": SkillSpec("DistanceController", "decorator", (PortSpec("distance", required=False, default="1.0"),)),
    "SpeedController": SkillSpec("SpeedController", "decorator", (PortSpec("min_rate"), PortSpec("max_rate"))),
    "SetBlackboard": SkillSpec("SetBlackboard", "blackboard", (PortSpec("output_key"), PortSpec("value"))),
    "RemovePassedGoals": SkillSpec("RemovePassedGoals", "navigation", (PortSpec("input_goals"), PortSpec("output_goals", "output"))),
    "TruncatePath": SkillSpec("TruncatePath", "navigation", (PortSpec("input_path"), PortSpec("output_path", "output"))),
}


def allowed_node_names() -> set[str]:
    return set(DEFAULT_SKILLS) | CONTROL_NODES | {"root", "BehaviorTree", "TreeNodesModel"}
