from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from xml.dom import minidom

from nav4rail_graph_rag.catalog import DEFAULT_SKILLS
from nav4rail_graph_rag.domain import Step


def _safe_id(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip())[:64].strip("_")
    return cleaned or "MainTree"


def _apply_defaults(step: Step) -> dict[str, str]:
    params = dict(step.params)
    spec = DEFAULT_SKILLS.get(step.skill)
    if spec:
        for port in spec.ports:
            if port.default is not None and port.name not in params:
                params[port.name] = port.default
    return params


def render_bt_xml(steps: tuple[Step, ...], tree_id: str = "MainTree") -> str:
    tree_id = _safe_id(tree_id)
    root = ET.Element("root", {"BTCPP_format": "4", "main_tree_to_execute": tree_id})
    behavior_tree = ET.SubElement(root, "BehaviorTree", {"ID": tree_id})
    sequence = ET.SubElement(behavior_tree, "Sequence", {"name": "MissionSequence"})
    recovery_steps = [step for step in steps if step.skill == "ClearEntireCostmap"]
    action_steps = [step for step in steps if step.skill != "ClearEntireCostmap"]
    parent = sequence
    recovery_node = None
    if recovery_steps:
        recovery_node = ET.SubElement(sequence, "RecoveryNode", {"name": "MissionRecovery", "number_of_retries": "2"})
        parent = ET.SubElement(recovery_node, "Sequence", {"name": "PrimaryMission"})
    for step in action_steps:
        ET.SubElement(parent, step.skill, _apply_defaults(step))
    if recovery_node is not None:
        for step in recovery_steps:
            ET.SubElement(recovery_node, step.skill, _apply_defaults(step))
    rough = ET.tostring(root, encoding="unicode")
    return minidom.parseString(rough).toprettyxml(indent="  ").replace('<?xml version="1.0" ?>\n', "")
