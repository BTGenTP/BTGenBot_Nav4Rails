from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from nav4rail_graph_rag.catalog import CONTROL_NODES, DEFAULT_SKILLS, allowed_node_names
from nav4rail_graph_rag.domain import ValidationIssue, ValidationReport
from nav4rail_graph_rag.ingestion.xml_parser import parse_xml

_BLACKBOARD = re.compile(r"\{[^{}]+\}")


def _effective_name(element: ET.Element) -> str:
    if element.tag in {"Action", "Condition", "SubTree"} and "ID" in element.attrib:
        return element.attrib["ID"]
    return element.tag


def _iter(root: ET.Element):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(list(node)))


def validate_bt_xml(xml: str, strict_blackboard: bool = True) -> ValidationReport:
    issues: list[ValidationIssue] = []
    root, _, error = parse_xml(xml)
    if root is None:
        issue = ValidationIssue("L1", "XML_PARSE_ERROR", error or "XML parse error")
        return ValidationReport(False, False, False, (issue,), 0)
    l2_ok = True
    if root.tag != "root":
        issues.append(ValidationIssue("L2", "ROOT_TAG", "Expected <root>."))
        l2_ok = False
    trees = root.findall("BehaviorTree")
    if not trees:
        issues.append(ValidationIssue("L2", "MISSING_BEHAVIOR_TREE", "Expected at least one <BehaviorTree>."))
        l2_ok = False
    for tree in trees:
        if "ID" not in tree.attrib:
            issues.append(ValidationIssue("L2", "MISSING_TREE_ID", "BehaviorTree requires ID."))
            l2_ok = False
        if len(list(tree)) != 1:
            issues.append(ValidationIssue("L2", "TREE_ARITY", "BehaviorTree should expose one root child."))
            l2_ok = False
    l3_ok = True
    allowed = allowed_node_names()
    n_nodes = 0
    for element in _iter(root):
        n_nodes += 1
        name = _effective_name(element)
        if name not in allowed and element.tag not in allowed:
            issues.append(ValidationIssue("L3", "UNKNOWN_SKILL", f"Node '{name}' is not in the allowlist."))
            l3_ok = False
        if name in DEFAULT_SKILLS:
            for port in DEFAULT_SKILLS[name].ports:
                if port.required and port.name not in element.attrib:
                    issues.append(ValidationIssue("L3", "MISSING_PORT", f"{name} requires port '{port.name}'."))
                    l3_ok = False
        if element.tag in CONTROL_NODES and element.tag not in {"AlwaysFailure", "AlwaysSuccess"} and len(list(element)) == 0:
            issues.append(ValidationIssue("L2", "EMPTY_CONTROL", f"{element.tag} should have children."))
            l2_ok = False
        if strict_blackboard:
            for key, value in element.attrib.items():
                if ("{" in value or "}" in value) and not _BLACKBOARD.fullmatch(value):
                    issues.append(ValidationIssue("L3", "BLACKBOARD_SYNTAX", f"Invalid blackboard value in {key}."))
                    l3_ok = False
    return ValidationReport(True, l2_ok, l3_ok, tuple(issues), n_nodes)
