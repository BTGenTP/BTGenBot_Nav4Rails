from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter

from nav4rail_graph_rag.domain import BTEdge, BTNode, BTPattern

_COMMENT = re.compile(r"<!--(.*?)-->", re.S)
_BLACKBOARD = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def sanitize_xml(xml_text: str) -> str:
    text = xml_text.strip().lstrip("\ufeff")
    text = re.sub(r"^<\?xml[^>]*>\s*", "", text)
    return _COMMENT.sub(lambda m: "<!--" + m.group(1).replace("--", "-") + "-->", text)


def parse_xml(xml_text: str) -> tuple[ET.Element | None, str, str | None]:
    try:
        return ET.fromstring(xml_text), "parsed", None
    except ET.ParseError:
        try:
            return ET.fromstring(sanitize_xml(xml_text)), "parsed_sanitized", None
        except ET.ParseError as exc:
            return None, "parse_error", str(exc)


def _node_name(element: ET.Element) -> str:
    if element.tag in {"Action", "Condition", "SubTree"} and "ID" in element.attrib:
        return element.attrib["ID"]
    return element.tag


def _signature(element: ET.Element, depth: int) -> str:
    name = _node_name(element)
    if depth <= 0 or not list(element):
        return name
    children = ", ".join(_signature(child, depth - 1) for child in list(element))
    return f"{name}({children})"


def blackboard_vars(attrs: dict[str, str]) -> tuple[str, ...]:
    found: set[str] = set()
    for value in attrs.values():
        found.update(_BLACKBOARD.findall(value))
    return tuple(sorted(found))


def extract_graph(record_id: int, root: ET.Element, pattern_depth: int = 2) -> tuple[list[BTNode], list[BTEdge], list[BTPattern]]:
    nodes: list[BTNode] = []
    edges: list[BTEdge] = []
    patterns: Counter[str] = Counter()

    def visit(element: ET.Element, depth: int, path: str, parent: str | None) -> None:
        attrs = {str(k): str(v) for k, v in element.attrib.items()}
        nodes.append(BTNode(record_id, path, _node_name(element), depth, attrs))
        if parent is not None:
            edges.append(BTEdge(record_id, parent, path))
        children = list(element)
        if children:
            patterns[_signature(element, pattern_depth)] += 1
        for index, child in enumerate(children):
            visit(child, depth + 1, f"{path}/{index}:{_node_name(child)}", path)

    visit(root, 0, _node_name(root), None)
    return nodes, edges, [BTPattern(sig, count, "structural", pattern_depth) for sig, count in patterns.items()]
