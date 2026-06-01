from nav4rail_graph_rag.domain import Step
from nav4rail_graph_rag.generation import render_bt_xml
from nav4rail_graph_rag.validation import validate_bt_xml


def test_rendered_xml_validates() -> None:
    xml = render_bt_xml((
        Step("ComputePathToPose", {"goal": "{goal}", "path": "{path}", "planner_id": "GridBased"}),
        Step("FollowPath", {"path": "{path}", "controller_id": "FollowPath"}),
    ))
    report = validate_bt_xml(xml)
    assert report.ok, report.issues


def test_unknown_skill_fails_l3() -> None:
    xml = '<root BTCPP_format="4" main_tree_to_execute="MainTree"><BehaviorTree ID="MainTree"><Sequence><FakeSkill /></Sequence></BehaviorTree></root>'
    report = validate_bt_xml(xml)
    assert report.l1_ok
    assert report.l2_ok
    assert not report.l3_ok
