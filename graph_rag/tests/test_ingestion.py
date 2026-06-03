from nav4rail_graph_rag.ingestion.xml_parser import extract_graph, parse_xml


def test_parse_and_extract_graph() -> None:
    xml = """
    <root BTCPP_format="4" main_tree_to_execute="MainTree">
      <BehaviorTree ID="MainTree">
        <Sequence><Wait wait_duration="1.0"/></Sequence>
      </BehaviorTree>
    </root>
    """
    root, status, error = parse_xml(xml)
    assert root is not None
    assert status == "parsed"
    assert error is None
    nodes, edges, patterns = extract_graph(0, root)
    assert [node.tag for node in nodes][-1] == "Wait"
    assert edges
    assert patterns
