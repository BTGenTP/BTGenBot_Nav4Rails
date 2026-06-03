from nav4rail_graph_rag.config import LLMSettings
from nav4rail_graph_rag.domain import Mission, RetrievalBundle
from nav4rail_graph_rag.generation.llm import LiteLLMClient


def test_litellm_client_parses_json_steps() -> None:
    def fake_completion(**kwargs):
        assert kwargs["model"] == "mistral/mistral-large-latest"
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '[{"skill":"ComputePathToPose",'
                            '"params":{"goal":"{goal}","path":"{path}","planner_id":"GridBased"}},'
                            '{"skill":"FollowPath","params":{"path":"{path}","controller_id":"FollowPath"}}]'
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "_hidden_params": {"response_cost": 0.001},
        }

    client = LiteLLMClient(
        LLMSettings(provider="mistral", model="mistral/mistral-large-latest"),
        completion_fn=fake_completion,
    )
    steps = client.plan_steps(
        Mission("Navigate to the goal pose."),
        RetrievalBundle(candidate_skills=("ComputePathToPose", "FollowPath")),
    )
    assert [step.skill for step in steps] == ["ComputePathToPose", "FollowPath"]
    assert client.last_metadata is not None
    assert client.last_metadata.total_tokens == 30
