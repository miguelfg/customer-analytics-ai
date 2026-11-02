"""Agent tests with a scripted fake Anthropic client (no network)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from customer_analytics import agent, config


def tool_use(id_, name, input_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def reply(*blocks, stop="tool_use"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop)


class FakeClient:
    """``client.messages.create`` returns scripted responses and records each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def tool_results(request):
    """``tool_result`` blocks of the last user message of a request."""
    return [b for b in request["messages"][-1]["content"] if b["type"] == "tool_result"]


@pytest.fixture(autouse=True)
def charts_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHARTS_DIR", tmp_path / "charts")


def test_ask_executes_tools_and_collects_charts(scored):
    client = FakeClient(
        [
            reply(tool_use("t1", "top_at_risk", {"ltv_min": 0, "limit": 5})),
            reply(tool_use("t2", "render_chart", {"name": "segment-sizes"})),
            reply(SimpleNamespace(type="text", text="Send a win-back offer."), stop="end_turn"),
        ]
    )
    out = agent.ask("Who is at risk?", scored, client=client, model="m")
    assert [c["name"] for c in out["tool_calls"]] == ["top_at_risk", "render_chart"]
    assert len(out["charts"]) == 1 and Path(out["charts"][0]).exists()
    assert out["answer"] == "Send a win-back offer."
    assert client.requests[0]["system"] == agent.SYSTEM_PROMPT
    assert client.requests[0]["tools"] == agent.TOOLS


def test_tool_error_returned_not_raised(scored):
    client = FakeClient(
        [
            reply(tool_use("t1", "get_customer", {"customer_id": 999999})),
            reply(SimpleNamespace(type="text", text="Not found."), stop="end_turn"),
        ]
    )
    out = agent.ask("Tell me about 999999", scored, client=client, model="m")
    (result,) = tool_results(client.requests[1])
    assert result["is_error"] is True
    assert "999999 not found" in result["content"]
    assert out["answer"] == "Not found."


def test_tools_schema_names_match_dispatch():
    names = [t["name"] for t in agent.TOOLS]
    assert sorted(names) == sorted(agent.DISPATCH)
    assert len(names) == 7


def test_eval_questions_reference_known_tools():
    items = agent.load_evals()
    assert len(items) == 10
    assert all(set(i["expected_tools"]) <= set(agent.DISPATCH) for i in items)


def test_eval_pass_rule():
    hit = {"tool_calls": [{"name": "top_at_risk"}, {"name": "render_chart"}], "charts": ["a"]}
    assert agent.eval_passed({"expected_tools": ["top_at_risk", "render_chart"]}, hit)
    assert not agent.eval_passed({"expected_tools": ["render_chart"]}, {**hit, "charts": []})
    assert not agent.eval_passed({"expected_tools": ["list_segments"]}, hit)
