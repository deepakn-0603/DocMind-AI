import pytest
from langchain_core.messages import HumanMessage

from app.langgraph import chat_nodes


@pytest.mark.asyncio
async def test_question_answering_agent_returns_default_when_context_is_missing(monkeypatch):
    async def fake_query_context(*args, **kwargs):
        return []

    class DummyLLM:
        async def ainvoke(self, *args, **kwargs):
            raise AssertionError("LLM should not be called when context is missing")

    monkeypatch.setattr(chat_nodes, "query_context", fake_query_context)
    monkeypatch.setattr(chat_nodes, "_chat_llm", DummyLLM())

    state = {"messages": [HumanMessage(content="What is Deepak's role?")]}
    result = await chat_nodes.question_answering_agent(state)

    assert result["answer"] == "I'm not sure based on the available information."
    assert result["context_chunks"] == 0
