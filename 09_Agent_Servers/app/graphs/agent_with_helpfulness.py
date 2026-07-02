from __future__ import annotations

import os
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.models import get_chat_model
from app.tools import get_tool_belt

SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in feline (cat) health. "
    "Use the retrieve_information tool for cat-health questions, web search for "
    "current information, and Arxiv for research papers. Cite tool results when "
    "they inform your answer. "
    "Format responses in clear GitHub-flavored Markdown with short headings, "
    "bullet points where helpful, and concise actionable guidance."
)

MAX_HELPFULNESS_RETRIES = 2


class GraphState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    attempt_count: int
    helpfulness_passed: bool
    helpfulness_reason: str


class HelpfulnessVerdict(BaseModel):
    helpful: bool = Field(
        description="Whether the assistant response is clear, accurate, and useful for the user query."
    )
    reason: str = Field(description="Short reason for the verdict.")


agent = create_agent(
    model=get_chat_model(),
    tools=get_tool_belt(),
    system_prompt=SYSTEM_PROMPT,
)

judge = get_chat_model(
    model_name=os.environ.get("OPENAI_JUDGE_MODEL", os.environ.get("OPENAI_CHAT_MODEL")),
    temperature=0,
).with_structured_output(HelpfulnessVerdict)


def _run_agent(state: GraphState) -> GraphState:
    prior_messages = list(state.get("messages", []))
    result = agent.invoke({"messages": prior_messages})
    new_messages = result["messages"][len(prior_messages) :]
    return {
        "messages": new_messages,
        "attempt_count": state.get("attempt_count", 0) + 1,
    }


def _judge_helpfulness(state: GraphState) -> GraphState:
    messages = state.get("messages", [])

    latest_user_message = ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            latest_user_message = str(message.content)
            break

    latest_ai_message = ""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            latest_ai_message = str(message.content)
            break

    verdict = judge.invoke(
        [
            (
                "system",
                "You evaluate whether a response is helpful. Mark helpful=true only if the answer is relevant, accurate, and actionable.",
            ),
            (
                "human",
                (
                    f"User request:\n{latest_user_message}\n\n"
                    f"Assistant response:\n{latest_ai_message}"
                ),
            ),
        ]
    )

    return {
        "helpfulness_passed": verdict.helpful,
        "helpfulness_reason": verdict.reason,
    }


def _add_retry_feedback(state: GraphState) -> GraphState:
    reason = state.get("helpfulness_reason", "The previous response was not sufficiently helpful.")
    return {
        "messages": [
            HumanMessage(
                content=(
                    "Please improve your previous answer. "
                    f"Judge feedback: {reason} "
                    "Provide a concise, directly useful response grounded in available tools and evidence."
                )
            )
        ]
    }


def _route_after_judging(state: GraphState) -> str:
    if state.get("helpfulness_passed", False):
        return "end"
    if state.get("attempt_count", 0) >= MAX_HELPFULNESS_RETRIES:
        return "end"
    return "retry"


builder = StateGraph(GraphState)
builder.add_node("run_agent", _run_agent)
builder.add_node("judge_helpfulness", _judge_helpfulness)
builder.add_node("add_retry_feedback", _add_retry_feedback)

builder.add_edge(START, "run_agent")
builder.add_edge("run_agent", "judge_helpfulness")
builder.add_conditional_edges(
    "judge_helpfulness",
    _route_after_judging,
    {
        "end": END,
        "retry": "add_retry_feedback",
    },
)
builder.add_edge("add_retry_feedback", "run_agent")

graph = builder.compile()
