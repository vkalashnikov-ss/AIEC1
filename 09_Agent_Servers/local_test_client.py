import asyncio

from langgraph_sdk import get_client


async def main() -> None:
    # Build an SDK client pointed at the locally running LangGraph server.
    #
    # Why localhost:2024?
    # - This is the default URL used in this session for the dev server.
    # - Keeping it explicit makes it obvious where requests are going during testing.
    #
    # If the server is not running yet, this script will fail to connect.
    client = get_client(url="http://localhost:2024")

    # Stream execution events from the graph instead of waiting for one final response.
    #
    # Why stream?
    # - It lets us observe progress incrementally (useful for debugging graph steps).
    # - We can inspect intermediate updates to verify tool calls/state transitions.
    #
    # Positional args:
    # - None: no assistant/agent ID namespace is provided (use default setup).
    # - "simple_agent": the graph/assistant name defined in langgraph.json.
    #
    # input:
    # - messages follows chat format expected by the graph.
    # - We provide one human prompt to trigger a single run.
    #
    # stream_mode="updates": request step-by-step update chunks.
    async for chunk in client.runs.stream(
        None,
        "simple_agent",
        input={"messages": [{"role": "human", "content": "How often should I deworm my cat?"}]},
        stream_mode="updates",
    ):
        # Print each streamed chunk as it arrives so we can inspect runtime behavior.
        # In a richer client, this could be parsed and rendered in a UI instead.
        print(chunk)


if __name__ == "__main__":
    # Run the async entry point from a normal Python script context.
    # asyncio.run() creates and manages the event loop for this one-shot test.
    asyncio.run(main())
