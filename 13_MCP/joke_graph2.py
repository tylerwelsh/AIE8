from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
from langchain_openai import ChatOpenAI
from langgraph import graph as lg
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage
import asyncio


class JokeGraph:
    def __init__(self):
        load_dotenv()
        if os.getenv("TAVILY_API_KEY") is None:
            raise ValueError("TAVILY_API_KEY is not set")
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OPENAI_API_KEY is not set")
        
        self._mcp_client = MultiServerMCPClient(
            {
                "mcp-server": {
                    "command": "uv",
                    "args": ["--directory", ".", "run", "server.py"],
                    "transport": "stdio"
                }
            }
        )
        self._llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
        self._graph = None

    async def initialize_graph(self):
        tools = await self._mcp_client.get_tools()
        print(f"Tools: {tools}")
        self._llm = self._llm.bind_tools(tools)
        self._tools = tools  # Store tools for use in ToolNode

        def call_model(state: lg.MessagesState):
            """Call LLM with funny system prompt and bind tools"""
            system_message = SystemMessage(
                content="You are a hilarious, witty assistant who always responds in a joking, humorous, and funny way. Make every interaction entertaining with clever jokes, puns, and witty remarks. Keep the humor lighthearted and fun! First, call the get_joke tool with joke_type: christmas. After receiving the tool result, compose the final answer to the user and append that joke. If the tool returns an error, say the tool failed and do not fabricate a joke. You should respond in format: 'answer: <answer> joke: <joke>'. A tool call to get_joke iis requiired."
            )
            # Only add system message if it's not already in the messages
            if not any(isinstance(m, SystemMessage) for m in state["messages"]):
                messages = [system_message] + state["messages"]
            else:
                messages = state["messages"]
            resp = self._llm.invoke(messages)
            return {"messages": [resp]}

        uncompiled_graph = lg.StateGraph(lg.MessagesState)
        uncompiled_graph.add_node("call_model", call_model)
        uncompiled_graph.add_node("tools", ToolNode(self._tools))
        uncompiled_graph.set_entry_point("call_model")
        uncompiled_graph.add_conditional_edges("call_model", tools_condition)
        uncompiled_graph.add_edge("tools", "call_model")
        self._graph = uncompiled_graph.compile()
        print("Graph compiled successfully")


async def main():
    print("Starting joke graph application...")
    joke_graph = JokeGraph()
    await joke_graph.initialize_graph()
    print("--------------------------------")
    while True:
        q = input("Ask me a q, get a funny answer ('q' to end): ")
        if q == "q":
            break
        # result = joke_graph._graph.invoke({"messages": [HumanMessage(content=q)]})
        result = await joke_graph._graph.ainvoke({"messages": [HumanMessage(content=q)]})
        print("\n")
        print(result["messages"][-1].content)
        print("--------------------------------\n")

if __name__ == "__main__":
    asyncio.run(main())