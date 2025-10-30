import asyncio
import shutil
import textwrap
from typing import Any, Dict, List, TypedDict, Annotated
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


DEFAULT_BASE_URL = "http://localhost:10000"


class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


async def _call_server_via_a2a(query: str, base_url: str = DEFAULT_BASE_URL) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)

        payload: Dict[str, Any] = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": query}],
                "message_id": uuid4().hex,
            }
        }
        req = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**payload))
        resp = await client.send_message(req)
        return resp.model_dump(mode="json", exclude_none=True)


def _extract_text_from_a2a(resp: Dict[str, Any]) -> str:
    """Extract readable text from common A2A response shapes.

    Tries, in order:
    1) result.parts[].root.text
    2) result.artifacts[].parts[].root.text
    3) Generic recursive search for any 'text' fields likely to be content
    """

    def _collect_texts_from_parts(parts: Any) -> List[str]:
        texts: List[str] = []
        if isinstance(parts, list):
            for p in parts:
                if isinstance(p, dict):
                    root = p.get("root", {}) if isinstance(p.get("root"), dict) else {}
                    text = root.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text)
        return texts

    # 1) Root result parts
    try:
        parts = resp["root"]["result"]["parts"]
        texts = _collect_texts_from_parts(parts)
        if texts:
            return "\n".join(texts).strip()
    except Exception:
        pass

    # 2) Artifacts parts (e.g., name='result')
    try:
        artifacts = resp["root"]["result"].get("artifacts")
        if isinstance(artifacts, list):
            collected: List[str] = []
            for art in artifacts:
                if isinstance(art, dict) and "parts" in art:
                    collected.extend(_collect_texts_from_parts(art.get("parts")))
            if collected:
                return "\n".join(collected).strip()
    except Exception:
        pass

    # 3) Generic recursive fallback
    def _collect_texts_generic(node: Any, bucket: List[str]) -> None:
        if isinstance(node, dict):
            # Prefer 'root.text' shape
            if "root" in node and isinstance(node["root"], dict):
                txt = node["root"].get("text")
                if isinstance(txt, str) and txt.strip():
                    bucket.append(txt)
            # Also capture direct 'text' if it looks like content
            txt2 = node.get("text")
            if isinstance(txt2, str) and len(txt2.strip()) > 0:
                bucket.append(txt2)
            for v in node.values():
                _collect_texts_generic(v, bucket)
        elif isinstance(node, list):
            for item in node:
                _collect_texts_generic(item, bucket)

    generic_bucket: List[str] = []
    _collect_texts_generic(resp, generic_bucket)
    if generic_bucket:
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: List[str] = []
        for t in generic_bucket:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return "\n".join(deduped).strip()

    # Fallback to raw JSON string if nothing found
    return str(resp)


async def agent_node(state: AgentState) -> AgentState:
    # Find the latest human message content
    user_text = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_text = msg.content
            break

    a2a_resp = await _call_server_via_a2a(user_text)
    answer_text = _extract_text_from_a2a(a2a_resp)
    return {"messages": [AIMessage(content=answer_text)]}


def build_graph() -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()


async def run(query: str) -> str:
    compiled = build_graph()
    result: AgentState = await compiled.ainvoke({
        "messages": [HumanMessage(content=query)]
    })
    last = result["messages"][-1]
    return getattr(last, "content", str(last))


def _print_pretty(text: str) -> None:
    try:
        term_width = max(60, min(120, shutil.get_terminal_size().columns))
    except Exception:
        term_width = 100
    sep = "-" * term_width
    title = " Agent Response "
    pad = max(0, (term_width - len(title)) // 2)
    header = f"{'=' * pad}{title}{'=' * (term_width - len(title) - pad)}"

    # Wrap each paragraph while preserving blank lines
    paragraphs = [p.strip() for p in text.strip().split("\n\n")]
    wrapped_parts: List[str] = []
    for p in paragraphs:
        lines = [ln.rstrip() for ln in p.splitlines()]
        if any(ln.startswith("    ") or ln.startswith("\t") for ln in lines):
            # Likely code block; print as-is
            wrapped_parts.append("\n".join(lines))
        else:
            wrapped_parts.append(textwrap.fill(" ".join(lines), width=term_width))
    body = "\n\n".join(wrapped_parts)

    print(f"\n{header}\n{sep}\n{body}\n{sep}\n")


if __name__ == "__main__":
    # Interactive CLI: ensure the A2A server is running: uv run python -m app
    print("A2A LangGraph client. Type 'q' to quit.")
    while True:
        try:
            query = input("Question> ").strip()
        except EOFError:
            break
        if not query:
            continue
        if query.lower() in {"q", "quit", "exit"}:
            break
        try:
            answer = asyncio.run(run(query))
            _print_pretty(answer)
        except Exception as e:
            print(f"Error: {e}\n")


