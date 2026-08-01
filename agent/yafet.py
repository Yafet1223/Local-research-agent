"""
PROJECT: Personal Research & Notes Assistant
=============================================

A ReAct-style LangGraph agent that can:
  - search the web (mocked — no API key needed)
  - do real arithmetic
  - save notes to long-term memory (persists across separate conversations)
  - recall notes from long-term memory

This is the standard "agent" shape you'll meet in almost every real LangGraph
project: an LLM node bound to tools, a tool-execution node, and a conditional
edge that loops between them until the LLM stops requesting tools.

    ┌──────┐  has tool_calls?  ┌───────┐
    │agent │ ────────────────► │ tools │
    └──┬───┘ ◄──────────────── └───────┘
       │        no tool_calls
       ▼
      END

Requires:
    pip install langgraph langchain-ollama --break-system-packages

Make sure Ollama is running locally and you've pulled a tool-calling-capable model:
    ollama pull llama3.1
    (or: qwen2.5, llama3.2, llama3-groq-tool-use)

Run it:
    python assistant.py
"""

import ast
import operator
import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_store, get_config
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------------------------------------------------------
# TOOLS
# Each @tool becomes something the LLM can choose to call. get_store() and
# get_config() give tools access to the graph's store/config at runtime —
# no special injection wiring needed.
# ---------------------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """Search the web for information on a topic. Use this for facts,
    current events, or anything you don't already know."""
    # Mocked — swap this out for a real search API (e.g. Tavily) later.
    return f"[mock search result for '{query}']: LangGraph is a graph-based agent framework by LangChain."


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (4 + 3)'."""
    try:
        result = _safe_eval(ast.parse(expression, mode="eval").body)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def save_note(note: str) -> str:
    """Save a piece of information to long-term memory so it can be
    recalled in future, separate conversations."""
    store = get_store()
    config = get_config()
    user_id = config["configurable"].get("user_id", "anonymous")
    key = f"note-{len(store.search(('user', user_id, 'notes')))}"
    store.put(("user", user_id, "notes"), key, {"text": note})
    return f"Saved note: {note}"


@tool
def recall_notes(query: str) -> str:
    """Search previously saved notes for relevant information."""
    store = get_store()
    config = get_config()
    user_id = config["configurable"].get("user_id", "anonymous")
    results = store.search(("user", user_id, "notes"), query=query)
    if not results:
        return "No relevant notes found."
    return "\n".join(f"- {r.value['text']}" for r in results)


TOOLS = [web_search, calculator, save_note, recall_notes]


# ---------------------------------------------------------------------------
# AGENT NODE
# ---------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful personal research assistant. "
    "Use tools when they'd help: search for facts, calculator for math, "
    "save_note to remember things the user wants kept, recall_notes to "
    "check what you already know about the user before answering."
))

def agent(state: MessagesState) -> dict:
    response = llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# BUILD THE GRAPH
# ---------------------------------------------------------------------------
graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(TOOLS))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # -> "tools" or END, built in
graph.add_edge("tools", "agent")  # after running tools, loop back to the agent

checkpointer = InMemorySaver()  # short-term: remembers this conversation
store = InMemoryStore()         # long-term: remembers notes across conversations

app = graph.compile(checkpointer=checkpointer, store=store)


# ---------------------------------------------------------------------------
# RUN IT — simple CLI chat loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Personal Research & Notes Assistant. Type 'quit' to exit.\n")
    print("Try: 'What's 47 * 12?' / 'Remember that I prefer metric units' / "
          "'What do you know about my preferences?'\n")

    config = {"configurable": {"thread_id": "cli-session", "user_id": "you"}}

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ("quit", "exit"):
            break

        result = app.invoke({"messages": [("user", user_input)]}, config=config)
        print("Assistant:", result["messages"][-1].content, "")