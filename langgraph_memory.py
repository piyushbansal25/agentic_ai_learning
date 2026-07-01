from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import operator
import sqlite3
from dotenv import load_dotenv

load_dotenv()

# Step 1: Define State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# Step 2: Initialize Claude
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

# Step 3: Define a simple tool
@tool
def get_user_info(username: str) -> str:
    """Gets information about a user from the database.
    Args:
        username: The username to look up.
    """
    users = {
        "piyush": "Data Engineer at LTIMindtree, Pune. Learning Agentic AI.",
        "alice": "Data Scientist at TechCorp, Mumbai. Works on NLP.",
        "bob": "ML Engineer at StartupXYZ, Bangalore. Specializes in CV."
    }
    return users.get(username.lower(), f"User '{username}' not found.")

tools = [get_user_info]
llm_with_tools = llm.bind_tools(tools)

# Step 4: Define Nodes
def call_llm(state: AgentState) -> AgentState:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tool"
    return "end"

# Step 5: Build Graph
from langgraph.prebuilt import ToolNode

workflow = StateGraph(AgentState)
workflow.add_node("call_llm", call_llm)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("call_llm")
workflow.add_conditional_edges("call_llm", should_continue, {"use_tool": "tools", "end": END})
workflow.add_edge("tools", "call_llm")

# Step 6: Compile WITH a checkpointer (this is the key difference)
conn = sqlite3.connect("memory.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = workflow.compile(checkpointer=memory)

# Step 7: Run multiple conversations with same thread_id
thread_config = {"configurable": {"thread_id": "piyush_session_1"}}

print("=== Conversation 1 ===")
result1 = app.invoke(
    {"messages": [HumanMessage("Hi! Can you look up info about the user 'piyush'?")]},
    config=thread_config
)
print(result1["messages"][-1].content)

print("\n=== Conversation 2 (same thread — should remember) ===")
result2 = app.invoke(
    {"messages": [HumanMessage("Based on what you just found, what city is that user based in?")]},
    config=thread_config
)
print(result2["messages"][-1].content)

print("\n=== Conversation 3 (different thread — fresh start) ===")
new_thread_config = {"configurable": {"thread_id": "different_session"}}
result3 = app.invoke(
    {"messages": [HumanMessage("What city is piyush based in?")]},
    config=new_thread_config
)
print(result3["messages"][-1].content)



print("\n=== Conversation 4 (different thread — proves isolation) ===")
isolation_config = {"configurable": {"thread_id": "isolation_test"}}
result4 = app.invoke(
    {"messages": [HumanMessage("What was the last thing we talked about?")]},
    config=isolation_config
)
print(result4["messages"][-1].content)