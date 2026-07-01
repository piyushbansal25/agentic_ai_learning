from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
import operator
from dotenv import load_dotenv

load_dotenv()

# Step 1: Define State
# This is the shared object that flows through every node
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # messages accumulate (add new to existing)
    step_count: int

# Step 2: Initialize Claude
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

# Step 3: Define Nodes (each node is just a function)
def call_llm(state: AgentState) -> AgentState:
    """Node 1: Send messages to Claude, get response"""
    print(f"\n[Node: call_llm] Step {state['step_count']}")
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "step_count": state["step_count"] + 1
    }

def should_continue(state: AgentState) -> str:
    """Conditional edge: decide what happens next"""
    last_message = state["messages"][-1]
    print(f"[Edge: should_continue] Checking last message...")
    # If the response contains a question, loop back for another round
    if "?" in last_message.content and state["step_count"] < 3:
        print("[Edge] → Looping back to call_llm")
        return "continue"
    else:
        print("[Edge] → Going to END")
        return "end"

# Step 4: Build the Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("call_llm", call_llm)

# Set entry point
workflow.set_entry_point("call_llm")

# Add conditional edges
workflow.add_conditional_edges(
    "call_llm",        # from this node
    should_continue,   # use this function to decide
    {
        "continue": "call_llm",  # if "continue" → loop back
        "end": END               # if "end" → finish
    }
)

# Step 5: Compile the graph
app = workflow.compile()

# Step 6: Run it
print("=== LangGraph Basic Agent ===\n")
initial_state = {
    "messages": [HumanMessage(content="Tell me one interesting fact about AI agents. Then ask me a follow-up question about it.")],
    "step_count": 1
}

result = app.invoke(initial_state)

print("\n=== CONVERSATION ===")
for msg in result["messages"]:
    role = "Human" if isinstance(msg, HumanMessage) else "Claude"
    print(f"\n{role}: {msg.content}")

print(f"\nTotal steps: {result['step_count']}")