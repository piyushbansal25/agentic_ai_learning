from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import operator
from dotenv import load_dotenv

load_dotenv()

@tool
def get_weather(location: str) -> str:
    """Gets the current weather for a given location.
    Args:
        location: The city to get weather for.
    """
    weather_data = {
        "pune": "Sunny, 34°C, humidity 45%",
        "mumbai": "Cloudy, 30°C, humidity 80%",
        "delhi": "Hazy, 38°C, humidity 30%",
        "bangalore": "Pleasant, 26°C, humidity 60%"
    }
    return weather_data.get(location.lower(), f"Weather data not available for {location}")


@tool
def calculate(expression: str) -> str:
    """Evaluates a basic mathematical expression.
    Args:
        expression: A math expression like '100 * 0.18' or '5 + 3'.
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


tools = [get_weather, calculate]
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
llm_with_tools = llm.bind_tools(tools)


def call_llm(state: AgentState) -> AgentState:
    """Node: calls Claude with tools available"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """Edge: checks if Claude wants to use a tool or is done"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tool"
    return "end"



workflow = StateGraph(AgentState)


workflow.add_node("call_llm", call_llm)
workflow.add_node("tools", ToolNode(tools)) 


workflow.set_entry_point("call_llm")


workflow.add_conditional_edges(
    "call_llm",
    should_continue,
    {
        "use_tool": "tools",  # Claude wants a tool → run it
        "end": END            # Claude is done → finish
    }
)


workflow.add_edge("tools", "call_llm")

app = workflow.compile()

print("=== LangGraph ReAct Agent ===\n")
result = app.invoke({
    "messages": [HumanMessage(content="What's the weather in Pune and Mumbai? Also calculate 34 * 1.15 for me.")]
})

print("=== FINAL RESPONSE ===")
print(result["messages"][-1].content)