from smolagents import CodeAgent, tool, DuckDuckGoSearchTool, VisitWebpageTool, LiteLLMModel
from dotenv import load_dotenv

load_dotenv()

model = LiteLLMModel(model_id="anthropic/claude-haiku-4-5-20251001")

@tool
def calculate_average(numbers : list) -> str:
    """Calculates the average of a list of numbers
    Args:
        numbers: A list of numeric values to average.
    """
    if not numbers:
        return "No numbers provided"
    
    avg = sum(numbers)/ len(numbers)
    return f"Average: {avg:.2f} (from {len(numbers)} data points: {numbers})"


@tool
def calculate_growth_rate(old_value: float, new_value: float) -> str:
    """Calculates percentage growth rate between two values.
    Args:
        old_value: The starting/older value.
        new_value: The ending/newer value.
    """
    if old_value == 0:
        return "Cannot calculate growth rate from zero."
    growth = ((new_value - old_value) / old_value) * 100
    return f"Growth rate: {growth:.2f}%"


web_agent = CodeAgent(
    model=model,
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    name="web_research_agent",
    description="Searches the web and visits pages to find specific information and data points.",
    max_steps=8
)

data_agent = CodeAgent(
    model=model,
    tools=[calculate_average, calculate_growth_rate],
    name="data_analyst_agent",
    description="Performs calculations and numerical analysis on data points, such as averages and growth rates.",
    max_steps=5
)

manager_agent = CodeAgent(
    model=model,
    tools=[],
    managed_agents=[web_agent, data_agent],
    max_steps=10
)


result = manager_agent.run(
    "Research the current market size of the AI agents industry. "
    "Find at least 3 data points/estimates from different sources. "
    "Then use the data analyst agent to calculate the average market size estimate. "
    "Summarize your findings in a clear report."
)

print("\n\n=== FINAL REPORT ===")
print(result)