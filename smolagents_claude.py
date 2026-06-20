from smolagents import ToolCallingAgent, tool
from smolagents import LiteLLMModel
import datetime
import pytz
from dotenv import load_dotenv

load_dotenv()

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """

    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


@tool 
def calculator(a: float, b: float, operation: str) -> str:
    """A tool that performs a basic operation
    Args:
        a: the first number
        b: the second number
        operation: the operation to perform - must be 'add', 'subtract', 'multiply', or 'divide'
    """
    if operation == "add":
        return str(a + b)
    elif operation == "subtract":
        return str(a - b)
    elif operation == "multiply":
        return str(a * b)
    elif operation == "divide":
        if b == 0:
            return "Error: cannot divide by zero"
        return str(a / b)
    else:
        return "Error: operation must be add, subtract, multiply, or divide"


model = LiteLLMModel(model_id="anthropic/claude-haiku-4-5-20251001")


agent = ToolCallingAgent(
    tools= [get_current_time_in_timezone, calculator],
    model= model
)

response = agent.run("What time is it in Tokyo right now? Also what is 1337 multiplied by 42?")
print(response)






