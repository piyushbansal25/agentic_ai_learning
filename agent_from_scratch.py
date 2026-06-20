from anthropic import Anthropic
from dotenv import load_dotenv
import json

load_dotenv()

client = Anthropic()

SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

ALWAYS use the following format:

Thought: you should always think about what to do
Action:
```json
{
  "action": "get_weather",
  "action_input": {"location": "London"}
}
```
Observation: the result of the action

Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

def get_weather(loaction):
    return f"The weather in {loaction} is sunny with low temperature"

messages = [
    {"role" : "user", "content" : "What's the weather in London?"}
]

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=300,
    system=SYSTEM_PROMPT,
    messages=messages,
    stop_sequences=["Observation:"]
)

llm_output = response.content[0].text
print("=== LLM Output ===")
print(llm_output)

if "get_weather" in llm_output:
    weather_result = get_weather("London")
    print("\n=== Tool Result ===")
    print(weather_result)

    messages.append({"role": "assistant", "content": llm_output})
    messages.append({"role": "user", "content": f"Observation: {weather_result}\nNow give the Final Answer."})

    final_response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=messages
    )


    print("\n=== Final Answer ===")
    print(final_response.content[0].text)

