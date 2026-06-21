from smolagents import CodeAgent, LiteLLMModel
from dotenv import load_dotenv
from PIL import Image
import requests
from io import BytesIO


load_dotenv()

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Extract%2C_Transform%2C_Load_Data_Flow_Diagram.svg/1920px-Extract%2C_Transform%2C_Load_Data_Flow_Diagram.svg.png"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

response = requests.get(image_url, headers = headers)
image = Image.open(BytesIO(response.content)).convert("RGB")

model = LiteLLMModel(model_id="anthropic/claude-haiku-4-5-20251001")

agent = CodeAgent(
    tools=[],
    model=model,
    max_steps=5
)

response = agent.run(
    "Look at this data pipeline architecture diagram. Describe the components you see, "
    "explain the data flow from source to destination, and identify what kind of "
    "architecture pattern this represents (e.g., ETL, ELT, Medallion, etc.)",
    images=[image]
)
print(response)