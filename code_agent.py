from smolagents import CodeAgent, tool, LiteLLMModel
from dotenv import load_dotenv

load_dotenv()

@tool
def suggest_menu(occasion : str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion: The type of occasion for the party. Allowed values are:
                  'casual', 'formal', 'superhero', 'custom'.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."


model = LiteLLMModel(model_id="anthropic/claude-haiku-4-5-20251001")

agent = CodeAgent(tools =[suggest_menu], model=model)

agent.run("Prepare a formal menu for the party.")