from smolagents import CodeAgent,tool,LiteLLMModel,DuckDuckGoSearchTool
from dotenv import load_dotenv

load_dotenv()

@tool
def suggest_menu(occasion: str) -> str:
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


@tool
def catering_service_tool(query: str) -> str:
    """
    Returns the highest rated catering service in Gotham City.
    Args:
        query: A search term for finding catering services.
    """

    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }
    best_service = max(services, key=services.get)
    return best_service

@tool
def superhero_party_theme(category: str) -> str:
    """
    Suggests creative superhero themed party ideas based on a category.
    Args:
        category: The type of superhero party. Allowed values are:
                  'classic heroes', 'villain masquerade', 'futuristic gotham'.
    """
    themes = {
        "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes.",
        "villain masquerade": "Gotham Rogues Ball: A mysterious masquerade where guests dress as Batman villains.",
        "futuristic gotham": "Neo-Gotham Night: A cyberpunk party inspired by Batman Beyond with neon decorations."
    }
    return themes.get(category.lower(), "Try 'classic heroes', 'villain masquerade', or 'futuristic gotham'.")


model = LiteLLMModel(model_id="anthropic/claude-haiku-4-5-20251001")

agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        suggest_menu,
        catering_service_tool,
        superhero_party_theme
    ],
    model=model,
    max_steps=10
)

agent.run("We are having a villain masquerade party at Wayne's mansion. Suggest a theme, menu and the best catering service. Also search for good music recommendations for this type of party.")
