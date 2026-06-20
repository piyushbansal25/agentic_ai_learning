from smolagents import CodeAgent, LiteLLMModel, Tool
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

chroma_client = chromadb.Client()
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')

collection = chroma_client.create_collection(
    name="party_planning",
    embedding_function=embedding_function
)

documents = [
    "A superhero-themed masquerade ball with luxury decor, including gold accents and velvet curtains.",
    "Hire a professional DJ who can play themed music for superheroes like Batman and Wonder Woman.",
    "For catering, serve dishes named after superheroes, like 'The Hulk's Green Smoothie' and 'Iron Man's Power Steak.'",
    "Decorate with iconic superhero logos and projections of Gotham and other superhero cities around the venue.",
    "Interactive experiences with VR where guests can engage in superhero simulations or compete in themed games."
]

collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)


class PartyPlanningRetrieverTool(Tool):
    name = "party_planning_retriever"
    description = "Uses semantic search to retrieve relevant party planning ideas for a superhero-themed party."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to search for, related to party planning or superhero themes."
        }
    }
    output_type = 'string'

    def forward(self, query:str) -> str:
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        retrieved_docs = results["documents"][0]
        return "\nRetrieved ideas:\n" + "\n".join(
            [f"\n===== Idea {i} =====\n{doc}" for i, doc in enumerate(retrieved_docs)]
        )

retriever_tool = PartyPlanningRetrieverTool()
model = LiteLLMModel(model_id="anthropic/claude-haiku-4-5-20251001")

agent = CodeAgent(tools=[retriever_tool], model=model)

# Step 5: Run it
response = agent.run(
    "Find ideas for entertainment and catering for a superhero-themed party."
)
print(response)