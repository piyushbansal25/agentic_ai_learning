# Agentic AI Learning Journey

This repo documents my hands-on journey learning to build AI agents — from scratch, frameworks, to RAG systems — as part of my transition into Agentic AI / AI Data Engineering roles.

## What's Inside

| File | Description |
|---|---|
| `first_call.py` | First API call to Claude using the Anthropic SDK |
| `agent_from_scratch.py` | A manually built agent loop (Thought → Action → Observation) without any framework, demonstrating how agents work under the hood |
| `smolagents_claude.py` | First agent built using the `smolagents` framework (`ToolCallingAgent`) with Claude as the model |
| `code_agent.py` | Agent using `CodeAgent` — writes and executes Python code as actions instead of JSON |
| `alfred_agent.py` | Multi-tool agent combining web search and 3 custom tools to plan a themed party end-to-end |
| `retrieval_agent.py` | A RAG (Retrieval-Augmented Generation) agent using ChromaDB for semantic search over a custom knowledge base |

## Tech Stack

- **Model:** Claude (Anthropic API) via `LiteLLMModel`
- **Framework:** smolagents
- **Vector DB:** ChromaDB
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)

## Learning Path

Following a structured 6-month roadmap:
1. Foundations & Architecture
2. Agent Frameworks & Memory
3. Tools, APIs & Multi-Agent Systems
4. Evaluation, Safety & Deployment
5. Specialization & Capstone Project

## About

Data Engineer transitioning into Agentic AI / AI Data Engineering roles. Background in Azure Databricks, PySpark, Delta Lake, and now building hands-on expertise in LLM agents, RAG, and multi-agent systems.