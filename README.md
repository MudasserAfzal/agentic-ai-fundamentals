# LLM Learning Project

Progressive build-up from a simple chatbot to production multi-agent systems.

## Learning path

| Phase | File | Docs |
|-------|------|------|
| 1 — Chatbot | [`phase1_chatbot.py`](phase1_chatbot.py) | [docs/phase1-chatbot.md](docs/phase1-chatbot.md) |
| 2 — Research Agent | [`phase2_research_agent.py`](phase2_research_agent.py) | [docs/phase2-research-agent.md](docs/phase2-research-agent.md) |
| 3 — Memory Agent | [`phase3_memory_agent.py`](phase3_memory_agent.py) | [docs/phase3-memory-agent.md](docs/phase3-memory-agent.md) |
| 4 — Multi-Agent | [`phase4_multi_agent.py`](phase4_multi_agent.py) | [docs/phase4-multi-agent.md](docs/phase4-multi-agent.md) |
| 5 — Production | [`phase5_production_agent.py`](phase5_production_agent.py) | [docs/phase5-production-agent.md](docs/phase5-production-agent.md) |
| 4 (LangGraph) | [`langchain/multi_agent_langgraph.py`](langchain/multi_agent_langgraph.py) | [langchain/README.md](langchain/README.md) |

## Setup

```bash
python -m venv llm_practice
source llm_practice/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```env
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
OPENAI_API_KEY=your-key
OPENAI_API_VERSION=2025-04-01-preview
AZURE_DEPLOYMENT=gpt-4o
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

## Progression

```
Phase 1  Chat API + history
   ↓
Phase 2  + tool calling (search, fetch)
   ↓
Phase 3  + ChromaDB memory (remember, recall)
   ↓
Phase 4  + multi-agent (researcher, critic, orchestrator)
   ↓
Phase 5  + FastAPI, tracing, cost guard, evals
```

LangGraph variant of Phase 4: [`langchain/multi_agent_langgraph.py`](langchain/multi_agent_langgraph.py)

## Other projects

- [**VerifAI**](verifAI/README.md) — AI hallucination detector (LangGraph + FastAPI + web UI)
- `langchain/week1.py` → `week3_reducer.py` — verification pipeline leading to VerifAI
