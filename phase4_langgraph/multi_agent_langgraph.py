import os
import json
from datetime import date
from pathlib import Path
from typing import TypedDict
from openai import AzureOpenAI
from langgraph.graph import StateGraph, END, START
import chromadb

# ── setup ────────────────────────────────────────────────────────
env_file = Path(__file__).resolve().parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

MODEL = os.environ.get("AZURE_DEPLOYMENT", "gpt-4o")
EMBEDDING_MODEL = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_key=os.environ["OPENAI_API_KEY"],
    api_version=os.environ["OPENAI_API_VERSION"],
)
chroma = chromadb.PersistentClient(path="./memory")
collection = chroma.get_or_create_collection(
    name="research", metadata={"hnsw:space": "cosine"}
)


# ════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ════════════════════════════════════════════════════════════════

def llm(messages: list) -> str:
    response = client.chat.completions.create(
        model=MODEL, max_tokens=2048, temperature=0, messages=messages
    )
    return response.choices[0].message.content


def embed(text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def search_web(query: str, max_results: int = 5) -> str:
    import time
    from ddgs import DDGS
    results = []
    for _ in range(3):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if results:
                break
        except Exception:
            pass
        time.sleep(1)
    if not results:
        return "No results found."
    return "\n\n".join(
        f"{i}. {r['title']}\n   URL: {r['href']}\n   {r['body']}"
        for i, r in enumerate(results, 1)
    )


def recall(query: str, n_results: int = 3) -> str:
    results = collection.query(
        query_embeddings=[embed(query)], n_results=n_results
    )
    if not results["documents"][0]:
        return "Nothing in memory."
    return "\n\n".join(
        f"[{m['topic']} — {m['date']}]\n{d}"
        for d, m in zip(results["documents"][0], results["metadatas"][0])
    )


def remember(content: str, topic: str) -> str:
    import uuid
    words = content.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+500]))
        i += 450
    for chunk in chunks:
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embed(chunk)],
            documents=[chunk],
            metadatas=[{"topic": topic, "date": str(date.today())}]
        )
    return f"Saved under: {topic}"


# ════════════════════════════════════════════════════════════════
# STATE SCHEMA
# ════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    question: str
    session_history: list[dict]
    decision: str
    research: str
    critique: str
    needs_more_research: bool
    final_answer: str
    research_iterations: int   # ← add this


# ════════════════════════════════════════════════════════════════
# NODE FUNCTIONS
# Each receives the full state, returns only what it changed
# ════════════════════════════════════════════════════════════════

ORCHESTRATOR_SYSTEM = f"""You are a senior research coordinator. Today is {date.today()}.

If conversation history contains enough context to answer, say ANSWER_DIRECTLY.
Otherwise say NEEDS_RESEARCH.

Reply with ONLY one of those two words."""

def node_decide(state: AgentState) -> dict:
    """Decides whether to answer from history or do fresh research."""
    print(f"\n[decide] Question: {state['question'][:60]}")

    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM},
    ] + state["session_history"] + [
        {"role": "user", "content": f"Question: {state['question']}\n\nANSWER_DIRECTLY or NEEDS_RESEARCH?"}
    ]

    decision = llm(messages).strip()
    print(f"[decide] → {decision}")

    # TODO 1: return a dict updating only the decision field
    return {"decision": decision}


RESEARCHER_SYSTEM = f"""You are a focused research agent. Today is {date.today()}.

STRICT sequence:
1. Call recall() first — check memory before web
2. If memory has the answer → return it
3. If not → search_web(), then remember() the findings
4. Return a detailed summary with sources"""

def node_researcher(state: AgentState) -> dict:
    print(f"\n[researcher] Researching: {state['question'][:60]}")
    is_second_pass = state["research_iterations"] > 0

    if not is_second_pass:
        # First pass: check memory first
        memory = recall(state["question"])
        print(f"  → recall: {memory[:80]}...")

        if "Nothing in memory" not in memory:
            print(f"  → Using memory, skipping web search")
            return {
                "research": memory,
                "research_iterations": state["research_iterations"] + 1
            }
    else:
        print(f"  → Second pass: critic flagged gaps, going to web regardless of memory")

    # Web search (always on second pass, or when memory is empty)
    web = search_web(state["question"])
    print(f"  → web search complete")

    prior = state.get("research", "")
    synthesis = llm([
        {"role": "system", "content": RESEARCHER_SYSTEM},
        {"role": "user", "content": f"""
Question: {state['question']}

Prior research (from memory or previous pass):
{prior if prior else "None"}

New web search results:
{web}

Synthesize a comprehensive research summary combining both sources."""}
    ])

    remember(synthesis, topic=state["question"][:50])
    print(f"  → saved to memory")

    return {
        "research": synthesis,
        "research_iterations": state["research_iterations"] + 1
    }




CRITIC_SYSTEM = """You are a critical reviewer evaluating research quality.

Output in this exact format:
GAPS: [list gaps]
CONFIDENCE: [High / Medium / Low]
SUGGESTIONS: [what would strengthen this]
VERDICT: [SUFFICIENT or NEEDS_MORE_RESEARCH]"""

def node_critic(state: AgentState) -> dict:
    """Reviews research quality. Sets needs_more_research based on verdict."""
    print(f"\n[critic] Reviewing research quality...")

    critique = llm([
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": f"""
Question: {state['question']}

Research:
{state['research']}

Review this research."""}
    ])

    needs_more = "NEEDS_MORE_RESEARCH" in critique
    print(f"[critic] → {'NEEDS MORE' if needs_more else 'SUFFICIENT'}")

    # TODO 3: return a dict updating critique and needs_more_research
    return {"critique": critique, "needs_more_research": needs_more}

SYNTHESIZER_SYSTEM = """You are a research synthesizer.
Write a comprehensive, well-structured final answer.
Use the research and critique provided.
Output ONLY the final answer."""

def node_synthesize(state: AgentState) -> dict:
    """Synthesizes research + critique into a final answer."""
    print(f"\n[synthesize] Writing final answer...")

    messages = [
        {"role": "system", "content": SYNTHESIZER_SYSTEM},
    ] + state["session_history"] + [
        {"role": "user", "content": f"""
Question: {state['question']}

Research findings:
{state['research']}

Critic review:
{state['critique']}

Write the final answer:"""}
    ]

    answer = llm(messages)

    # Update session history
    new_history = state["session_history"] + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": answer}
    ]

    # TODO 4: return a dict updating final_answer and session_history
    return {"final_answer": answer, "session_history": new_history}


def node_synthesize_direct(state: AgentState) -> dict:
    print(f"\n[synthesize_direct] Answering from context...")

    messages = [
        {"role": "system", "content": """You are a helpful assistant.
The conversation history contains detailed research and answers.
Use that history to answer the new question directly and specifically.
Do NOT say you lack information — the history has what you need."""},
    ] + state["session_history"] + [
        {"role": "user", "content": state["question"]}
    ]

    answer = llm(messages)


    new_history = state["session_history"] + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": answer}
    ]

    # TODO 5: return a dict updating final_answer and session_history
    return {"final_answer": answer, "session_history": new_history}


# ════════════════════════════════════════════════════════════════
# CONDITIONAL EDGE FUNCTIONS
# These are the routing decisions — return the name of next node
# ════════════════════════════════════════════════════════════════

def route_after_decide(state: AgentState) -> str:
    """After decide node: go to researcher or synthesize_direct?"""
    # TODO 6: if "ANSWER_DIRECTLY" in state["decision"] return "synthesize_direct"
    #         otherwise return "researcher"
    return "synthesize_direct" if "ANSWER_DIRECTLY" in state["decision"] else "researcher"


def route_after_critic(state: AgentState) -> str:
    """After critic node: do a second research pass or synthesize?"""
    # TODO 7: if state["needs_more_research"] return "researcher"
    #         otherwise return "synthesize"
    if state["needs_more_research"] and state["research_iterations"] < 2:
        return "researcher"
    return "synthesize"


# ════════════════════════════════════════════════════════════════
# BUILD THE GRAPH
# ════════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("decide",            node_decide)
    graph.add_node("researcher",        node_researcher)
    graph.add_node("critic",            node_critic)
    graph.add_node("synthesize",        node_synthesize)
    graph.add_node("synthesize_direct", node_synthesize_direct)

    # Entry point
    graph.set_entry_point("decide")

    # Conditional edge from decide
    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {
            "researcher":        "researcher",
            "synthesize_direct": "synthesize_direct"
        }
    )

    # Fixed edge: researcher always goes to critic
    graph.add_edge("researcher", "critic")

    # Conditional edge from critic
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "researcher": "researcher",
            "synthesize":  "synthesize"
        }
    )

    # Fixed edges to END
    graph.add_edge("synthesize",        END)
    graph.add_edge("synthesize_direct", END)

    return graph.compile()


agent_graph = build_graph()


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    session_history = []
    print(f"Multi-Agent Assistant (LangGraph) — {date.today()}")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break

        initial_state: AgentState = {
            "question": question,
            "session_history": session_history,
            "decision": "",
            "research": "",
            "critique": "",
            "needs_more_research": False,
            "final_answer": "",
            "research_iterations": 0
        }

        final_state = agent_graph.invoke(initial_state)

        # Persist session history for next question
        session_history = final_state["session_history"]

        print(f"\nAnswer:\n{final_state['final_answer']}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()