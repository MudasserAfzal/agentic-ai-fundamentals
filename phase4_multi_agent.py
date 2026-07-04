import json
import os
import time
import re
import uuid
from datetime import date
from pathlib import Path
from openai import AzureOpenAI
import chromadb

# --- Setup (same as Phase 3) ---
env_file = Path(__file__).parent / ".env"
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
    name="research",
    metadata={"hnsw:space": "cosine"}
)


# ============================================================
# SHARED UTILITIES (same as Phase 3)
# ============================================================

def embed(text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

def llm(messages: list, tools: list = None, model: str = MODEL) -> any:
    """Single reusable LLM call. Returns the message object."""
    kwargs = dict(model=model, max_tokens=2048, messages=messages)
    if tools:
        kwargs["tools"] = tools
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message


# ============================================================
# TOOLS (same as Phase 3 — researcher uses these)
# ============================================================

def search_web(query: str, max_results: int = 5) -> str:
    results = []
    from ddgs import DDGS
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

def fetch_page(url: str) -> str:
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()[:3000]
    except Exception as e:
        return f"Could not fetch: {e}"

def remember(content: str, topic: str = "general") -> str:
    for chunk in chunk_text(content):
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embed(chunk)],
            documents=[chunk],
            metadatas=[{"topic": topic, "date": str(date.today())}]
        )
    return f"Saved under: {topic}"

def recall(query: str, n_results: int = 3) -> str:
    results = collection.query(
        query_embeddings=[embed(query)],
        n_results=n_results
    )
    if not results["documents"][0]:
        return "Nothing in memory."
    return "\n\n".join(
        f"[{m['topic']} — {m['date']}]\n{d}"
        for d, m in zip(results["documents"][0], results["metadatas"][0])
    )

def run_tool(tool_call) -> str:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    if name == "remember":
        args.setdefault("topic", "general")
    tools_map = {
        "search_web": search_web,
        "fetch_page": fetch_page,
        "remember": remember,
        "recall": recall,
    }
    if name not in tools_map:
        return f"Unknown tool: {name}"
    return tools_map[name](**args)

RESEARCHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search long-term memory first. If relevant content found, use it and skip web search.",
            "parameters": {"type": "object",
                "properties": {"query": {"type": "string"}, "n_results": {"type": "integer", "default": 3}},
                "required": ["query"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for new information. Only use if recall found nothing relevant.",
            "parameters": {"type": "object",
                "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "default": 5}},
                "required": ["query"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch full content of a URL. Use when you have a specific URL.",
            "parameters": {"type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save findings to long-term memory. Always call after web search.",
            "parameters": {"type": "object",
                "properties": {"content": {"type": "string"}, "topic": {"type": "string"}},
                "required": ["content", "topic"]}
        }
    },
]


# ============================================================
# AGENT 1 — RESEARCHER
# ============================================================

RESEARCHER_SYSTEM = f"""You are a focused research agent. Today is {date.today()}.

Your only job is to find accurate, comprehensive information on a given topic.


STRICT sequence — no exceptions:
1. Call recall() first
2. If recall returns relevant content → use it and return findings
3. If recall returns nothing relevant → you MUST call search_web(), then fetch_page() on the best URL, then remember() the findings
4. Never stop after just recall — if memory is empty, always search the web
6. Fetch at most 3 pages per research task — choose the most promising URLs only.
5. Return a detailed research summary with sources
"""

def researcher_agent(task: str) -> str:
    print(f"\n  [Researcher] Starting: {task[:80]}")
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM},
        {"role": "user", "content": task}
    ]

    # Force recall on the very first call
    first_call = True

    while True:
        kwargs = dict(
            model=MODEL,
            max_tokens=2048,
            tools=RESEARCHER_TOOLS,
            messages=messages
        )
        
        # First turn: force recall. After that: let model decide freely
        if first_call:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": "recall"}}
            first_call = False

        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg.model_dump())
            for tc in msg.tool_calls:
                print(f"    → {tc.function.name}({tc.function.arguments[:60]})")
                result = run_tool(tc)
                
                # If recall returned nothing, inject a push to search
                if tc.function.name == "recall" and "Nothing in memory" in result:
                    result += "\n\nMemory is empty. You MUST now call search_web() to find this information."
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })
        else:
            print(f"  [Researcher] Done.")
            return msg.content


# ============================================================
# AGENT 2 — CRITIC
# ============================================================

CRITIC_SYSTEM = """You are a critical reviewer. Your job is to evaluate research quality.

Given a research summary and the original question, you must:
1. Check if the question is fully answered
2. Identify any gaps, missing angles, or weak points
3. Assess confidence level: High / Medium / Low
4. Suggest what additional research would strengthen the answer

Be specific and constructive. Output in this format:

GAPS: [list any gaps or unanswered parts]
CONFIDENCE: [High / Medium / Low]
SUGGESTIONS: [what would make this stronger]
VERDICT: [SUFFICIENT or NEEDS_MORE_RESEARCH]
"""

def critic_agent(research: str, original_question: str) -> str:
    """Reviews research quality. Returns structured critique."""
    # TODO 1: call llm() with CRITIC_SYSTEM, the research, and original question
    # No tools needed — critic just reads and thinks
    # Return the critic's response text
    print(f"\n  [Critic] Reviewing research quality...")
    

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": f"""
        Original question: {original_question}

        Research findings:
        {research}

        Please review this research.
        """}
    ]
    
    msg = llm(messages)  # no tools
    print(f"  [Critic] Done.")
    return msg.content


# ============================================================
# AGENT 3 — ORCHESTRATOR
# ============================================================

ORCHESTRATOR_SYSTEM = """You are a senior research coordinator managing a team.

You have access to the conversation history. Before delegating to the researcher, ask yourself:
- Does the conversation history already contain enough information to answer this question?
- Is this a followup to a previous question that was already researched?

If YES → answer directly from history. Do NOT call the researcher.
If NO → this is a genuinely new topic that needs research.

When synthesizing, always consider the full conversation context.
"""

def orchestrator(question: str, session_history: list) -> str:
    print(f"\n[Orchestrator] Received: {question}")

    # Combine question + decision request into one user message
    decision_prompt = f"""User question: {question}

Based on the conversation history above, do you have enough information to answer this question directly, or does it require new research?

Reply with ONLY one of these two words:
ANSWER_DIRECTLY
NEEDS_RESEARCH"""

    decision_messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM},
    ] + session_history + [
        {"role": "user", "content": decision_prompt}
    ]

    decision = llm(decision_messages).content.strip()
    print(f"[Orchestrator] Decision: {decision}")

    if "ANSWER_DIRECTLY" in decision:
        print(f"[Orchestrator] Answering from context...")
        synthesis_messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM},
        ] + session_history + [
            {"role": "user", "content": question}
        ]
        msg = llm(synthesis_messages)
        final_answer = msg.content

    else:
        research = researcher_agent(question)
        critique = critic_agent(research, question)

        if "NEEDS_MORE_RESEARCH" in critique:
            print(f"\n[Orchestrator] Critic flagged gaps — running second research pass...")
            additional = researcher_agent(
                f"Additional research needed. Original question: {question}\nGaps: {critique}"
            )
            research = research + "\n\nAdditional findings:\n" + additional

        print(f"\n[Orchestrator] Synthesizing final answer...")
        synthesis_messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM},
        ] + session_history + [
            {"role": "user", "content": f"""Question: {question}

Research findings:
{research}

Critic review:
{critique}

Please provide the final answer."""}
        ]
        msg = llm(synthesis_messages)
        final_answer = msg.content

    session_history.append({"role": "user", "content": question})
    session_history.append({"role": "assistant", "content": final_answer})

    return final_answer

# ============================================================
# MAIN
# ============================================================

def main():
    session_history = []
    print(f"Multi-Agent Research Assistant — {date.today()}")
    print("Powered by: Orchestrator → Researcher + Critic\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break

        answer = orchestrator(question, session_history)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()