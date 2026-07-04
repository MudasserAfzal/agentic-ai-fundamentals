import json
import os
import time
import re
import uuid
from datetime import date
from pathlib import Path
from openai import AzureOpenAI
import chromadb

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

# --- Vector store setup ---
# ChromaDB persists to disk automatically
chroma = chromadb.PersistentClient(path="./memory")
collection = chroma.get_or_create_collection(
    name="research",
    metadata={"hnsw:space": "cosine"}
)


# --- Embedding helper ---
def embed(text: str) -> list[float]:
    """Turn text into a vector using OpenAI embeddings."""
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return resp.data[0].embedding


# --- Tool implementations ---
def search_web(query: str, max_results: int = 5) -> str:
    from ddgs import DDGS
    results = []
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if results:
                break
        except Exception:
            pass
        time.sleep(1)
    if not results:
        return "No results found. Try a different query."
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"{i}. {r['title']}\n   URL: {r['href']}\n   {r['body']}")
    return "\n\n".join(formatted)


def fetch_page(url: str) -> str:
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:3000]
    except Exception as e:
        return f"Could not fetch page: {e}"

def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)
        i += size - overlap
    return chunks



def remember(content: str, topic: str) -> str:
    chunks = chunk_text(content)
    for chunk in chunks:
        vector = embed(chunk)
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[{"topic": topic, "date": str(date.today())}]
        )
    return f"Saved {len(chunks)} chunks under topic: {topic}"


def recall(query: str, n_results: int = 3) -> str:
    """Search the vector store for relevant past research."""
    # TODO 3: embed the query
    vector = embed(query)

    # TODO 4: query ChromaDB for similar chunks
    results = collection.query(
        query_embeddings=[vector],
        n_results=n_results
    )

    if not results["documents"][0]:
        return "Nothing relevant found in memory."

    formatted = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        formatted.append(f"[{meta['topic']} — {meta['date']}]\n{doc}")
    return "\n\n".join(formatted)

def run_tool(tool_call) -> str:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    tools_map = {
        "search_web": search_web,
        "fetch_page": fetch_page,
        "remember": remember,
        "recall": recall,
    }
    if name not in tools_map:
        return f"Unknown tool: {name}"
    return tools_map[name](**args)


# --- Tool schemas ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information. Use this to discover new information. Do NOT use if you already have a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch full content of a URL. Use when you have a specific URL and need its full text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save important research findings to long-term memory. Use this after finding valuable information so it can be recalled in future sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The information to save"},
                    "topic": {"type": "string", "description": "Short label for this memory, e.g. 'RAG papers'"}
                },
                "required": ["content", "topic"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "recall",
        "description": "Search long-term memory for relevant information from past research sessions. ALWAYS call this first. If this returns content, use it to answer directly WITHOUT calling search_web.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "n_results": {"type": "integer", "default": 3}
            },
            "required": ["query"]
        }
    }
}
]

SYSTEM = f"""You are a research assistant with long-term memory. Today is {date.today()}.

For EVERY question follow this exact sequence — no exceptions:

1. Call recall() to check memory first
2. If recall returns relevant content → answer immediately, skip web search
3. If recall returns nothing relevant → call search_web() to find information
4. Call fetch_page() on the most promising URL for full content
5. Call remember() to save what you found — do this ALWAYS after any web search, automatically, without being asked
6. Answer the user with a clear cited response

You are building a personal knowledge base. Every piece of research must be saved.
Treat remember() as mandatory, not optional.
"""


# --- Agent loop ---
def run_agent(history: list, question: str) -> str:
    history.append({"role": "user", "content": question})

    print(f"\nThinking about: {question}\n")

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=2048,
            tools=TOOLS,
            messages=[{"role": "system", "content": SYSTEM}] + history
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            history.append(msg.model_dump())

            for tool_call in msg.tool_calls:
                print(f"  [{tool_call.function.name}] {tool_call.function.arguments}")
                result = run_tool(tool_call)
                print(f"  → {result[:100]}...")
                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            reply = msg.content
            history.append({"role": "assistant", "content": reply})
            return reply


def main():
    history = []
    print(f"Knowledge Assistant — session started {date.today()}")
    print("Your research is saved between sessions. Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break
        answer = run_agent(history, question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()