import json
import argparse
from openai import AzureOpenAI
from ddgs import DDGS
from pathlib import Path
from datetime import date

import os

env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

MODEL = os.environ.get("AZURE_DEPLOYMENT", "gpt-4o")
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_key=os.environ["OPENAI_API_KEY"],
    api_version=os.environ["OPENAI_API_VERSION"],
)

# --- Tool definitions (what the model can call) ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for URLs and snippets. Use this FIRST to discover relevant pages. Do NOT use this if you already have a URL — use fetch_page instead.",
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
            "description": "Fetch the full content of a URL. Use this when you already have a specific URL and need its full content — especially when the user gives you a URL directly or when search snippets are not detailed enough.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full URL to fetch"}
                },
                "required": ["url"]
            }
        }
    }
]

SYSTEM = f"""You are a research assistant. Today's date is {date.today()}.

When asked a question:
1. Use search_web to find relevant pages
2. Use fetch_page to read the full content of promising URLs
3. If the user gives you a URL directly, call fetch_page on it immediately — do not search first
4. Search multiple times with varied queries if needed
5. Synthesize findings into a cited answer with URLs
"""

# --- Tool executor (your code runs this) ---

import time

import urllib.request

def fetch_page(url: str) -> str:
    import re
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 100:
            return f"Page fetched but little text extracted. Try search_web instead. URL: {url}"
        return text[:3000]
    except Exception as e:
        return f"Could not fetch page: {e}. Try search_web with the site name instead."

def search_web(query: str, max_results: int = 5) -> str:
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if results:
                break
        except Exception as e:
            print(f"  Search error: {e}")
        time.sleep(1)
    
    if not results:
        return "No results found. Try a different query."
    
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"{i}. {r['title']}\n   URL: {r['href']}\n   {r['body']}")
    return "\n\n".join(formatted)


def run_tool(tool_call) -> str:
    """Dispatch a tool_call object to the right function."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)  # string → dict
    
    if name == "search_web":
        return search_web(**args)
    if name == "fetch_page":
        return fetch_page(**args)
    return f"Unknown tool: {name}"


# --- The agent loop ---

def run_agent(question: str) -> str:
    """
    Core agentic loop:
    1. Send question to model
    2. If model returns tool_calls → run them, feed results back, repeat
    3. If model returns text → done, return it
    """
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": question}
    ]
    
    print(f"\nResearching: {question}\n")
    
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=2048,
            tools=TOOLS,
            messages=messages
        )
        
        msg = response.choices[0].message
        
        # TODO 1: Check if the model wants to call tools
        # Hint: check if msg.tool_calls is not None and not empty
        if msg and msg.tool_calls is not None and len(msg.tool_calls) > 0:
            
            # TODO 2: Append the assistant's tool-call message to history
            # Hint: msg has .content, .tool_calls — convert to dict
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})
            
            # TODO 3: Run each tool call and append results
            for tool_call in msg.tool_calls:
                print(f"  Calling: {tool_call.function.name}({tool_call.function.arguments})")
                result = run_tool(tool_call)
                print(f"  Result preview: {result[:200]}\n")

                
                # TODO 4: Append tool result to messages
                # Hint: role="tool", need tool_call_id
                messages.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
        
        else:
            # Model returned text — we're done
            return msg.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=None)
    args = parser.parse_args()

    if args.question:
        answer = run_agent(args.question)
        print(f"\nAnswer:\n{answer}")
    else:
        print("Research Assistant. Type 'quit' to exit.\n")
        while True:
            question = input("Question: ").strip()
            if not question:
                continue
            if question.lower() == "quit":
                break
            answer = run_agent(question)
            print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()