import argparse
import os
from pathlib import Path
from openai import AzureOpenAI

SYSTEM = "You are a concise, direct assistant. Answer in 2-3 sentences max."
MAX_HISTORY_TURNS = 5

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


def trim(history):
    while len(history) > MAX_HISTORY_TURNS * 2:
        history.pop(0)
        history.pop(0)


def chat(history, text, system, stream, totals):
    history.append({"role": "user", "content": text})
    trim(history)
    messages = [{"role": "system", "content": system}] + history
    opts = dict(model=MODEL, max_tokens=1024, messages=messages)

    if stream:
        stream_resp = client.chat.completions.create(
            **opts, stream=True, stream_options={"include_usage": True}
        )
        print("Bot: ", end="", flush=True)
        parts, usage = [], None
        for chunk in stream_resp:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                print(token, end="", flush=True)
                parts.append(token)
            if chunk.usage:
                usage = chunk.usage
        print()
        reply = "".join(parts)
    else:
        resp = client.chat.completions.create(**opts)
        reply, usage = resp.choices[0].message.content, resp.usage
        print(f"Bot: {reply}")

    history.append({"role": "assistant", "content": reply})
    totals["in"] += usage.prompt_tokens
    totals["out"] += usage.completion_tokens
    totals["total"] += usage.total_tokens
    print(
        f"     {usage.prompt_tokens} in / {usage.completion_tokens} out"
        f" | session: {totals['in']} in / {totals['out']} out\n"
    )
    return reply

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", default=SYSTEM)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    history, totals = [], {"in": 0, "out": 0, "total": 0}
    print("Type 'quit' to exit, '/tokens' for stats.\n")

    while True:
        text = input("You: ").strip()
        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            break
        if text == "/tokens":
            print(f"Session: {totals['in']} in / {totals['out']} out | {len(history) // 2} pairs kept\n")
            continue
        chat(history, text, args.persona, args.stream, totals)


if __name__ == "__main__":
    main()
