
import asyncio
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Load .env
from pathlib import Path
env_path = Path("/home/leonb/maei/.env")
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip("'").strip('"')
            if k not in os.environ:
                os.environ[k] = v

from elenchus.llm import complete

async def main():
    model = "openrouter/google/gemini-2.5-pro"
    print(f"Testing {model}...")
    try:
        response = await complete(
            model=model,
            messages=[{"role": "user", "content": 'Solve for x: 2x + 3 = 17. Return ONLY valid JSON with key "answer". No markdown fences.'}],
            system="You are a precise mathematical calculator. Return only valid JSON.",
            max_tokens=256,
        )
        print(f"Raw response: {repr(response.text)}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
