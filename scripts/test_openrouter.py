#!/usr/bin/env python3
"""
Test script for OpenRouter connection.
Manually loads .env (if present) and attempts a simple completion.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path so we can import elenchus modules if needed,
# though using litellm directly is often simpler for a raw connectivity test.
# But testing via elenchus.llm ensures the project wrapper works too.
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

def load_env_manual():
    """Manually parse .env file since python-dotenv is not a dependency."""
    # User specified location
    env_path = Path("/home/leonb/maei/.env")
    if not env_path.exists():
        # Fallback to project root
        env_path = Path(__file__).parent.parent / ".env"
    
    if not env_path.exists():
        print(f"⚠️  No .env file found at {env_path}")
        return

    print(f"📂 Loading .env from {env_path}")
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                # clear quotes if present
                value = value.strip().strip("'").strip('"')
                if key not in os.environ:
                    os.environ[key] = value
                    # Mask key for printing
                    masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
                    print(f"   Set {key}={masked}")

async def main():
    load_env_manual()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not found in environment or .env")
        sys.exit(1)
    
    # Force the base URL if not set, though LiteLLM usually handles "openrouter/..."
    if "openrouter" not in (os.getenv("ELENCHUS_MODEL_FAST") or ""):
         print("ℹ️  Note: ELENCHUS_MODEL_FAST does not start with 'openrouter/'.")

    # Test DeepSeek R1 latency
    test_model = os.getenv("ELENCHUS_MODEL_CAPABLE", "openrouter/deepseek/deepseek-r1")
    
    print(f"\n🚀 Testing latency for {test_model}...")
    
    try:
        from elenchus.llm import complete
        import time
        t0 = time.time()
        response = await complete(
            model=test_model,
            messages=[{"role": "user", "content": "Calculate the square root of 54321 to 2 decimal places. Show your reasoning."}],
            max_tokens=1000
        )
        elapsed = time.time() - t0
        
        print("\n✅ Success!")
        print(f"Time: {elapsed:.2f}s")
        print(f"Model: {response.model}")
        print(f"Response: {response.text.strip()}")
        print(f"Cost: ${response.usage.cost_usd:.6f}")
        
    except Exception as e:
        print("\n❌ Connection Failed!")
        print(f"Error: {str(e)}")
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
