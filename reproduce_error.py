
import asyncio
import os
import sys
from elenchus.llm import complete
from elenchus.config import get_model_config

async def main():
    print("Testing LLM connection...")
    config = get_model_config()
    print(f"Model fast: {config.fast}")
    print(f"Model capable: {config.capable}")
    
    try:
        response = await complete(
            model=config.fast,
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(f"Success! Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
