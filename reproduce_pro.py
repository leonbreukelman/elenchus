
import asyncio
import os
import sys
from elenchus.llm import complete
from elenchus.config import get_model_config

async def main():
    print("Testing LLM connection for CAPABLE model...")
    config = get_model_config()
    print(f"Model capable: {config.capable}")
    
    try:
        response = await complete(
            model=config.capable,
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
        print(f"Success! Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
