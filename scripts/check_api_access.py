"""
scripts/check_api_access.py
Diagnostic script — run this to see exactly which models each Gemini key can access.
Usage: python scripts/check_api_access.py
"""
import os
from dotenv import load_dotenv
from pathlib import Path
from google import genai

load_dotenv(Path(__file__).parent.parent / ".env")

KEYS = {
    "KEY_A": os.environ.get("GEMINI_API_KEY_A"),
    "KEY_B": os.environ.get("GEMINI_API_KEY_B"),
    "KEY_C": os.environ.get("GEMINI_API_KEY_C"),
}

PROBE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]

print("\n" + "="*60)
print("Meridian — API Key Diagnostic")
print("="*60)

for key_name, api_key in KEYS.items():
    print(f"\n--- {key_name} ---")
    if not api_key:
        print("  NOT SET in .env")
        continue

    client = genai.Client(api_key=api_key)

    # 1. List all available models
    try:
        models = list(client.models.list())
        generative = [m.name for m in models if "generateContent" in (m.supported_actions or [])]
        print(f"  Available generative models ({len(generative)}):")
        for m in sorted(generative):
            print(f"    {m}")
    except Exception as e:
        print(f"  ERROR listing models: {e}")
        generative = []

    # 2. Probe which specific models respond
    print(f"  Probe results:")
    for model_id in PROBE_MODELS:
        try:
            r = client.models.generate_content(
                model=model_id,
                contents="Reply with the single word: OK",
            )
            print(f"    ✅ {model_id} — WORKS")
        except Exception as e:
            err = str(e)
            if "404" in err:
                print(f"    ❌ {model_id} — NOT AVAILABLE (404)")
            elif "429" in err:
                print(f"    ⚠️  {model_id} — QUOTA/RATE LIMIT (429)")
            else:
                print(f"    ❌ {model_id} — ERROR: {err[:80]}")

print("\n" + "="*60)
print("Done. Share this output to decide which model to use for all nodes.")
print("="*60 + "\n")
