"""
Quick sanity check: confirms your .env file is set up correctly and both
free-tier API keys (Gemini, Groq) actually work, before we build anything
that depends on them.

NOTE: uses `truststore` to make Python trust Windows' own certificate
store. This fixes SSL errors common on machines where antivirus software
or a network does "SSL inspection" (rewrites HTTPS certificates to scan
traffic) — Python's default certificate bundle doesn't know about those
inspection certificates, but Windows itself does.

Usage:
    python src/test_keys.py
"""

import truststore
truststore.inject_into_ssl()  # must run before any network libraries connect

import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file in the project root

gemini_key = os.getenv("GEMINI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

print("=== Checking .env file ===")
if not gemini_key or gemini_key == "your_gemini_key_here":
    print("[FAIL] GEMINI_API_KEY missing or still the placeholder value.")
else:
    print(f"[OK] GEMINI_API_KEY found (starts with: {gemini_key[:6]}...)")

if not groq_key or groq_key == "your_groq_key_here":
    print("[FAIL] GROQ_API_KEY missing or still the placeholder value.")
else:
    print(f"[OK] GROQ_API_KEY found (starts with: {groq_key[:6]}...)")

print("\n=== Testing Gemini (google-genai SDK) ===")
try:
    from google import genai
    client = genai.Client(api_key=gemini_key)
    response = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents="Reply with exactly one word: hello",
    )
    print(f"[OK] Gemini responded: {response.text.strip()}")
except Exception as e:
    print(f"[FAIL] Gemini error: {e}")

print("\n=== Testing Groq ===")
try:
    from groq import Groq
    client = Groq(api_key=groq_key)
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": "Reply with exactly one word: hello"}],
    )
    print(f"[OK] Groq responded: {completion.choices[0].message.content.strip()}")
except Exception as e:
    print(f"[FAIL] Groq error: {e}")

print("\nDone.")