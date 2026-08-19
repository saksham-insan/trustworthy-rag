"""
Diagnostic: reveals the FULL underlying error from Groq's SDK, since
"Connection error" alone hides the real cause (often an SSL/certificate
issue, especially common with antivirus software that inspects HTTPS).

Usage:
    python src/debug_groq.py
"""

import os
import traceback
from dotenv import load_dotenv

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

print("=== Testing Groq with full traceback ===")
try:
    from groq import Groq
    client = Groq(api_key=groq_key)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Reply with exactly one word: hello"}],
    )
    print(f"[OK] Groq responded: {completion.choices[0].message.content.strip()}")
except Exception:
    print("[FAIL] Full error below:\n")
    traceback.print_exc()