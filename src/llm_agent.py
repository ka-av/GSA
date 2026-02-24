from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Tuple
from io import BytesIO

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

DEFAULT_MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTIONS = """You are a visual assistant doing grounded QA on a synthetic image.
RULES:
- Use ONLY what you can see in the image.
- If the image is blurry/occluded and you truly cannot determine, abstain for that subquestion.
- Never invent objects.
- Output MUST be valid JSON ONLY (no markdown).
- Include a short 'reason' string summarizing why (keep it brief).
"""

def _img_to_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def call_gemini(
    image: Image.Image,
    questions: List[str],
    api_key: str | None = None,
    model: str = DEFAULT_MODEL
) -> Tuple[str, Dict[str, Any]]:
    load_dotenv()

    if api_key is None:
        api_key = (os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("API_KEY", "").strip())

    if not api_key or "PASTE_YOUR_KEY_HERE" in api_key:
        raise RuntimeError("Missing GEMINI_API_KEY (or API_KEY). Put it in .env or set env var.")

    client = genai.Client(api_key=api_key)

    q_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

    user_prompt = f"""Answer these 4 questions about the image:

{q_text}

Return JSON with keys:
exist {{answer, bbox, conf}}
attribute {{answer, conf}}
spatial {{answer, conf}}
geometry {{answer, conf}}
abstain {{exist, attribute, spatial, geometry}}
reason (short)

Constraints:
- exist.answer and spatial.answer must be "yes" or "no" unless abstaining.
- bbox must be [x1,y1,x2,y2] integers if exist.answer="yes", else null.
- conf values must be in [0,1].
- If abstaining a subquestion, set abstain.<field>=true and set answer to "" (or null), conf to 0.
Output JSON ONLY.
"""

    img_bytes = _img_to_png_bytes(image)

    # KEY POINT: keyword-only text= fixes your earlier TypeError
    parts = [
        types.Part.from_text(text=SYSTEM_INSTRUCTIONS),
        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        types.Part.from_text(text=user_prompt),
    ]

    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=700,
        ),
    )

    raw_text = (resp.text or "").strip()

    try:
        parsed = json.loads(raw_text)
        return raw_text, parsed
    except Exception:
        return raw_text, {"_parse_error": True, "_raw": raw_text}