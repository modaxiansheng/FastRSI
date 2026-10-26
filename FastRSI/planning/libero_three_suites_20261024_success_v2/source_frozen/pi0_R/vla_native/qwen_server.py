"""Minimal OpenAI-compatible server for the local Qwen2.5-VL verifier.

Avoids vLLM because the image's transformers 4.53.2 dropped deepseek_v2,
which vLLM 0.11 imports at startup.
"""

from __future__ import annotations

import base64
import io
import time

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_PATH = "/model"
SERVED_NAME = "qwen25vl7b"

app = FastAPI()

print("[boot] loading processor ...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("[boot] loading model ...", flush=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",
    trust_remote_code=True,
).eval()
print("[boot] READY", flush=True)


def _decode_image(url: str) -> Image.Image:
    if not url.startswith("data:"):
        raise ValueError("only data URLs are supported")
    raw = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": SERVED_NAME, "object": "model", "created": int(time.time())}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages") or []
    max_tokens = int(body.get("max_tokens") or 1024)

    system_parts: list[str] = []
    qwen_content: list[dict] = []
    images: list[Image.Image] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            system_parts.append(content if isinstance(content, str) else str(content))
            continue
        if role != "user":
            continue
        if isinstance(content, str):
            qwen_content.append({"type": "text", "text": content})
            continue
        for part in content:
            ptype = part.get("type")
            if ptype == "text":
                qwen_content.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image_url":
                qwen_content.append({"type": "image"})
                images.append(_decode_image(part["image_url"]["url"]))

    convo: list[dict] = []
    if system_parts:
        convo.append({"role": "system", "content": [{"type": "text", "text": "\n".join(system_parts)}]})
    convo.append({"role": "user", "content": qwen_content})

    text = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images or None, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    completion = processor.batch_decode(
        generated[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return JSONResponse(
        {
            "id": "chatcmpl-qwen25vl7b",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": SERVED_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(prompt_len),
                "completion_tokens": int(generated.shape[1] - prompt_len),
                "total_tokens": int(generated.shape[1]),
            },
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=38710, log_level="info")
