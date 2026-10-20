"""사진 분석을 위한 OpenAI Responses 클라이언트."""

import base64
import io
import json

import aiohttp
from PIL import Image

import llm

OPENAI_URL = "https://api.openai.com/v1/responses"


def image_data_url(img: Image.Image, max_side=1536):
    im = img.convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def request_body(model, img, note=""):
    user_text = "Analyze this photo for 360 outpainting."
    if note:
        user_text += f"\nOperator note (may be Korean): {note}"
    return {
        "model": model,
        "input": [
            {"role": "system", "content": llm.SYSTEM},
            {"role": "user", "content": [
                {"type": "input_text", "text": user_text},
                {"type": "input_image", "image_url": image_data_url(img), "detail": "high"},
            ]},
        ],
        "text": {"format": {"type": "json_schema", "name": "pano_prompt", "strict": True, "schema": llm.SCHEMA}},
    }


def parse_response(data: dict) -> dict:
    if data.get("error"):
        raise RuntimeError(f"OpenAI 오류: {data['error']}")
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                return json.loads(c["text"])
            if c.get("type") == "refusal":
                raise RuntimeError(f"모델이 분석을 거절했어요: {c.get('refusal')}")
    raise RuntimeError("OpenAI 응답에서 결과 텍스트를 찾지 못했어요.")


class OpenAIClient:
    def __init__(self, session: aiohttp.ClientSession, api_key: str, model: str):
        self.s, self.key, self.model = session, api_key, model

    @property
    def ready(self):
        return bool(self.key)

    async def analyze(self, img: Image.Image, note: str = "", model: str | None = None) -> tuple[dict, dict]:
        """사진 → (LLM 재료 JSON, usage). model 을 주면 그 모델로, 아니면 기본 모델로."""
        if not self.key:
            raise RuntimeError("OpenAI 키가 없어요. config.json 의 openai_key_file 을 확인해 주세요.")
        body = request_body(model or self.model, img, note)
        async with self.s.post(OPENAI_URL, json=body, headers={"Authorization": f"Bearer {self.key}"}) as r:
            data = await r.json(content_type=None)
            if r.status != 200:
                raise RuntimeError(f"OpenAI {r.status}: {json.dumps(data.get('error', data), ensure_ascii=False)}")
        return parse_response(data), data.get("usage")
