"""사진 분석을 위한 OpenAI Responses 클라이언트."""

import base64
import io
import json

import aiohttp
from PIL import Image

from .. import llm

OPENAI_URL = "https://api.openai.com/v1/responses"


def image_data_url(img: Image.Image, max_side=1536, short_side=768, quality=85):
    """LLM 에 보낼 그림: API(detail high)가 어차피 짧은 변 768 로 줄여 보므로 그 크기로 미리 줄여 보냄
    → 화질(토큰 수) 같고 전송량만 줄어듦. 작은 그림은 그대로."""
    im = img.convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    if min(im.size) > short_side:
        s = short_side / min(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
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


def motion_body(model, grid_img, photo=None, context="", note="", system=None, schema=None, name="motion_plan"):
    """격자를 그린 360 결과 (+원본 사진, 생성 프롬프트) → 움직일 영역 계획."""
    content = [{"type": "input_text", "text": "360 panorama with yaw/pitch grid:"},
               {"type": "input_image", "image_url": image_data_url(grid_img, 2048), "detail": "high"}]
    if photo is not None:
        content += [{"type": "input_text", "text": "Original source photo:"},
                    {"type": "input_image", "image_url": image_data_url(photo, 1024), "detail": "low"}]
    if context:
        content.append({"type": "input_text", "text": "Prompts and analysis used to generate this panorama:\n" + context})
    if note:
        content.append({"type": "input_text", "text": f"Operator note (may be Korean): {note}"})
    return {
        "model": model,
        "input": [{"role": "system", "content": system or llm.MOTION_SYSTEM}, {"role": "user", "content": content}],
        "text": {"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema or llm.MOTION_SCHEMA}},
    }


def review_body(model, views_img, photo=None, context=""):
    """2K 초안 4방향 화면(+원본 사진, 생성 프롬프트) → 검수 결과."""
    content = [{"type": "input_text", "text": "Draft panorama, four views (top-left FRONT, top-right RIGHT, "
                                              "bottom-left BACK, bottom-right LEFT):"},
               {"type": "input_image", "image_url": image_data_url(views_img, 2048), "detail": "high"}]
    if photo is not None:
        content += [{"type": "input_text", "text": "Original photo:"},
                    {"type": "input_image", "image_url": image_data_url(photo, 1024), "detail": "low"}]
    if context:
        content.append({"type": "input_text", "text": "Generation prompt used:\n" + context})
    return {
        "model": model,
        "input": [{"role": "system", "content": llm.REVIEW_SYSTEM}, {"role": "user", "content": content}],
        "text": {"format": {"type": "json_schema", "name": "pano_review", "strict": True, "schema": llm.REVIEW_SCHEMA}},
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

    async def _request(self, body: dict) -> tuple[dict, dict]:
        """Responses API 호출 → (구조화된 JSON 결과, usage)."""
        if not self.key:
            raise RuntimeError("OpenAI 키가 없어요. config.json 의 openai_key_file 을 확인해 주세요.")
        async with self.s.post(OPENAI_URL, json=body, headers={"Authorization": f"Bearer {self.key}"}) as r:
            data = await r.json(content_type=None)
            if r.status != 200:
                raise RuntimeError(f"OpenAI {r.status}: {json.dumps(data.get('error', data), ensure_ascii=False)}")
        return parse_response(data), data.get("usage")

    async def analyze(self, img: Image.Image, note: str = "", model: str | None = None) -> tuple[dict, dict]:
        """사진 → (LLM 재료 JSON, usage). model 을 주면 그 모델로, 아니면 기본 모델로."""
        return await self._request(request_body(model or self.model, img, note))

    async def review(self, views_img, photo=None, context="", model=None):
        """2K 초안 검수 → (JSON, usage)."""
        return await self._request(review_body(model or self.model, views_img, photo, context))

    async def plan_motion(self, grid_img, photo=None, context="", note="", model=None):
        """영상화: 움직일 곳(일부만) 계획 → (JSON, usage)."""
        return await self._request(motion_body(model or self.model, grid_img, photo, context, note))

    async def plan_event(self, grid_img, photo=None, context="", note="", model=None):
        """사건 영상: 한 번 일어나는 연출 계획 → (JSON, usage)."""
        return await self._request(motion_body(model or self.model, grid_img, photo, context, note,
                                               system=llm.EVENT_SYSTEM, schema=llm.EVENT_SCHEMA, name="event_plan"))
