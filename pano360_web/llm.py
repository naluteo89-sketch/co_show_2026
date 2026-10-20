"""
LLM 지시문 · 응답 스키마 · 프롬프트 조립 (HTTP 호출은 openai_client.py)
최종 프롬프트는 검증된 고정 문구(LoRA 트리거, 조명 일관성, 공통 네거티브)와 합쳐서 만든다.
"""

# LoRA 트리거 문구 — 첫 패스 프롬프트는 반드시 이 문장으로 시작해야 함 (test.json 노트)
TRIGGER = ("Fill the green spaces according to the image. Outpaint as a seamless 360 equirectangular "
           "panorama (2:1). Keep the horizon level. Match left and right edges.")
LIGHT_TMPL = ("Same {lighting} as the reference photo everywhere: identical sun direction, exposure, "
              "white balance and color grading across the full 360 degrees, {lighting_detail}.")
TAIL = "Photorealistic raw photo, natural contrast, true-to-life colors, fine natural detail."

NEG_COMMON = ("inconsistent lighting, different time of day, exposure mismatch, brightness gradient, color shift, "
              "color cast, green tint, desaturated area, visible seams, mismatched lighting, repeated textures, "
              "warped perspective, blurry edges, low quality, out of frame, floating objects, mismatched colors, "
              "hard edges, text, captions, watermark, worst quality, blurry, ugly, split screen, collage, frame, "
              "black borders, broken horizon, duplicated objects, duplicate people, HDR look, oversharpened, oversaturated, "
              "overexposed sky, blown-out highlights, halos around edges, painterly texture, smeared ground, swirl at the bottom")
SEAM_POS_TMPL = ("Fill the green spaces according to the image. Repair only the central vertical seam of this 360 equirectangular panorama. Seamlessly continue "
                 "the {seam_elements} across the band. Match the brightness, white balance, color and {lighting_short} "
                 "of both sides exactly; the band must not be darker or lighter than its neighbors. "
                 "Photorealistic, no visible join.")
SEAM_NEG_COMMON = ("inconsistent lighting, exposure mismatch, brightness gradient, color shift, color cast, green tint, "
                   "desaturated area, split screen, vertical dividing line, hard edge, mismatched horizon, text, watermark")

SYSTEM = """You write prompts for a FLUX.2 Klein + 360-ERP-outpaint LoRA pipeline.
The user's single photo is pasted onto a 2:1 equirectangular canvas; everything else (the other ~300 degrees,
sky and ground) is generated. Your job is to describe the WHOLE surrounding environment so the outpaint is
plausible, consistent and high quality.

Rules:
- English, concrete visual nouns and materials. No camera/lens jargon, no style words like "cinematic", "8k".
- scene: one or two sentences. Start with the place (name it only if clearly identifiable, e.g. a famous landmark),
  then the main subject, then what plausibly surrounds it in all directions including behind the camera,
  the ground directly beneath the camera (material and texture, since VR viewers look straight down) and the sky
  overhead (clouds/haze/clear, so it is not a flat white blank). 25-60 words.
- lighting: short phrase that matches the photo, e.g. "natural daytime lighting", "warm late-afternoon sunlight",
  "overcast diffuse daylight", "night street lighting".
- lighting_detail: 4-10 words, e.g. "soft daylight, clear consistent sky".
- lighting_short: 1-3 words used in the seam prompt, e.g. "daylight", "night lighting".
- lighting_negatives: lighting conditions that must NOT appear (the opposite times of day / weather).
  For a daytime photo: night, dusk, sunset, twilight, dark sky, stars, moon, artificial lights, day-night split.
- content_negatives: 6-12 failure modes specific to THIS scene (e.g. "distorted bridge geometry", "duplicated rocks",
  "invented architectural details", "unnatural water flow", "sky on the bottom").
- seam_elements: comma-separated list of 4-8 elements that must continue across the seam band.
- horizon_y: where the true horizon (eye level) sits in the photo, 0=top edge, 1=bottom edge; null if not visible.
- hfov_estimate: horizontal field of view in degrees of the photo (phone main camera ~ 65-75, ultra-wide ~ 100-120).
- ground_below: 8-20 words, what is directly beneath the camera seen from straight above (e.g. "weathered flat granite
  paving stones with thin grass in the joints and a few fallen yellow leaves"). No camera, tripod, feet or shadows.
- sky_above: 6-15 words, what is directly overhead looking straight up (e.g. "soft grey overcast sky with light mist").
- summary_ko / tips_ko: Korean, for the operator. tips_ko: risks for this photo (e.g. people, text, tilted horizon,
  very narrow FOV) and what to adjust.
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "scene": {"type": "string"},
        "lighting": {"type": "string"},
        "lighting_detail": {"type": "string"},
        "lighting_short": {"type": "string"},
        "lighting_negatives": {"type": "array", "items": {"type": "string"}},
        "content_negatives": {"type": "array", "items": {"type": "string"}},
        "seam_elements": {"type": "string"},
        "ground_below": {"type": "string"},
        "sky_above": {"type": "string"},
        "horizon_y": {"type": ["number", "null"]},
        "hfov_estimate": {"type": "number"},
        "summary_ko": {"type": "string"},
        "tips_ko": {"type": "string"},
    },
    "required": ["scene", "lighting", "lighting_detail", "lighting_short", "lighting_negatives",
                 "content_negatives", "seam_elements", "ground_below", "sky_above", "horizon_y", "hfov_estimate", "summary_ko", "tips_ko"],
}


def _join(items):
    seen, out = set(), []
    for s in items:
        for p in s.split(","):
            p = p.strip()
            if p and p.lower() not in seen:
                seen.add(p.lower())
                out.append(p)
    return ", ".join(out)


def assemble(a: dict) -> dict:
    """LLM 재료 → 최종 4개 프롬프트."""
    positive = " ".join([TRIGGER, a["scene"].strip(),
                         LIGHT_TMPL.format(lighting=a["lighting"].strip(), lighting_detail=a["lighting_detail"].strip()),
                         TAIL])
    negative = _join([", ".join(a["lighting_negatives"]), NEG_COMMON, ", ".join(a["content_negatives"])])
    seam_positive = SEAM_POS_TMPL.format(seam_elements=a["seam_elements"].strip(),
                                         lighting_short=a["lighting_short"].strip())
    seam_negative = _join([", ".join(a["lighting_negatives"]), SEAM_NEG_COMMON])
    return {"positive": positive, "negative": negative,
            "seam_positive": seam_positive, "seam_negative": seam_negative}


POLE_POS_TMPL = ("Repair only the two round patches of this rotated 360 panorama: on the left the ground directly beneath "
                 "the camera seen from straight above ({ground}); on the right the sky directly overhead ({sky}). "
                 "Continue the surrounding textures and colors seamlessly with even, natural detail and the same "
                 "{lighting_short}. No swirl, no stretching, no radial streaks. Photorealistic.")
POLE_NEG = ("swirl, vortex, radial blur, stretched texture, smeared ground, pinched center, tripod, camera, photographer, "
            "feet, shoes, shadow of a person, hole, black spot, logo, text, watermark, visible circle edge, seam, "
            "duplicated objects, buildings, trees growing sideways, oversharpened, HDR look")


def pole_prompts(a: dict) -> dict:
    ground = (a.get("ground_below") or "natural ground matching the surrounding area").strip()
    sky = (a.get("sky_above") or "sky matching the surrounding sky").strip()
    pos = POLE_POS_TMPL.format(ground=ground, sky=sky, lighting_short=(a.get("lighting_short") or "daylight").strip())
    neg = _join([", ".join(a.get("lighting_negatives") or []), POLE_NEG])
    return {"positive": pos, "negative": neg}


REFINE_TMPL = ("A high-resolution photograph. {scene} Natural fine detail, realistic textures of stone, wood, "
               "foliage and water, crisp but not oversharpened, true-to-life colors, natural contrast.")
REFINE_NEG = ("painting, illustration, cartoon, anime, digital art, CGI, 3d render, watercolor, flat colors, "
              "posterized, smooth plastic texture, oversharpened, halos, HDR look, oversaturated, blurry, "
              "text, watermark")


def refine_prompts(a: dict, fallback_positive: str = "") -> dict:
    scene = (a.get("scene") or "").strip()
    if not scene and fallback_positive:
        scene = fallback_positive.replace(TRIGGER, "").strip()
    return {"positive": REFINE_TMPL.format(scene=scene), "negative": REFINE_NEG}
