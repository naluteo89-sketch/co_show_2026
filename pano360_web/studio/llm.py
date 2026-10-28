"""
LLM 지시문 · 응답 스키마 · 프롬프트 조립 (HTTP 호출은 openai_client.py)
최종 프롬프트는 검증된 고정 문구(LoRA 트리거, 조명 일관성, 공통 네거티브)와 합쳐서 만든다.
"""

# LoRA 트리거 문구 — 첫 패스 프롬프트는 반드시 이 문장으로 시작해야 함 (test.json 노트)
TRIGGER = ("Fill the green spaces according to the image. Outpaint as a seamless 360 equirectangular "
           "panorama (2:1). Keep the horizon level. Match left and right edges.")
LIGHT_TMPL = ("Same {lighting} as the photo all around: identical sun direction, exposure, "
              "white balance and colors across the full 360 degrees; {lighting_detail}.")
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

# 생성 모델 사실 (분석·검수 LLM 이 프롬프트를 이 모델에 맞게 쓰도록 둘 다에 넣음)
MODEL_FACTS = """GENERATOR FACTS (write for this model):
- Model: FLUX.2 Klein 4B, step-distilled: 4 sampling steps at CFG 1. Only the positive prompt has any effect; the
  negative prompt is ignored. Text encoder: Qwen3-4B, which reads plain English sentences and spatial relations well.
- It draws every noun it reads, literally. Naming something (even after "no", "without", "not") makes it appear.
  Negation and orders ("make", "extend", "avoid", "remove") are not understood.
- Earlier sentences weigh more than later ones; a long prompt dilutes the end. The final prompt is about 200 words:
  the code adds a fixed LoRA trigger first, a lighting sentence, constraint sentences and a quality sentence, so never
  add style, quality, camera or lens words yourself.
- LoRA: "360 ERP outpaint" (trigger "Fill the green spaces according to the image..."). The photo sits on a green
  2:1 equirectangular canvas and the model paints the green part; behind the viewer is the part farthest from the
  photo and is invented almost entirely from the words. Typical failures when the words are vague: a second copy of
  the photo's main subject, roads/paths/railings/bridges that fork or stop, translucent overlapping trees or poles
  where two continuations meet, sand or beach ground, people.
- What works: photo-caption sentences with concrete nouns, materials, colors, sizes and positions relative to the
  viewer ("behind the viewer", "to the left", "along the horizon", "underfoot"), one idea per sentence, singular for
  things that appear once, and an explicit statement of how every continuous line (road, path, railing, fence, wall,
  shoreline, power line) runs on around the viewer.
"""

SYSTEM = """You write prompts for a FLUX.2 Klein + 360-ERP-outpaint LoRA pipeline.
The user's single photo is pasted onto a 2:1 equirectangular canvas; everything else (the other ~300 degrees,
sky and ground) is generated from your words, so everything you write must already describe exactly the right
world, in the photo's own mood.

""" + MODEL_FACTS + """
Work in two steps.

STEP 1 - "observed": read the photo carefully and record only what you can actually see.
- place_type: the kind of place (e.g. "lakeside with a long pedestrian bridge", "dirt courtyard of a Korean hall").
- time_weather: time of day and weather as seen.
- light: light direction, shadow length/darkness, contrast, color temperature.
- atmosphere: the mood in 2-5 words, no colors (e.g. "calm, crisp autumn").
- people_amount: "none" if no people are visible; "few_distant" if only a few small or distant people;
  "many" if people are a clear part of the scene (a busy street, a crowd).
- haze_present: true only if fog, mist or haze is clearly visible.
- main_subject: the one main subject in 2-5 words, singular (e.g. "wooden pavilion on the bridge"); "none" if none.
- main_subject_count: how many of the main subject are visible (usually 1; 0 if none).
- subject_is_structure: true only if the main subject is a built object (building, pavilion, bridge, tower, statue,
  gate, boat); false for natural or open things (a path, avenue, lake, forest, mountain, field).
- viewer_standing_on: the solid surface the camera stands on, from the photo's foreground, with an article (e.g. "a lakeside
  stone path", "a wooden bridge deck", "a gravel courtyard"). Water only if the photo is clearly taken from a boat or in the water;
  otherwise choose the nearest plausible shore, deck or path.
- ground_material: the most specific everyday name of what is underfoot, named by its firm main surface
  (packed dirt, compacted earth, gravel, granite paving, asphalt, wooden deck, lawn ...). Use "sand" only when loose
  sand is clearly the main surface, and "beach" only for a real seashore or dunes.
- ground_mistake: the one wrong ground a generator is most likely to paint instead of THIS ground, judged from THIS
  photo (another material, e.g. asphalt for fine gravel, lawn for a dirt yard, loose sand for packed earth);
  "" if there is no clear risk. Never the ground that is actually there.
- ground_details: color, dry/wet, and how plain or littered it looks overall (e.g. "mostly bare, a few grass tufts").
- sky: cloud amount and shape, color, brightness; "not visible" if no sky.
- left_edge / right_edge: what is visible at the photo's left and right borders.
- absent: 3-8 things that are NOT in the photo and a generator might wrongly add here
  (e.g. "people", "second pavilion", "extra bridge", "beach sand", "boats", "buildings on the shore").

STEP 2 - prompt fields, written from STEP 1 only:
- scene: 30-60 words. The place and the main subject exactly as in the photo, with its real materials, colors and
  details. Name the place only if clearly identifiable. Refer to things that appear once in the singular
  ("a pavilion", never "pavilions"). Never mention people, visitors, crowds or activities: the generated
  world must stay empty of people (people already in the photo stay as they are).
- surroundings: 30-55 words, a plain description (not instructions like "extend" or "continue the"). What is to the
  left, right and behind the viewer at a distance: the left_edge / right_edge things (trees, walls, fences, buildings,
  water, hills ...) continued with the same kind, size and spacing as in the photo, so the world feels full, not empty.
  For every continuous line in the photo (road, path, railing, fence, wall, shoreline, power line, bridge deck),
  say in one plain sentence how it runs on around and behind the viewer as a person would expect (one road that
  curves on, a railing that follows the walkway, a shoreline that closes the bay ...). Do not describe the ground or things lying on it here (that is ground_below), and no extra branches, bushes or
  grass in the foreground. Never add a second main subject, new buildings, extra bridges or structures that the
  edges do not suggest.
- ground_below: 6-12 words: viewer_standing_on, its color and how plain it is, exactly like the ground in the
  photo's foreground (e.g. "a packed-earth courtyard, dry beige and mostly bare"). Do not list litter (needles,
  twigs, branches, leaves, pebbles) unless it clearly covers the photo's ground. No camera, tripod, feet, shadows
  or viewing-angle words.
- sky_above: 6-15 words, what is directly overhead. Same weather as the photo.
- lighting: short phrase that matches the photo, e.g. "clear autumn afternoon sunlight".
- lighting_detail: 4-10 words, a lowercase phrase without a final period.
- lighting_short: 1-3 words used in the seam prompt.
- lighting_negatives: lighting conditions that must NOT appear (the opposite times of day / weather).
- content_negatives: 6-12 failure modes specific to THIS scene (e.g. "second pavilion", "extra lake crossing",
  "people on the bridge", "warped railing").
- seam_elements: comma-separated 4-8 continuous background elements behind the viewer (from the edges, ground, sky),
  never the main subject.
- horizon_y: where the true horizon (eye level) sits in the photo, 0=top edge, 1=bottom edge; null if not visible.
- hfov_estimate: horizontal field of view in degrees of the photo (phone main camera ~ 65-75, ultra-wide ~ 100-120).
- summary_ko / tips_ko: Korean, for the operator. tips_ko: risks for this photo and what to adjust.

Style: English, concrete visual nouns, materials and colors. No camera/lens jargon, no style words like
"cinematic", "dramatic", "8k". Haze, mist or fog words only if haze_present is true.
"""

_OBSERVED = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "place_type": {"type": "string"},
        "time_weather": {"type": "string"},
        "light": {"type": "string"},
        "atmosphere": {"type": "string"},
        "people_amount": {"type": "string", "enum": ["none", "few_distant", "many"]},
        "haze_present": {"type": "boolean"},
        "main_subject": {"type": "string"},
        "main_subject_count": {"type": "integer"},
        "subject_is_structure": {"type": "boolean"},
        "viewer_standing_on": {"type": "string"},
        "ground_material": {"type": "string"},
        "ground_mistake": {"type": "string"},
        "ground_details": {"type": "string"},
        "sky": {"type": "string"},
        "left_edge": {"type": "string"},
        "right_edge": {"type": "string"},
        "absent": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["place_type", "time_weather", "light", "atmosphere", "people_amount", "haze_present", "main_subject",
                 "main_subject_count", "subject_is_structure", "viewer_standing_on", "ground_material", "ground_mistake", "ground_details", "sky", "left_edge", "right_edge", "absent"],
}

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "observed": _OBSERVED,                       # 먼저 관찰 (응답이 이 순서로 채워짐)
        "scene": {"type": "string"},
        "surroundings": {"type": "string"},
        "ground_below": {"type": "string"},
        "sky_above": {"type": "string"},
        "lighting": {"type": "string"},
        "lighting_detail": {"type": "string"},
        "lighting_short": {"type": "string"},
        "lighting_negatives": {"type": "array", "items": {"type": "string"}},
        "content_negatives": {"type": "array", "items": {"type": "string"}},
        "seam_elements": {"type": "string"},
        "horizon_y": {"type": ["number", "null"]},
        "hfov_estimate": {"type": "number"},
        "summary_ko": {"type": "string"},
        "tips_ko": {"type": "string"},
    },
    "required": ["observed", "scene", "surroundings", "ground_below", "sky_above", "lighting", "lighting_detail",
                 "lighting_short", "lighting_negatives", "content_negatives", "seam_elements", "horizon_y",
                 "hfov_estimate", "summary_ko", "tips_ko"],
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


import re as _re

_BANNED = r"\b(cinematic|dramatic|epic|8k|4k|hdr|fisheye|wide[- ]angle|bokeh|lens flares?|god ?rays?|sun ?rays?|light beams?)\b"
_HAZE_ADJ = r"\b(hazy|misty|foggy|smoky)\s*"
_HAZE_NOUN = r"\b(haze|mist|fog|smog)\b"
_PEOPLE = r"\b(people|persons?|pedestrians|visitors?|tourists?|crowds?|walkers?|couples?|families|children|hikers?|strollers?)\b"


def _clean(t: str, max_words: int | None = None) -> str:
    """LLM 문장 다듬기: 금지 표현 제거 · 빈자리 정리 · 길이 제한."""
    t = _re.sub(_BANNED, "", t or "", flags=_re.I)
    t = _re.sub(r"\s{2,}", " ", t)
    t = _re.sub(r"\b(with|and|of|in|by)\s+(?=(and|with|of|in|by)\b|[,.;]|$)", "", t, flags=_re.I)
    t = _re.sub(r"\s{2,}", " ", t)
    t = _re.sub(r"\s+([,.;])", r"\1", t).strip(" ,;")
    if max_words:
        w = t.split()
        if len(w) > max_words:
            t = " ".join(w[:max_words]).rstrip(",;") + "."
    return t


def _sentence(t: str) -> str:
    t = t.strip()
    return (t if t.endswith((".", "!", "?")) else t + ".") if t else ""


def _people(o: dict) -> str:
    """none / few_distant / many (예전 분석의 people_present 도 읽음)."""
    v = o.get("people_amount")
    if v in ("none", "few_distant", "many"):
        return v
    return "many" if o.get("people_present", False) else "none"


def _prep(a: dict) -> dict:
    """LLM 해석을 프롬프트용으로 가공: 관찰(observed)에 맞지 않는 말 지우기(안개·사람) + 길이 제한."""
    a = dict(a)
    o = a.get("observed") or {}
    lim = {"scene": 55, "surroundings": 60, "ground_below": 14, "sky_above": 14, "lighting": 12, "lighting_detail": 12,
           "lighting_short": 4, "seam_elements": 30}
    for k, n in lim.items():
        if k in a:
            t = a.get(k) or ""
            if o and not o.get("haze_present", False):
                t = _re.sub(_HAZE_ADJ, "", t, flags=_re.I)
                t = _re.sub(_HAZE_NOUN, "clouds" if k == "sky_above" else "light", t, flags=_re.I)
            if o and k in ("scene", "surroundings", "seam_elements"):   # 생성 영역엔 사람 안 그림 (사진 속 사람은 사진 픽셀로 남음)
                # 사람이 나오는 구절(쉼표 단위)을 통째로 뺌 — "visitors strolling" 같은 말이 남지 않게
                t = ", ".join(f for f in _re.split(r",\s*", t) if not _re.search(_PEOPLE, f, flags=_re.I))
            a[k] = _clean(t, n)
    for k in ("lighting", "lighting_detail", "lighting_short"):   # 템플릿 문장 중간에 들어감: 끝 마침표 없음
        if a.get(k):
            a[k] = a[k].rstrip(" .,;")
    if a.get("lighting_detail"):                          # 소문자 시작
        t = a["lighting_detail"]
        a["lighting_detail"] = t[:1].lower() + t[1:]
    return a


def _constraints(a: dict) -> str:
    """증류 모델(CFG 1)은 네거티브를 안 보므로, 꼭 지켜야 할 것을 Positive 문장으로 (관찰 기반, 사진마다 자동)."""
    o = a.get("observed") or {}
    if not o:
        return ""
    out = []
    subj = _clean(o.get("main_subject") or "", 6)
    n = o.get("main_subject_count")
    structure = o.get("subject_is_structure", True)       # 예전 분석은 필드 없음 → 기존처럼 적용
    if structure and subj and subj.lower() not in ("none", "n/a") and isinstance(n, int) and n >= 1:
        num = {1: "only one", 2: "only two", 3: "only three"}.get(n, f"only {n}")
        out.append(f"There is {num} {subj}, where it is in the photo, and no other similar structure")
    out.append("The whole place is quiet and deserted")   # 사람 수와 관계없이 항상 ('people' 단어는 넣으면 오히려 그림)
    if not o.get("haze_present", False):
        out.append("The air is clear")
    return " ".join(_sentence(x) for x in out)


def _ground_mistake(a: dict) -> str:
    """LLM 이 사진을 보고 고른 '잘못 그려지기 쉬운 바닥' — 실제 바닥과 겹치면 쓰지 않음 (고정 규칙 없음)."""
    o = a.get("observed") or {}
    gx = _clean(o.get("ground_mistake") or "", 5).rstrip(" .")
    if not gx or gx.lower() in ("none", "n/a"):
        return ""
    real = " ".join([o.get("viewer_standing_on") or "", o.get("ground_material") or "", a.get("ground_below") or ""]).lower()
    return "" if gx.lower() in real else gx


def _guard_negatives(a: dict) -> str:
    o = a.get("observed") or {}
    out = [_clean(x, 6) for x in (o.get("absent") or [])]
    if o:
        out.append("people, crowd, pedestrians, figures")
    subj = _clean(o.get("main_subject") or "", 6)
    if subj and subj.lower() not in ("none", "n/a"):
        out.append(f"second {subj}, duplicate {subj}")
    gx = _ground_mistake(a)
    if gx:
        out.append(gx)
    return ", ".join(x for x in out if x)


def assemble(a: dict) -> dict:
    """LLM 재료 → 최종 4개 프롬프트. 생성 모델이 잘 따르도록 정해진 순서의 짧은 문장으로 정리:
    트리거 → 장면(주인공) → 주변·뒤 → 발밑 → 머리 위 → 분위기·빛 → 꼭 지킬 것 → 화질."""
    a = _prep(a)
    o = a.get("observed") or {}
    parts = [TRIGGER, _sentence(a["scene"])]
    if a.get("surroundings"):
        parts.append(_sentence("All around: " + a["surroundings"]))
    if a.get("ground_below"):                              # 서 있는 곳 (물 위에 뜬 시점 방지 — viewer_standing_on 으로 시작)
        g = a["ground_below"].rstrip(" .")
        # 'not X' 는 넣지 않음: 증류 모델(CFG 1)은 부정을 못 알아듣고 X(예: beach sand)를 오히려 그림 → X 는 네거티브에만
        parts.append(_sentence("The viewer stands on " + g[:1].lower() + g[1:]))
    if a.get("sky_above"):
        parts.append(_sentence("Overhead: " + a["sky_above"]))
    if o.get("atmosphere"):
        parts.append(_sentence("Mood: " + _clean(o["atmosphere"], 8)))
    parts.append(LIGHT_TMPL.format(lighting=a["lighting"].strip(), lighting_detail=a["lighting_detail"].strip())
                 if a.get("lighting_detail") else
                 _sentence(LIGHT_TMPL.split(";")[0].format(lighting=a["lighting"].strip())))
    c = _constraints(a)
    if c:
        parts.append(c)
    parts.append(TAIL)
    positive = " ".join(p for p in parts if p)
    guards = _guard_negatives(a)
    negative = _join([", ".join(a["lighting_negatives"]), NEG_COMMON, ", ".join(a["content_negatives"]), guards])
    seam_positive = SEAM_POS_TMPL.format(seam_elements=a["seam_elements"].strip(),
                                         lighting_short=a["lighting_short"].strip())
    seam_negative = _join([", ".join(a["lighting_negatives"]), SEAM_NEG_COMMON, guards])
    return {"positive": positive, "negative": negative,
            "seam_positive": seam_positive, "seam_negative": seam_negative}


# ----------------------------------------------------------------------------- 2K 초안 검수 (업스케일 전)
REVIEW_SYSTEM = """You are the quality checker for a 360 panorama generator used in a VR (HMD) exhibit.
You get a DRAFT (low resolution) generated from one real photo, shown as four flat views taken from the viewer's
position: FRONT (contains the original photo), RIGHT, BACK, LEFT; plus the original photo.
Judge it the way an ordinary visitor would: does the whole place look like one real, natural place that matches
the photo? Ignore softness, small blur and low resolution (it is a draft and will be upscaled later).

Major problems (any one fails the draft):
- roads, paths, railings, fences, walls, shorelines, power lines or bridges that fork, end abruptly, bend
  impossibly, float, or do not continue the way a person would expect;
- structures or objects with no reason to be there, a second copy of the photo's main subject, half objects;
- semi-transparent ghost shapes, double exposure, objects blending into each other;
- people or animals that are not in the photo;
- ground material, weather, sky or time of day that do not match the photo;
- a visible vertical seam, hard brightness step or color band.
Minor problems: small oddities a visitor would hardly notice.

Return: ok (true only if there is no major problem), score 0-10 (10 = indistinguishable from a real 360 photo of
this place), problems (each: view, severity, what is wrong, what a person would expect there instead), fix_hint,
summary_ko (one Korean sentence for the operator).

fix_hint is added to the prompt of the next draft ("" if ok), right after the description of the surroundings.
Follow the generator facts below and write it as a plain description of what IS there, like a photo caption:
- 1-2 short declarative sentences, at most 45 words, most important thing first.
- Only physical things that should be visible, with their material, color, size and where they are relative to the
  viewer ("behind the viewer", "to the left", "along the horizon", "underfoot").
- Describe how lines continue in space: e.g. "Behind the viewer the concrete walkway runs straight on toward a low
  wooded island, lined on both sides by gray posts with four pale cables."
- Never name the defect or what must disappear (no "seam", "line", "ghost", "transparent", "duplicate", "no",
  "without", "remove", "avoid", "instead"); never use verbs that give orders ("make", "extend", "fix", "ensure").
- No camera, lens, style or quality words; no people.

""" + MODEL_FACTS

REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "ok": {"type": "boolean"},
        "score": {"type": "number"},
        "problems": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"view": {"type": "string", "enum": ["front", "right", "back", "left"]},
                           "severity": {"type": "string", "enum": ["major", "minor"]},
                           "what": {"type": "string"}, "expected": {"type": "string"}},
            "required": ["view", "severity", "what", "expected"]}},
        "fix_hint": {"type": "string"},
        "summary_ko": {"type": "string"},
    },
    "required": ["ok", "score", "problems", "fix_hint", "summary_ko"],
}


_HINT_BAD = r"\b(seams?|lines?|ghost(ed|ly|s)?|transparent|translucent|duplicates?|artifacts?|stitch(ing|ed)?)\b"


def with_fix_hints(prompts: dict, hints) -> dict:
    """검수에서 나온 고칠 점(사진 설명 같은 긍정 문장)을 다음 초안 Positive 의 주변 묘사 바로 뒤(서 있는 곳 문장 앞)에 넣음.
    검수 LLM 은 직전 프롬프트를 보고 쓰므로 가장 최근 것 하나만 씀 (쌓지 않음). 결함 이름 단어가 섞여 있으면 그 문장은 뺌."""
    hs = [h for h in hints if (h or "").strip()]
    if not hs:
        return dict(prompts)
    sents = [x for x in _re.split(r"(?<=[.!?])\s+", _clean(hs[-1])) if x and not _re.search(_HINT_BAD, x, flags=_re.I)]
    add = " ".join(_sentence(x) for x in sents)
    if not add:
        return dict(prompts)
    pos = prompts.get("positive", "")
    for anchor in ("The viewer stands on", "Overhead:", TAIL):
        if anchor in pos:
            pos = pos.replace(anchor, add + " " + anchor, 1)
            break
    else:
        pos = pos + " " + add
    return {**prompts, "positive": pos}


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


# ----------------------------------------------------------------------------- 영상화 (부분 움직임 · Cinemagraph)
# 화면 전체가 아니라 '일부만' 움직인다. LLM 은 움직일 곳(위치·움직임)만 정하고, 프롬프트 틀은 코드가 고정.
MOTION_TRIGGER = "CINEMAGRAPH_MOTION"
PROMPT_VER = 3          # 3: 원본에서 크게 안 바뀌는 잔잔한 움직임 강조
MOTION_POS_TMPL = ("{trigger}. A view taken from a 360 degree panorama photograph. Keep the existing image exactly as it is: "
                   "same composition, same objects, same details and colors, only subtle natural motion with minimal "
                   "change from the original. "
                   "Locked-off static camera on a tripod, no camera movement, no pan, no zoom. "
                   "Only the {element} moves: {motion}. "
                   "Everything else in the frame stays perfectly still like a photograph: {keep_still}. "
                   "Seamless natural loop, continuous looping motion, {speed} movement. "
                   "Same {lighting} as the still image. Photorealistic, natural detail.")
MOTION_NEG_COMMON = ("camera movement, pan, tilt, zoom, dolly, orbit, shaky camera, handheld, whole scene moving, "
                     "everything moving, swaying buildings, moving ground, morphing, warping, melting, flicker, "
                     "flashing, lighting change, exposure change, new objects appearing, objects disappearing, people, "
                     "animals, birds, text, watermark, scene cut, fast motion, motion blur, blurry, low quality, "
                     "changing the image, altering the scene, reframing, sliding, swaying image, distorted panorama")
SPEED_WORDS = {"subtle": "very subtle, slow", "slow": "slow, gentle", "medium": "steady, natural"}

MOTION_SYSTEM = """You plan a cinemagraph from a finished 360 equirectangular (ERP) panorama for a Meta Quest 3 exhibit.
The viewer stands still inside the panorama. ONLY A FEW SMALL PARTS MOVE; the rest of the 360 stays a perfectly still photo.
Typical good choices: water surface ripples / flowing stream or waterfall, drifting clouds or mist, hanging willow
branches or leaves swaying in a light breeze, falling leaves, flags or fabric, smoke, candle flame.
Never animate: buildings, rocks, ground, walls, roads, tree trunks, mountains, or the camera. No people or animals.

The first image is the panorama with a yellow/cyan grid: vertical lines = yaw in degrees (-180..180, 0 = centre,
positive = right), horizontal lines = pitch (+ = up). The second image (if given) is the original source photo.
You also get the prompts that generated the panorama — use them to name things correctly.

Return 1 to 4 regions (fewer is better; pick the elements where motion adds the most life). For each region:
- name_ko: short Korean label. element: the moving thing in English (e.g. "pond water surface").
- yaw, pitch: centre of the region in degrees, read from the grid. Keep pitch between -60 and 60.
- hfov: horizontal field of view of the crop in degrees (30-110). The crop must contain the moving element plus a
  little still context around it, and should NOT include other things that could also move.
- orientation: "landscape" (wide) or "portrait" (tall things like a waterfall or hanging willow).
- motion: 8-25 English words describing exactly how it moves (direction, rhythm), loop-friendly (repeating, no start/end).
- speed: subtle | slow | medium. Exhibits look best with subtle or slow.
- strength: cinemagraph LoRA strength 0.8-2.0 (higher = more isolated motion; use ~1.2 normally).
- keep_still: 6-15 English words naming what must stay still INSIDE this crop.
- risk_ko: one short Korean sentence: what could go wrong (e.g. reflections moving, tree trunks bending).
Also: summary_ko (what will move, Korean, 1-2 sentences), static_ko (what stays still overall, Korean),
lighting (short phrase for the light, e.g. "overcast diffuse daylight").
Regions must not overlap each other, and must not cross the panorama's pasted photo border if one is visible.

SPREAD (per region): the motion is NOT confined to the element's outline. Whatever the video model makes move as a
natural consequence (ripples, spray, mist, leaves stirred by it) is kept, fading out with distance from the region.
spread: "none" = only the element itself (isolated things: candle flame, flag), "near" = its natural surroundings
(~35 degrees, the usual choice for water, falls, willows), "wide" = up to ~100 degrees (a breeze through a whole grove;
use rarely). Rigid things never move even inside the spread.

INFLUENCE (optional, per region): motion that starts in a region may physically affect a FEW connected things nearby
(e.g. a waterfall -> ripples spreading on the pool and stream below it, spray making nearby ferns tremble;
a breeze through a willow -> the neighbouring willow branches). List them in "influence" ONLY when it is physically
natural for THIS scene; an empty list is the normal, safe default. Never list rocks, ground, buildings, trunks, sky
or anything rigid. For each: name_ko (Korean), target (English noun phrase for a segmentation model, e.g.
"stream water below the waterfall"), range ("near" = within ~35 degrees of the region, "wide" = up to ~100 degrees,
use wide rarely), strength 0.2-1.0 (how strongly it follows; ripples 0.6, trembling leaves 0.3),
reason_ko (one short Korean sentence why it is affected)."""

MOTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "summary_ko": {"type": "string"}, "static_ko": {"type": "string"}, "lighting": {"type": "string"},
        "regions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "name_ko": {"type": "string"}, "element": {"type": "string"},
                "yaw": {"type": "number"}, "pitch": {"type": "number"}, "hfov": {"type": "number"},
                "orientation": {"type": "string", "enum": ["landscape", "portrait"]},
                "motion": {"type": "string"}, "speed": {"type": "string", "enum": ["subtle", "slow", "medium"]},
                "strength": {"type": "number"}, "keep_still": {"type": "string"}, "risk_ko": {"type": "string"},
                "spread": {"type": "string", "enum": ["none", "near", "wide"]},
                "influence": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {"name_ko": {"type": "string"}, "target": {"type": "string"},
                                   "range": {"type": "string", "enum": ["near", "wide"]},
                                   "strength": {"type": "number"}, "reason_ko": {"type": "string"}},
                    "required": ["name_ko", "target", "range", "strength", "reason_ko"]}},
            },
            "required": ["name_ko", "element", "yaw", "pitch", "hfov", "orientation", "motion", "speed",
                         "strength", "keep_still", "risk_ko", "spread", "influence"]}},
    },
    "required": ["summary_ko", "static_ko", "lighting", "regions"],
}


def clamp_region(r: dict) -> dict:
    r = dict(r)
    r["yaw"] = ((float(r.get("yaw", 0)) + 180) % 360) - 180
    r["pitch"] = max(-70.0, min(70.0, float(r.get("pitch", 0))))
    r["hfov"] = max(20.0, min(120.0, float(r.get("hfov", 60))))
    r["strength"] = max(0.5, min(3.0, float(r.get("strength", 1.2))))
    if r.get("orientation") not in ("landscape", "portrait"):
        r["orientation"] = "landscape"
    if r.get("aspect"):                                 # 자유 비율 (마우스 선택·직접 입력)
        r["aspect"] = round(max(0.5, min(2.0, float(r["aspect"]))), 3)
        r["orientation"] = "portrait" if r["aspect"] < 1 else "landscape"
    else:
        r["aspect"] = round(704 / 512 if r["orientation"] == "landscape" else 512 / 704, 3)
    if r.get("speed") not in SPEED_WORDS:
        r["speed"] = "slow"
    inf = []
    for x in (r.get("influence") or [])[:4]:
        tgt = str(x.get("target", "")).strip()
        if not tgt:
            continue
        inf.append({"name_ko": str(x.get("name_ko", "")).strip() or tgt, "target": tgt,
                    "range": x.get("range") if x.get("range") in ("none", "near", "wide") else "near",
                    "strength": max(0.0, min(1.0, float(x.get("strength", 0.5) or 0))),
                    "reason_ko": str(x.get("reason_ko", "")).strip()})
    r["influence"] = inf
    if r.get("spread") not in ("none", "near", "wide"):
        r["spread"] = "near"
    r["spread_strength"] = max(0.0, min(1.0, float(r.get("spread_strength", 0.7) or 0)))
    return r


GLOBAL_POS_TMPL = ("{trigger}. Equirectangular 360 degree panorama photograph (full spherical view, the left and right "
                   "edges connect). Keep the original panorama exactly as it is: same composition, objects, details, colors "
                   "and lighting; only subtle natural motion with minimal change from the original. "
                   "Locked-off static camera, no camera movement, no zoom. "
                   "{sources} {effects} Everything else stays perfectly still like a photograph: rocks, ground, "
                   "buildings, tree trunks, sky. Small gentle movements only, no new objects, nothing disappears. "
                   "Seamless natural loop, {speed} movement. Same {lighting}. Photorealistic.")


def global_motion_prompt(regions, lighting: str = "") -> dict:
    """360 전체 저해상도 영상용 프롬프트: 원인 + 영향받는 것만 움직임 (흐름만 쓰고 픽셀은 버림)."""
    src = " ".join(f"The {r['element'].strip().rstrip('.')} moves: {r['motion'].strip().rstrip('.')}." for r in regions or []
                   if r.get("element") and r.get("motion"))
    eff = [x["target"].strip().rstrip(".") for r in regions or [] for x in r.get("influence") or []
           if x.get("range") in ("near", "wide") and x.get("strength", 0) > 0]
    effects = ("Affected by it, only these respond gently: " + "; ".join(eff) + ".") if eff else \
        "Only things physically affected by it respond gently nearby (ripples, spray, mist, stirred leaves)."
    speeds = [r.get("speed", "slow") for r in regions or []]
    speed = SPEED_WORDS.get("subtle" if "subtle" in speeds else "slow" if "slow" in speeds else "medium")
    return {"positive": GLOBAL_POS_TMPL.format(trigger=MOTION_TRIGGER, sources=src, effects=effects, speed=speed,
                                               lighting=(lighting or "natural daylight").strip()),
            "negative": MOTION_NEG_COMMON}


def motion_prompts(r: dict, lighting: str = "natural daylight") -> dict:
    """영역 하나 → Cinemagraph 프롬프트 (트리거·카메라 고정·'이것만 움직임'·반복은 고정 문구)."""
    pos = MOTION_POS_TMPL.format(trigger=MOTION_TRIGGER, element=r["element"].strip().rstrip("."),
                                 motion=r["motion"].strip().rstrip("."), keep_still=r["keep_still"].strip().rstrip("."),
                                 speed=SPEED_WORDS.get(r.get("speed"), "slow, gentle"), lighting=(lighting or "natural daylight").strip())
    if r.get("spread", "near") in ("near", "wide"):        # 요소 모양에 가두지 않음 — 그 영향은 허용, 단단한 것만 정지
        pos = pos.replace(f"Only the {r['element'].strip().rstrip('.')} moves:",
                          f"The {r['element'].strip().rstrip('.')} moves:")
        pos = pos.replace("Everything else in the frame stays perfectly still like a photograph:",
                          "Its natural effects on the immediate surroundings (ripples, spray, mist, stirred leaves) may move "
                          "gently; rigid things stay perfectly still like a photograph:")
    return {"positive": pos, "negative": MOTION_NEG_COMMON}


# ============================================================================= 사건 영상 (한 번 일어나는 연출)
EVENT_SYSTEM = """You direct a ONE-SHOT dramatic event inside a finished 360 equirectangular panorama for a Meta Quest 3
exhibit. The viewer stands still. A short video (about 5 seconds) is generated from a wide flat view cut from the
panorama: it STARTS calm and identical to the still image, a dramatic natural event happens, then it SETTLES back to
calm, ending close to the original image. The camera never moves. Typical events: a waterfall or stream upstream
suddenly swells, a surge of water pours through (e.g. under a stone bridge arch), the pool rises and OVERFLOWS the rocks,
white water and spray rush toward the viewer and splash across the view, then the water drains back and calms;
or a huge mass of water bursts from a waterfall and mist covers the view.

The first image is the panorama with a yaw/pitch grid (vertical lines = yaw -180..180, 0 = centre, + = right;
horizontal = pitch, + = up). The second (if given) is the original photo. You also get the generation prompts.

Return:
- name_ko: short Korean title. summary_ko: 1-2 Korean sentences describing the event for the operator.
- source: the thing the event comes from, English noun phrase (e.g. "the waterfall").
- yaw, pitch: centre of the view to generate (read from the grid). Put the source a little above centre so the
  event can spread toward the viewer. pitch between -30 and 30.
- hfov: horizontal field of view of that view, 100-130 (wide, so the event can fill the viewer's sight).
- calm_start: 10-20 English words, how the scene looks at the start (= the still image, gentle natural motion only).
- event: 25-45 English words, what happens, in time order, physically plausible for THIS scene, moving toward the
  viewer, big and dramatic, but no destruction of buildings and no new objects except water/spray/mist/leaves.
- peak: 10-25 English words, the moment of maximum coverage (e.g. "white spray and mist fill most of the view").
- calm_end: 10-20 English words, how it settles (water recedes, mist thins, scene returns to the original look).
- keep: 6-15 English words, rigid things that must stay in place (rocks, pavilion, stone walls, tree trunks).
- risk_ko: one short Korean sentence on what could go wrong."""

EVENT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {k: {"type": "string"} for k in ("name_ko", "summary_ko", "source", "calm_start", "event", "peak",
                                                   "calm_end", "keep", "risk_ko")} |
                  {k: {"type": "number"} for k in ("yaw", "pitch", "hfov")},
    "required": ["name_ko", "summary_ko", "source", "yaw", "pitch", "hfov", "calm_start", "event", "peak", "calm_end",
                 "keep", "risk_ko"],
}

EVENT_POS_TMPL = ("A locked-off static camera on a tripod, a wide photographic view of this exact scene. "
                  "The video begins calm and identical to the input image: {calm_start}. "
                  "Then {event} "
                  "At the peak, {peak}. "
                  "Afterwards it settles: {calm_end}, ending like the input image. "
                  "Throughout, {keep} stay exactly in place. One continuous shot, no camera movement, no cuts, "
                  "photorealistic, {lighting}.")
EVENT_NEG = ("camera movement, pan, tilt, zoom, dolly, shaky camera, scene cut, transition, different scene, "
             "buildings collapsing, structures moving, morphing, melting, warping rocks, new objects, people, animals, "
             "text, watermark, cartoon, CGI look, oversaturated, flicker, low quality, blurry")


def clamp_event(e: dict) -> dict:
    e = dict(e)
    e["yaw"] = ((float(e.get("yaw", 0)) + 180) % 360) - 180
    e["pitch"] = max(-40.0, min(40.0, float(e.get("pitch", 0))))
    e["hfov"] = max(60.0, min(140.0, float(e.get("hfov", 120))))
    for k in ("name_ko", "summary_ko", "source", "calm_start", "event", "peak", "calm_end", "keep", "risk_ko"):
        e[k] = str(e.get(k, "") or "").strip()
    return e


def event_prompts(e: dict, lighting: str = "") -> dict:
    s = lambda k: e.get(k, "").strip().rstrip(".")
    pos = EVENT_POS_TMPL.format(calm_start=s("calm_start"), event=s("event") + ".", peak=s("peak"), calm_end=s("calm_end"),
                                keep=s("keep") or "rocks, buildings and tree trunks",
                                lighting=(lighting or "natural daylight").strip())
    return {"positive": pos, "negative": EVENT_NEG}


# ============================================================================= 영상 노드 파일용 프롬프트 (LTX-2.3)
# 360 전체 영상 (LTX-2.3 + 360 LoRA 에 360 이미지를 그대로 넣을 때)
LTX360_POS = ("Equirectangular 360 degree panorama, immersive photorealistic VR video. Locked camera and fixed horizon. "
              "The video starts calm and exactly like the input panorama. {event} "
              "Then the water calms down and the scene settles, ending exactly like the input panorama. "
              "{keep} remain solid and perfectly still. Continuous, natural motion.")
LTX360_EVENT = ("A massive flood of water surges from the waterfall through the scene: waves crash and foam over the rocks, "
                "spray and mist rise toward the viewer.")
LTX360_NEG = ("camera movement, camera pan, zoom, dolly, rotation, shaking, moving bridge, moving rocks, morphing stonework, "
              "warped architecture, new objects, people, animals in view, flickering, boiling textures, visible seam, "
              "non-equirectangular perspective, scene cut, text, subtitles, watermark, music, speech, voices")

# 크롭 생성 (칠한 범위만 일반 영상으로 생성 → 서버가 4K 360 에 붙임)
LTX_CROP_POS = ("Wide-angle photorealistic nature video, locked-off static camera, no camera movement. "
                "The video starts exactly like the input photo. {event} "
                "Natural real-time speed, no fast motion. "
                "{keep} stay perfectly still. Clear liquid water with sharp detailed ripples, spray and foam that follow the flow.")
LTX_CROP_NEG = ("camera movement, camera pan, zoom, dolly, rotation, shaking, morphing rocks, changing cliff, new structures, "
                "warping background, floating objects, people, animals, thick fog, mist covering the scene, smoke, steam, "
                "milky blurred water, frozen water, water flowing over dry ground, smooth sliding sheet of foam, "
                "polygonal faceted texture, blocky artifacts, ghosting, semi-transparent overlay, trails, frame blending, "
                "smeared texture, painterly, oversharpened, flickering, boiling textures, scene cut, text, subtitles, "
                "watermark, music, speech, voices")


def video_event_prompt(ev: dict, crop: bool = False) -> str:
    """영상 노드 기본 Positive: 사건 만들기 내용(없으면 기본 물 넘침) + 제자리 유지할 것.
    crop=False → 360 전체용(LTX360_POS), True → 크롭 생성용(LTX_CROP_POS)."""
    if ev.get("event"):
        event = ev["event"].strip().rstrip(".") + "." + (f" At the peak, {ev['peak'].strip().rstrip('.')}." if ev.get("peak") else "")
    else:
        event = LTX360_EVENT
    default_keep = "The rocks, cliffs, trees and buildings" if crop else "The bridge, rocks, cliffs and buildings"
    keep = (ev.get("keep") or default_keep).strip().rstrip(".")
    return (LTX_CROP_POS if crop else LTX360_POS).format(event=event, keep=keep[:1].upper() + keep[1:])


def event_timeline(seconds) -> str:
    """영상 길이에 맞춘 사건 시간표 (장면과 무관한 공통 문장). 길이를 늘리지 않고 사건의 크기·속도를 길이에 맞춰
    시작 → 절정 → 끝(첫 화면과 같음)이 반드시 다 나오게 한다. 노드의 마지막 프레임 = 사진(1226)과 함께 씀."""
    s = float(seconds)
    a, b, c = round(s * 0.15, 1), round(s * 0.5, 1), round(s * 0.85, 1)
    return (f"Timing for this {s:g}-second clip: 0-{a:g}s calm, identical to the input photo; "
            f"{a:g}-{b:g}s the event appears and builds to its peak; {b:g}-{c:g}s it passes and subsides; "
            f"{c:g}-{s:g}s everything has settled and looks exactly like the first frame. "
            "The whole event happens and completely finishes inside the clip, sized to fit this short time; "
            "nothing is cut off mid-action.")
