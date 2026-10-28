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
# 360 고정 문구 (LLM 결과와 상관없이 항상): 하늘·발밑이 한 점으로 자연스럽게 모이고 주인공은 한 번만
SPHERE_TAIL = ("The sky continues smoothly and evenly up to the zenith and the ground continues evenly beneath the "
               "camera. The main subject appears only once.")
NEG_360 = ("radial streaks, light rays, sun rays, sunbeams, lens flare, sun disk overhead, starburst, star-shaped pattern in "
           "the sky, pinched sky, vertical line in the sky, sky color bands, duplicated main subject, second copy of the "
           "building, ghosted semi-transparent objects, floating structure, object hanging in the sky, structure cut in half, "
           "pillar directly under the camera, ground material changing abruptly, patchwork ground, railing ending abruptly, "
           "mismatched horizon height")
# LLM 이 써도 코드에서 지우는 말 (360 에서 별무늬·이음새·과장 문제를 만드는 표현)
_BANNED = (r"\b(sun ?rays?|sunbeams?|god ?rays?|light (?:rays?|beams?)|crepuscular rays?|lens flares?|starbursts?|"
           r"rainbows?|dramatic|cinematic|epic|8k|4k|hdr|fisheye|wide[- ]angle|bokeh|tilt[- ]shift)\b")

NEG_COMMON = ("inconsistent lighting, different time of day, exposure mismatch, brightness gradient, color shift, "
              "color cast, green tint, desaturated area, visible seams, mismatched lighting, repeated textures, "
              "warped perspective, blurry edges, low quality, out of frame, floating objects, mismatched colors, "
              "hard edges, garbled text, gibberish lettering, captions, watermark, worst quality, blurry, ugly, split screen, collage, frame, "
              "black borders, broken horizon, duplicated objects, duplicate people, HDR look, oversharpened, oversaturated, "
              "overexposed sky, blown-out highlights, halos around edges, painterly texture, smeared ground, swirl at the bottom")
SEAM_POS_TMPL = ("Fill the green spaces according to the image. Repair only the central vertical seam of this 360 equirectangular panorama. Seamlessly continue "
                 "the {seam_elements} across the band. The ground stays the same material and pattern from left to right "
                 "({ground}); the sky stays the same ({sky}); the horizon line, railings, walls and paths line up and run "
                 "straight through without ending, bending or changing height. Nothing new appears inside the band. "
                 "Match the brightness, white balance, color, sharpness and {lighting_short} "
                 "of both sides exactly; the band must not be darker, lighter, hazier or softer than its neighbors. "
                 "Photorealistic, no visible join.")
SEAM_NEG_COMMON = ("new object in the band, wall end, building edge, object cut in half, "
                   "different ground material, material change, texture change, paving pattern change, misaligned railing, "
                   "broken wall, step in the horizon, haze patch, foggy band, blurry band, semi-transparent ghost, "
                   "inconsistent lighting, exposure mismatch, brightness gradient, color shift, color cast, green tint, "
                   "desaturated area, split screen, vertical dividing line, hard edge, mismatched horizon, text, watermark")

SYSTEM = """You are a 360 VR panorama continuity specialist. You write prompts for a FLUX.2 Klein + 360-ERP-outpaint LoRA pipeline.
Your top priority is a world that is physically continuous all the way around the viewer: the left and right edges of the
panorama (0/360 degrees, directly behind the viewer) must join invisibly; the ground is one continuous surface with the same
material and pattern; the sky is one continuous sky; nothing appears in a strange place (no structure floating in the air,
cut in half at the seam, standing directly under or over the camera, or sticking out of the ground or sky). A plain, boring,
continuous surrounding is always better than an interesting but discontinuous one.
The user's single photo is pasted onto a 2:1 equirectangular canvas; everything else (the other ~300 degrees,
sky and ground) is generated. Your job is to describe the WHOLE surrounding environment so the outpaint is
plausible, consistent and high quality.

Rules:
- English, concrete visual nouns and materials. No camera/lens jargon, no style words like "cinematic", "8k".
- scene: one or two sentences. Start with the place (name it only if clearly identifiable, e.g. a famous landmark),
  then the main subject, then what plausibly surrounds it in all directions including behind the camera,
  the ground directly beneath the camera (material and texture, since VR viewers look straight down) and the sky
  overhead (clouds/haze/clear, so it is not a flat white blank). 25-60 words.

360 safety rules (the panorama is viewed in a VR headset; break these and the result shows seams, streaks or duplicates):
- The main subject of the photo exists ONCE, in front. Never describe it again for the sides or behind the camera.
- Behind the camera and to the sides, describe only simple, continuous surroundings that already appear at the
  photo's left/right edges (same ground material, same low distant background such as treeline, sea horizon, wall).
  Do not invent new landmarks, tall buildings, towers, large trees or anything close to the camera.
- Sky: describe it as smooth and continuous up to the zenith, same weather as the photo. No sun disk overhead, no sun
  rays, light beams, lens flare, rainbows, dramatic or radial clouds. Only mention a canopy, roof or ceiling overhead
  if the photo itself is clearly taken under one.
- Ground: one continuous, evenly textured material directly beneath; no holes, water or objects right below the camera
  unless the photo is clearly taken over water.
- Materials: name ONE ground material (and one wall/railing material if any) and keep it everywhere; never switch from
  paving to grass, sand to water, wood to stone etc. around the viewer unless the photo itself shows that change.
- Structures (railings, walls, paths, shorelines, horizons) continue at the same height and line all the way around and
  meet themselves behind the viewer; they never end abruptly, float, or change height.
- Keep lettering that is in the photo; never ask to remove or change it.
- lighting: short phrase that matches the photo, e.g. "natural daytime lighting", "warm late-afternoon sunlight",
  "overcast diffuse daylight", "night street lighting".
- lighting_detail: 4-10 words, e.g. "soft daylight, clear consistent sky".
- lighting_short: 1-3 words used in the seam prompt, e.g. "daylight", "night lighting".
- lighting_negatives: lighting conditions that must NOT appear (the opposite times of day / weather).
  For a daytime photo: night, dusk, sunset, twilight, dark sky, stars, moon, artificial lights, day-night split.
- content_negatives: 6-12 failure modes specific to THIS scene (e.g. "distorted bridge geometry", "duplicated rocks",
  "invented architectural details", "unnatural water flow", "sky on the bottom", "paving turning into grass",
  "railing ending mid-air").
  Never list text, letters, signs or signboards themselves as negatives: real lettering in the photo must be kept.
- seam_elements: comma-separated list of 4-8 elements that must continue across the seam band (behind the camera):
  only continuous background elements (sky, ground material, distant horizon, treeline, wall), never the main subject.
- horizon_y: where the true horizon (eye level) sits in the photo, 0=top edge, 1=bottom edge; null if not visible.
- hfov_estimate: horizontal field of view in degrees of the photo (phone main camera ~ 65-75, ultra-wide ~ 100-120).
- ground_below: 8-20 words, what is directly beneath the camera seen from straight above (e.g. "weathered flat granite
  paving stones with thin grass in the joints and a few fallen yellow leaves"). No camera, tripod, feet or shadows.
- sky_above: 6-15 words, what is directly overhead looking straight up (e.g. "soft grey overcast sky with light mist").
  Plain and even; no sun, rays or distinct objects unless the photo is under a canopy/roof.
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


_TEXT_WORDS = ("text", "letter", "lettering", "sign", "signs", "signage", "signboard", "caption", "captions",
               "writing", "inscription", "characters", "calligraphy")
_TEXT_OK = ("garbled", "gibberish", "watermark", "distorted lettering")


def photo_safe_negative(neg: str) -> str:
    """원본 사진의 글씨(현판·간판)를 지키는 네거티브: 'text', 'invented signs or lettering' 처럼 글씨 자체를 막는 항목은 빼고
    'garbled text, gibberish lettering'(깨진 글씨)만 막음. 사진에 있던 글씨를 모델이 지우거나 뭉개지 않게."""
    import re
    keep = []
    for p in (neg or "").split(","):
        t = p.strip()
        if not t:
            continue
        words = set(re.findall(r"[a-z]+", t.lower()))
        if words & set(_TEXT_WORDS) and not any(k in t.lower() for k in _TEXT_OK):
            continue
        keep.append(t)
    return _join([", ".join(keep), "garbled text, gibberish lettering"])


def _clean(t: str, max_words: int | None = None) -> str:
    """LLM 문장에서 금지 표현 제거 + 길이 제한."""
    import re
    t = re.sub(_BANNED, "", t or "", flags=re.I)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\b(with|and|of|in|by)\s+(?=(and|with|over|under|in|on|of|by)\b|[,.;]|$)", "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+([,.;])", r"\1", t).strip(" ,;")
    if max_words:
        w = t.split()
        if len(w) > max_words:
            t = " ".join(w[:max_words]).rstrip(",;") + "."
    return t


def sanitize_analysis(a: dict) -> dict:
    """LLM 분석 결과를 360 에 안전하게 (지시를 어겨도 코드에서 한 번 더 막음):
    금지 표현 제거 · 길이 제한 · 글씨를 막는 네거티브 제거 · 개수 제한."""
    a = dict(a)
    a["scene"] = _clean(a.get("scene", ""), 70)
    for k, n in (("lighting", 8), ("lighting_detail", 12), ("lighting_short", 4), ("ground_below", 24), ("sky_above", 18)):
        if k in a:
            a[k] = _clean(a.get(k, ""), n)
    a["seam_elements"] = _clean(a.get("seam_elements", ""), 30)
    cn = [x for x in (_clean(c, 10) for c in a.get("content_negatives") or []) if x]
    cn = [x for x in photo_safe_negative(", ".join(cn)).split(", ") if x not in ("garbled text", "gibberish lettering")]
    a["content_negatives"] = cn[:12]
    a["lighting_negatives"] = [x for x in (_clean(c, 6) for c in a.get("lighting_negatives") or []) if x][:10]
    return a


def assemble(a: dict) -> dict:
    """LLM 재료 → 최종 4개 프롬프트."""
    a = sanitize_analysis(a)
    positive = " ".join([TRIGGER, a["scene"].strip(),
                         LIGHT_TMPL.format(lighting=a["lighting"].strip(), lighting_detail=a["lighting_detail"].strip()),
                         SPHERE_TAIL, TAIL])
    negative = _join([", ".join(a["lighting_negatives"]), NEG_COMMON, NEG_360, ", ".join(a["content_negatives"]),
                      "garbled text, gibberish lettering"])
    seam_positive = SEAM_POS_TMPL.format(seam_elements=a["seam_elements"].strip(),
                                         ground=(a.get("ground_below") or "the same ground as both sides").strip().rstrip("."),
                                         sky=(a.get("sky_above") or "the same sky as both sides").strip().rstrip("."),
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
    a = sanitize_analysis(a)
    ground = (a.get("ground_below") or "natural ground matching the surrounding area").strip()
    sky = (a.get("sky_above") or "sky matching the surrounding sky").strip()
    pos = POLE_POS_TMPL.format(ground=ground, sky=sky, lighting_short=(a.get("lighting_short") or "daylight").strip())
    neg = _join([", ".join(a.get("lighting_negatives") or []), POLE_NEG])
    return {"positive": pos, "negative": neg}


REFINE_TMPL = ("A high-resolution photograph. {scene} Natural fine detail, realistic textures of stone, wood, "
               "foliage and water, crisp but not oversharpened, true-to-life colors, natural contrast.")
REFINE_NEG = ("painting, illustration, cartoon, anime, digital art, CGI, 3d render, watercolor, flat colors, "
              "posterized, smooth plastic texture, oversharpened, halos, HDR look, oversaturated, blurry, "
              "garbled text, gibberish lettering, watermark")


def refine_prompts(a: dict, fallback_positive: str = "") -> dict:
    scene = _clean(a.get("scene") or "", 70)
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
