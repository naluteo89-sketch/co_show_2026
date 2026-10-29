"""
LLM 분석(사실) → 파이썬 규칙 → 프롬프트 결정.

LLM 은 사진에서 '보이는 사실'만 작은 항목으로 답하고(facts), 허용·금지 같은 결론은 여기서 규칙으로 정한다.
- LLM 판단을 그대로 믿지 않음: 사진 속 근거(개수·가장자리 종류 등)가 있을 때만 허용하고, 확신이 낮으면 안전한 쪽으로.
- 극단적으로 나누지 않음: 사람·주변 건물은 3단계.
- 예전 분석(facts 없음)은 예전 동작 그대로.
"""

import re

PLACES = ["waterside_walkway_or_bridge", "coastal_road", "park_or_garden", "forest_trail",
          "palace_or_temple_courtyard", "plaza", "city_street", "market", "mountain_or_viewpoint", "other"]
SURFACES = ["earth", "gravel", "paving", "asphalt", "wooden_deck", "bridge_deck", "rock", "grass", "sand", "water"]
LINE_KINDS = ["road", "path", "walkway", "railing", "fence", "wall", "shoreline", "power_line", "bridge_deck",
              "hedge", "tree_row", "other"]
EDGE_KINDS = ["water", "trees", "buildings", "road", "wall", "open_sky", "rocks", "field", "mixed"]
LEVELS = ["high", "medium", "low"]

_S = lambda: {"type": "string"}
_E = lambda vals: {"type": "string", "enum": vals}
_OBJ = lambda props: {"type": "object", "additionalProperties": False, "properties": props, "required": list(props)}

FACTS_SCHEMA = _OBJ({
    "place_category": _E(PLACES),
    "standing": _OBJ({"surface": _E(SURFACES), "over_water": {"type": "boolean"},
                      "eye_height": _E(["ground_level", "standing", "elevated"])}),
    "people": _OBJ({"count_in_photo": _E(["0", "1-3", "4-10", "10+"]),
                    "nearest": _E(["none", "near", "mid", "far"]), "evidence": _S()}),
    "subject": _OBJ({"kind": _E(["unique_landmark", "repeated_type", "natural", "none"]),
                     "similar_count_in_photo": {"type": "integer"}, "evidence": _S()}),
    "edges": _OBJ({side: _OBJ({"kind": _E(EDGE_KINDS), "density": _E(["none", "sparse", "dense"]), "what": _S()})
                   for side in ("left", "right")}),
    "lines": {"type": "array", "items": _OBJ({
        "kind": _E(LINE_KINDS), "name": _S(), "side": _E(["left", "center", "right", "across"]),
        "course": _E(["straight", "curves_left", "curves_right", "ends"]), "ends_at": _S()})},
    "confidence": _OBJ({k: _E(LEVELS) for k in ("people", "subject", "edges", "lines", "standing")}),
})

FACTS_GUIDE = """STEP 1b - "facts": short, checkable facts the code turns into rules. Answer only what you can see; when
unsure, choose the safer value and set that group's confidence to "low".
- place_category: the closest kind of place.
- standing: surface the camera stands on (from the photo's foreground); over_water true only if shot from a boat or
  the water itself; eye_height: ground_level (very low), standing (normal), elevated (from a height).
- people: how many people are visible in the photo and how close the nearest one is; evidence = where they are.
- subject: unique_landmark if the main subject is a one-of-a-kind built thing (a pavilion, tower, gate, lighthouse,
  a bridge); repeated_type if more of the same kind naturally exist around (apartment blocks, houses, street lamps,
  boats); natural for trees, lakes, mountains, paths; similar_count_in_photo = how many of that kind are visible.
- edges: at the photo's left and right borders, the main kind of thing, how dense, and a few words naming it
  (a plain noun phrase, e.g. "dense green pines", "a red walkway with a silver railing").
- lines: every continuous line visible (road, path, railing, fence, wall, shoreline, power line, bridge deck,
  hedge, row of trees), max 4: name (2-5 words, e.g. "the asphalt coastal road"), where it is, and how it runs on
  out of the photo (straight, curving left/right, or where it ends, e.g. "ends at the shore").
- confidence: how sure you are about each group.
"""

_BUILT = r"\b(buildings?|houses?|apartments?|apartment blocks?|blocks|towers?|shops?|stores?|hotels?|offices?|" \
         r"factor(y|ies)|warehouses?|skyscrapers?|villas?|cottages?|huts?|sheds?|roofs?)\b"


def _conf(f, k):
    return ((f.get("confidence") or {}).get(k)) or "medium"


def decide(a: dict) -> dict:
    """facts(+observed) → 결정. 결정마다 이유(why)를 남겨 기록·화면에서 확인할 수 있게."""
    o = a.get("observed") or {}
    f = a.get("facts") or {}
    why = []
    if not f:                                              # 예전 분석: 예전 규칙
        return {"people": "none", "subject_unique": bool(o.get("subject_is_structure", True)),
                "built": "keep", "lines": [], "ground_fix": "", "why": ["facts 없음 → 예전 규칙"]}

    # 사람: 기본은 없음. 사진에 사람이 보이고(근거), 붐비는 장소이거나 여러 명일 때만 '멀리 작게 몇 명'.
    p = f.get("people") or {}
    cnt, near = p.get("count_in_photo", "0"), p.get("nearest", "none")
    place = f.get("place_category", "other")
    people = "none"
    if _conf(f, "people") == "low" or cnt == "0" or not (p.get("evidence") or "").strip():
        why.append(f"사람: 없음 (사진 {cnt}명, 확신 {_conf(f, 'people')})")
    elif cnt in ("4-10", "10+") or (cnt == "1-3" and place in ("city_street", "plaza", "market") and near != "near"):
        people = "few_far"
        why.append(f"사람: 멀리 작게 몇 명 (사진 {cnt}명, {place})")
    else:
        why.append(f"사람: 없음 (사진 {cnt}명뿐, {place})")

    # 주인공: 하나뿐인 랜드마크이고 사진에도 하나일 때만 'only one'
    s = f.get("subject") or {}
    unique = (s.get("kind") == "unique_landmark" and int(s.get("similar_count_in_photo") or 1) <= 1
              and _conf(f, "subject") != "low")
    why.append(f"주인공: {'하나뿐' if unique else '제한 안 함'} ({s.get('kind')}, 사진 {s.get('similar_count_in_photo')}개)")

    # 주변 건물: 가장자리에 건물·도로·담이 실제로 있을 때만 이어짐 허용
    e = f.get("edges") or {}
    built_sides = [k for k in ("left", "right") if (e.get(k) or {}).get("kind") in ("buildings", "road", "wall", "mixed")
                   and (e.get(k) or {}).get("density") != "none"]
    dense = any((e.get(k) or {}).get("kind") == "buildings" and (e.get(k) or {}).get("density") == "dense"
                for k in ("left", "right"))
    if not built_sides:
        built = "none"
    elif dense and _conf(f, "edges") != "low":
        built = "continuous"
    else:
        built = "sparse"
    why.append(f"주변 건물: {built} (가장자리 {', '.join(built_sides) or '자연만'})")

    # 좌우 가장자리가 다른 종류(물↔숲 등)면 뒤쪽에서 둘이 만나는 곳을 새로 지어야 함 → 원통 패스를 더 세게
    lk, rk = (e.get("left") or {}).get("kind"), (e.get("right") or {}).get("kind")
    edges_differ = bool(lk and rk and lk != rk and _conf(f, "edges") != "low")
    why.append(f"좌우 가장자리: {lk} / {rk} → {'다름 (뒤쪽 이음 문장·원통 패스 강하게)' if edges_differ else '비슷함'}")

    # 이어지는 선: 확신이 낮지 않으면 '어떻게 이어지는지' 문장
    lines = []
    if _conf(f, "lines") != "low":
        for ln in (f.get("lines") or [])[:4]:
            name = (ln.get("name") or ln.get("kind") or "").strip().rstrip(".")
            if not name:
                continue
            course = ln.get("course")
            tail = {"straight": "runs straight on and continues all around",
                    "curves_left": "curves gently to the left and continues on",
                    "curves_right": "curves gently to the right and continues on"}.get(course)
            if course == "ends":
                end = (ln.get("ends_at") or "").strip().rstrip(".")
                tail = f"comes to a natural end at {end}" if end else None
            if tail:
                lines.append((ln.get("kind", "other"), f"{name[:1].upper()}{name[1:]} {tail}."))
    why.append(f"이어지는 선: {len(lines)}개")

    # 서 있는 곳: 물 위가 아닌데 발밑 문장이 물이면 서 있는 곳으로 바꿈
    st = f.get("standing") or {}
    ground_fix = ""
    gb = (a.get("ground_below") or "").lower()
    if not st.get("over_water") and re.search(r"\b(water|lake|sea|river|pond)\b", gb.split(",")[0]):
        ground_fix = (o.get("viewer_standing_on") or "").strip()
        why.append(f"발밑: 물 → {ground_fix}")
    return {"people": people, "subject_unique": unique, "built": built, "lines": lines, "ground_fix": ground_fix,
            "edges_differ": edges_differ, "why": why}


def behind_fallback(a: dict) -> str:
    """LLM 이 뒤쪽 문장(behind)을 안 줬을 때(예전 분석): 좌우 가장자리 사실만으로 '뒤에서 어떻게 이어지는지' 한 문장.
    사진 오른쪽 끝 → 오른쪽 → 뒤 → 왼쪽 → 사진 왼쪽 끝 순서 (원통 한 바퀴). 새 물체는 만들지 않음."""
    f = a.get("facts") or {}
    o = a.get("observed") or {}
    e = f.get("edges") or {}
    r = ((e.get("right") or {}).get("what") or o.get("right_edge") or "").strip().rstrip(".")
    l = ((e.get("left") or {}).get("what") or o.get("left_edge") or "").strip().rstrip(".")
    if not r or not l or r.lower() == l.lower():
        return ""
    return f"Behind, there is a gradual, natural change from {r} on the right side to {l} on the left side."


def filter_built(text: str, scene: str, edges_text: str) -> str:
    """주변 묘사에서, 사진(장면·가장자리)에 근거 없는 건물 문장만 뺌 (환각 방지 · 근거 있으면 그대로)."""
    keep = []
    seen = (scene + " " + edges_text).lower()
    for sen in re.split(r"(?<=[.!?])\s+", text or ""):
        words = {m.group(0).lower().rstrip("s") for m in re.finditer(_BUILT, sen, flags=re.I)}
        if words and not any(w in seen for w in words):
            continue
        keep.append(sen)
    return " ".join(keep).strip()


def line_sentences(dec: dict, surroundings: str) -> str:
    """주변 묘사에 아직 없는 선만 문장으로 (같은 말 두 번 넣지 않음)."""
    out, low = [], (surroundings or "").lower()
    told = re.search(r"\b(continu\w*|runs?|curv\w*|follows?|bends?|ends?)\b", low)
    for kind, sen in dec.get("lines", []):
        noun = re.sub(r"[^a-z\-]", "", sen.split(" runs ")[0].split(" curves ")[0].split(" comes ")[0].split()[-1].lower())
        if told and noun and noun in low:                 # 주변 묘사가 이미 이 선의 이어짐을 말함
            continue
        out.append(sen)
    return " ".join(out[:3])
