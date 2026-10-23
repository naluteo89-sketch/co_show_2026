"""
ComfyUI 그래프 공통: UI 포맷(.json, 화면에서 저장한 노드 파일) → API 포맷 변환 · 노드 만들기 · UI 위젯 값 바꾸기
"""

import json
import random

UI_ONLY = {"MarkdownNote", "Note", "PreviewImage"}
SKIP_WIDGETS = {"upload", "control_after_generate"}


def node(class_type, inputs, title):
    """ComfyUI API 노드의 공통 형식."""
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


def ksampler(model, positive, negative, latent, *, seed, steps, cfg, denoise, title,
             sampler_name="euler", scheduler="simple"):
    return node("KSampler", {"model": model, "positive": positive, "negative": negative, "latent_image": latent,
                             "seed": int(seed), "steps": int(steps), "cfg": float(cfg), "sampler_name": sampler_name,
                             "scheduler": scheduler, "denoise": float(denoise)}, title)


def random_seed(seed=None) -> int:
    return int(seed) if seed is not None else random.randint(0, 2**31 - 1)


def load_template(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ui_to_api(ui: dict) -> dict:
    nodes = {n["id"]: n for n in ui["nodes"]}
    links = {l[0]: l for l in ui["links"]}
    api = {}
    for n in ui["nodes"]:
        if n["type"] in UI_ONLY or n.get("mode", 0) in (2, 4):      # 2=mute, 4=bypass
            continue
        named = n.get("widgets_values_named")
        if named is None and n.get("widgets_values"):
            raise ValueError(f"node {n['id']} ({n['type']}) 에 widgets_values_named 가 없어요. "
                             "최신 ComfyUI 프론트엔드에서 워크플로를 다시 저장해 주세요.")
        inputs = {k: v for k, v in (named or {}).items() if k not in SKIP_WIDGETS}
        for i in n.get("inputs", []):
            lid = i.get("link")
            if lid is None:
                continue
            l = links[lid]
            if nodes[l[1]]["type"] in UI_ONLY:
                continue
            inputs[i["name"]] = [str(l[1]), l[2]]
        api[str(n["id"])] = node(n["type"], inputs, n.get("title", n["type"]))
    return api


def ui_node(ui, nid):
    return next((n for n in ui["nodes"] if n["id"] == int(nid)), None)


def ui_values(ui, nid) -> dict:
    """UI 노드의 이름 붙은 위젯 값 (노드가 없으면 {})."""
    return (ui_node(ui, nid) or {}).get("widgets_values_named") or {}


def ui_set(ui, nid, key, value):
    """UI 노드의 위젯 값 바꾸기 (widgets_values 와 widgets_values_named 둘 다)."""
    n = ui_node(ui, nid)
    if n is None:
        raise ValueError(f"노드 {nid} 가 없어요")
    named = n.get("widgets_values_named") or {}
    keys = list(named)
    if key not in keys:
        raise ValueError(f"노드 {nid} ({n['type']}) 에 '{key}' 값이 없어요")
    n["widgets_values"][keys.index(key)] = value
    named[key] = value
