"""
test.json (ComfyUI UI 포맷) → API 포맷 변환 + 프롬프트/이미지/시드 주입
"""

import copy
import json
import random

UI_ONLY = {"MarkdownNote", "Note", "PreviewImage"}
SKIP_WIDGETS = {"upload", "control_after_generate"}


def node(class_type, inputs, title):
    """ComfyUI API 노드의 공통 형식."""
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


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
            src = nodes[l[1]]
            if src["type"] in UI_ONLY:
                continue
            inputs[i["name"]] = [str(l[1]), l[2]]
        api[str(n["id"])] = {"class_type": n["type"], "inputs": inputs,
                             "_meta": {"title": n.get("title", n["type"])}}
    return api


def build(template_ui: dict, ids: dict, *, control_image: str, prompts: dict,
          seed1=None, seed2=None, prefix="VR360/studio", settings: dict | None = None,
          keep_mask: str | None = None) -> tuple[dict, dict]:
    api = ui_to_api(copy.deepcopy(template_ui))

    def set_(key, field, value):
        nid = str(ids[key])
        if nid not in api:
            raise KeyError(f"워크플로에 {key} 노드(id {nid})가 없어요. config.json 의 node_ids 를 확인해 주세요.")
        api[nid]["inputs"][field] = value

    set_("load_image", "image", control_image)
    set_("positive", "text", prompts["positive"])
    set_("negative", "text", prompts["negative"])
    set_("seam_positive", "text", prompts["seam_positive"])
    set_("seam_negative", "text", prompts["seam_negative"])
    s1 = seed1 if seed1 is not None else random.randint(0, 2**31 - 1)
    s2 = seed2 if seed2 is not None else random.randint(0, 2**31 - 1)
    set_("sampler_main", "seed", int(s1))
    set_("sampler_seam", "seed", int(s2))
    # 생성 품질 설정 (웹 화면 '생성 설정')
    st = settings or {}
    if st.get("steps"):
        set_("sampler_main", "steps", int(st["steps"]))
        set_("sampler_seam", "steps", int(st["steps"]))
    if st.get("cfg"):
        set_("sampler_main", "cfg", float(st["cfg"]))
        set_("sampler_seam", "cfg", float(st["cfg"]))
    if st.get("seam_denoise"):
        set_("sampler_seam", "denoise", float(st["seam_denoise"]))
    if st.get("lora_strength") is not None and str(ids.get("lora")) in api:
        set_("lora", "strength_model", float(st["lora_strength"]))
    # 결과는 최종 360 한 장만: 2K 중간 저장 노드는 빼고 최종 저장만 남김
    if str(ids.get("save_2k")) in api and str(ids.get("save_2k")) != str(ids.get("save_final")):
        del api[str(ids["save_2k"])]
    set_("save_final", "filename_prefix", prefix + "_360")

    # 원본 사진 보존: 1차 생성에서 사진 영역은 다시 그리지 않음 (초록 부분만 생성)
    if keep_mask:
        lock_photo_region(api, ids, keep_mask)

    # 이음새 복원: 띠를 초록(#00ff00)으로 칠해 아웃페인트 LoRA 방식으로 '채우기'
    # (기존: 이음새가 그대로 있는 그림을 참조로 줘서, 편집 모델이 이음새까지 그대로 다시 그림)
    green_seam_band(api, ids, st)

    # 이음새 복원은 LoRA 를 약하게 (별도 LoRA 노드 복제)
    lora_id, seam_id = str(ids.get("lora")), str(ids.get("sampler_seam"))
    if st.get("seam_lora_strength") is not None and lora_id in api and seam_id in api:
        api[lora_id + "0"] = copy.deepcopy(api[lora_id])
        api[lora_id + "0"]["inputs"]["strength_model"] = float(st["seam_lora_strength"])
        api[lora_id + "0"]["_meta"] = {"title": "LoRA (이음새용, 약하게)"}
        api[seam_id]["inputs"]["model"] = [lora_id + "0", 0]

    # 이음새 2차 다듬기: 1차(초록 띠 채우기)가 띠 가운데서 좌우를 따로 그려 '세로 선'이 남는 경우가 있어
    # (특히 4스텝 distilled 모델) 1차 결과의 가운데 좁은 띠만 약하게 다시 그려 양쪽을 이어 줌
    seam_second_pass(api, ids, st, s2)

    # 디테일 리파인 (LoRA 없이) — 업스케일 블러/뭉개짐 대신 실제 질감
    rf = st.get("refine")
    if rf and rf.get("enabled", True):
        attach_refine_to_generate(api, ids, {**rf, "seed": s1}, prompts)
    return api, {"seed1": s1, "seed2": s2}


def add_photo_upscale(api: dict, photo_image: str, prefix: str, model: str = "4x_NMKD-Siax_200k.pth") -> str:
    """원본 사진을 '원래 해상도 그대로' AI 업스케일해 따로 저장 (생성 그래프는 사진을 2K 360 안에 줄여 넣으므로
    그 자리의 디테일이 버려짐 → 마지막에 이 업스케일 사진을 다시 붙여 원본 자리도 선명하게). 저장 노드 id 반환."""
    loader = next((k for k, v in api.items() if v.get("class_type") == "UpscaleModelLoader"), None)
    if loader is None:
        loader = "969"
        api[loader] = node("UpscaleModelLoader", {"model_name": model}, "업스케일 모델 (원본 사진)")
    api["970"] = node("LoadImage", {"image": photo_image}, "원본 사진 (원래 해상도)")
    api["971"] = node("ImageUpscaleWithModel", {"upscale_model": [loader, 0], "image": ["970", 0]}, "원본 사진 AI 업스케일 x4")
    api["972"] = node("SaveImage", {"images": ["971", 0], "filename_prefix": prefix + "_photo_up"}, "업스케일 원본 저장")
    return "972"


def build_photo_upscale(photo_image: str, model: str, prefix: str) -> tuple[dict, str]:
    api = {"969": node("UpscaleModelLoader", {"model_name": model}, "업스케일 모델")}
    return api, add_photo_upscale(api, photo_image, prefix, model)


def load_template(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_enhance(image: str, W: int, H: int, *, final_w=8192, upscale_model="4x_NMKD-Siax_200k.pth",
                  blend=0.2, noise=0.004, pad=128, prefix="VR360/enhance",
                  refine: dict | None = None, template_ui: dict | None = None, ids: dict | None = None) -> dict:
    """이미 360(2:1)인 이미지 → 이음새 패딩 → AI 업스케일 → 정확히 final_w × final_w/2 로 맞춤 → 블렌드 → 디더 → 저장.
    업스케일 모델 배율(x2/x4)과 관계없이 결과 크기가 정확하도록 업스케일 뒤 크기를 다시 맞춘다."""
    FW, FH = final_w, final_w // 2
    r = FW / W
    pad_out = round(pad * r)
    wf = {"1": node("LoadImage", {"image": image}, "360 원본")}
    if W >= FW:                                   # 이미 충분히 큼 → 8K 로 맞추기만
        wf["12"] = node("ImageScale", {"image": ["1", 0], "upscale_method": "lanczos", "width": FW, "height": FH,
                                    "crop": "disabled"}, "8K 로 맞춤")
        last = ["12", 0]
    else:
        wf.update({
            "2": node("EmptyImage", {"width": W + 2 * pad, "height": H, "batch_size": 1, "color": 0}, "패딩 캔버스"),
            "3": node("ImageCrop", {"image": ["1", 0], "width": pad, "height": H, "x": W - pad, "y": 0}, "오른쪽 끝"),
            "4": node("ImageCrop", {"image": ["1", 0], "width": pad, "height": H, "x": 0, "y": 0}, "왼쪽 끝"),
            "5": node("ImageCompositeMasked", {"destination": ["2", 0], "source": ["1", 0], "x": pad, "y": 0,
                                            "resize_source": False}, "가운데 원본"),
            "6": node("ImageCompositeMasked", {"destination": ["5", 0], "source": ["3", 0], "x": 0, "y": 0,
                                            "resize_source": False}, "왼쪽 패딩"),
            "7": node("ImageCompositeMasked", {"destination": ["6", 0], "source": ["4", 0], "x": W + pad, "y": 0,
                                            "resize_source": False}, "오른쪽 패딩"),
            "8": node("UpscaleModelLoader", {"model_name": upscale_model}, "업스케일 모델"),
            "9": node("ImageUpscaleWithModel", {"upscale_model": ["8", 0], "image": ["7", 0]}, "AI 업스케일"),
            "10": node("ImageScale", {"image": ["9", 0], "upscale_method": "lanczos", "width": FW + 2 * pad_out,
                                   "height": FH, "crop": "disabled"}, "배율 보정"),
            "11": node("ImageCrop", {"image": ["10", 0], "width": FW, "height": FH, "x": pad_out, "y": 0}, "패딩 제거"),
            "12": node("ImageScale", {"image": ["1", 0], "upscale_method": "lanczos", "width": FW, "height": FH,
                                   "crop": "disabled"}, "Lanczos 기준본"),
            "13": node("ImageBlend", {"image1": ["11", 0], "image2": ["12", 0], "blend_factor": float(blend),
                                   "blend_mode": "normal"}, "AI/Lanczos 블렌드"),
        })
        last = ["13", 0]
    if refine and refine.get("enabled", True) and template_ui and ids and W < FW:
        base = ui_to_api(copy.deepcopy(template_ui))
        for key in ("unet", "clip", "vae"):
            wf[str(ids[key])] = base[str(ids[key])]
        wf.pop("12", None)                       # Lanczos 블렌드(블러 원인) 대신 리파인
        wf.pop("13", None)
        last = add_refine(wf, ["11", 0], FW, FH, unet=str(ids["unet"]), clip=str(ids["clip"]), vae=str(ids["vae"]),
                          positive=refine["positive"], negative=refine.get("negative", ""),
                          seed=refine.get("seed", 0), steps=refine.get("steps", 20), cfg=refine.get("cfg", 4.0),
                          denoise=refine.get("denoise", 0.3), tile=refine.get("tile", 1536),
                          overlap=refine.get("overlap", 192))
    if noise:
        wf["14"] = node("ImageAddNoise", {"image": last, "seed": 0, "strength": float(noise)}, "디더")
        last = ["14", 0]
    wf["15"] = node("SaveImage", {"images": last, "filename_prefix": prefix + "_360"}, "8K 저장")
    return wf


def build_pole(template_ui: dict, *, image: str, mask: str, positive: str, negative: str,
               seed=None, steps=25, cfg=4.0, denoise=0.85, lora_strength=0.6, prefix="VR360/pole",
               loader_ids=("48", "63", "44", "43")) -> tuple[dict, int]:
    """돌린 ERP + 마스크 → 극점 부분만 다시 그리기 (이음새 복원과 같은 방식).
    모델/LoRA/CLIP/VAE 로더는 생성 워크플로(template)의 것을 그대로 가져온다."""
    base = ui_to_api(copy.deepcopy(template_ui))
    unet, lora, clip, vae = loader_ids
    wf = {k: base[k] for k in loader_ids}
    wf[lora]["inputs"]["strength_model"] = float(lora_strength)
    s = seed if seed is not None else random.randint(0, 2**31 - 1)
    wf.update({
        "201": node("LoadImage", {"image": image}, "돌린 ERP"),
        "202": node("LoadImageMask", {"image": mask, "channel": "red"}, "극점 마스크"),
        "203": node("CLIPTextEncode", {"clip": [clip, 0], "text": positive}, "극점 Positive"),
        "204": node("CLIPTextEncode", {"clip": [clip, 0], "text": negative}, "극점 Negative"),
        "205": node("VAEEncode", {"pixels": ["201", 0], "vae": [vae, 0]}, "인코드"),
        "206": node("ReferenceLatent", {"conditioning": ["203", 0], "latent": ["205", 0]}, "주변 참조"),
        "207": node("SetLatentNoiseMask", {"samples": ["205", 0], "mask": ["202", 0]}, "극점만 다시"),
        "208": node("KSampler", {"model": [lora, 0], "positive": ["206", 0], "negative": ["204", 0],
                              "latent_image": ["207", 0], "seed": int(s), "steps": int(steps), "cfg": float(cfg),
                              "sampler_name": "euler", "scheduler": "simple", "denoise": float(denoise)}, "극점 인페인트"),
        "209": node("VAEDecode", {"samples": ["208", 0], "vae": [vae, 0]}, "디코드"),
        "210": node("PreviewImage", {"images": ["209", 0]}, "중간 결과(임시, output 폴더에 저장 안 함)"),
    })
    return wf, s


# ----------------------------------------------------------------------------- 디테일 리파인 (LoRA 없이, 타일)
# 업스케일 결과를 1536px 타일로 나눠 생성 모델(LoRA 없음)로 약하게(denoise≈0.3) 다시 그려
# 업스케일러의 뭉개짐·블러 대신 실제 질감을 만든다. 타일은 겹쳐서 부드럽게 합성.

def _tiles(size, tile, overlap):
    if size <= tile:
        return [0]
    step = tile - overlap
    xs = list(range(0, size - tile + 1, step))
    if xs[-1] != size - tile:
        xs.append(size - tile)
    return xs


def add_refine(wf: dict, src: list, W: int, H: int, *, unet: str, clip: str, vae: str, positive: str,
               negative: str, seed=0, steps=20, cfg=4.0, denoise=0.3, tile=1536, overlap=192,
               id_base=400, wrap=True) -> list:
    """wf 에 타일 리파인 노드를 추가하고 최종 이미지 참조를 반환. unet/clip/vae 는 기존 로더 노드 id."""
    k = id_base
    def nid():
        nonlocal k
        k += 1
        return str(k)
    pos, neg = nid(), nid()
    wf[pos] = node("CLIPTextEncode", {"clip": [clip, 0], "text": positive}, "리파인 Positive")
    wf[neg] = node("CLIPTextEncode", {"clip": [clip, 0], "text": negative}, "리파인 Negative")
    # 좌우 이음새: 오른쪽 끝에 왼쪽 앞부분(E px)을 이어 붙여 넓힌 뒤 타일을 돌리면
    # 이음새를 가로지르는 타일이 생겨 양쪽이 같은 타일에서 그려짐 → 끝나고 그 부분을 왼쪽에 서서히 섞어 되돌림
    W0, E = W, (min(tile, W) if wrap else 0)
    if E:
        p0, p1, p2, p3 = nid(), nid(), nid(), nid()
        wf[p0] = node("EmptyImage", {"width": W + E, "height": H, "batch_size": 1, "color": 0}, "리파인 이음새 패딩")
        wf[p1] = node("ImageCompositeMasked", {"destination": [p0, 0], "source": src, "x": 0, "y": 0,
                                            "resize_source": False}, "원본")
        wf[p2] = node("ImageCrop", {"image": src, "width": E, "height": H, "x": 0, "y": 0}, "왼쪽 앞부분")
        wf[p3] = node("ImageCompositeMasked", {"destination": [p1, 0], "source": [p2, 0], "x": W, "y": 0,
                                            "resize_source": False}, "오른쪽 끝에 이어 붙임")
        src = [p3, 0]
        W = W + E
    canvas = src
    xs, ys = _tiles(W, tile, overlap), _tiles(H, tile, overlap)
    tw, th = min(tile, W), min(tile, H)
    for y in ys:
        for x in xs:
            crop, enc, smp, dec, msk, fea, comp = (nid() for _ in range(7))
            wf[crop] = node("ImageCrop", {"image": src, "width": tw, "height": th, "x": x, "y": y}, f"리파인 타일 {x},{y}")
            wf[enc] = node("VAEEncode", {"pixels": [crop, 0], "vae": [vae, 0]}, "타일 인코드")
            wf[smp] = node("KSampler", {"model": [unet, 0], "positive": [pos, 0], "negative": [neg, 0],
                                     "latent_image": [enc, 0], "seed": int(seed), "steps": int(steps),
                                     "cfg": float(cfg), "sampler_name": "euler", "scheduler": "simple",
                                     "denoise": float(denoise)}, "리파인 샘플러")
            wf[dec] = node("VAEDecode", {"samples": [smp, 0], "vae": [vae, 0]}, "타일 디코드")
            wf[msk] = node("SolidMask", {"value": 1.0, "width": tw, "height": th}, "타일 마스크")
            fl = overlap // 2 if x > 0 else 0
            fr = overlap // 2 if x + tw < W else 0
            ft = overlap // 2 if y > 0 else 0
            fb = overlap // 2 if y + th < H else 0
            wf[fea] = node("FeatherMask", {"mask": [msk, 0], "left": fl, "top": ft, "right": fr, "bottom": fb},
                        "타일 경계 페더")
            wf[comp] = node("ImageCompositeMasked", {"destination": canvas, "source": [dec, 0], "mask": [fea, 0],
                                                  "x": x, "y": y, "resize_source": False}, "타일 합성")
            canvas = [comp, 0]
    if E:
        q1, q2, q3, q4, q5 = nid(), nid(), nid(), nid(), nid()
        wf[q1] = node("ImageCrop", {"image": canvas, "width": W0, "height": H, "x": 0, "y": 0}, "원래 폭으로")
        wf[q2] = node("ImageCrop", {"image": canvas, "width": E, "height": H, "x": W0, "y": 0}, "이음새 너머 부분")
        wf[q3] = node("SolidMask", {"value": 1.0, "width": E, "height": H}, "이음새 섞기 마스크")
        wf[q4] = node("FeatherMask", {"mask": [q3, 0], "left": 0, "top": 0, "right": E, "bottom": 0},
                   "왼쪽 1 → 오른쪽 0")
        wf[q5] = node("ImageCompositeMasked", {"destination": [q1, 0], "source": [q2, 0], "mask": [q4, 0],
                                            "x": 0, "y": 0, "resize_source": False}, "이음새 이어지게 섞기")
        canvas = [q5, 0]
    return canvas


def attach_refine_to_generate(api: dict, ids: dict, rf: dict, prompts: dict, W=8192, H=4096) -> bool:
    """test_quest3 생성 그래프: 8K 크롭(117) → [리파인] → 디더(119). Lanczos 블렌드(118, 89)는 블러 원인이라 제거."""
    src_node, blend_node, lanczos_node, noise_node = "117", "118", "89", "119"
    if src_node not in api or noise_node not in api:
        return False
    last = add_refine(api, [src_node, 0], W, H, unet=str(ids["unet"]), clip=str(ids["clip"]), vae=str(ids["vae"]),
                      positive=rf["positive"], negative=prompts.get("negative", ""), seed=rf.get("seed", 0),
                      steps=rf.get("steps", 20), cfg=rf.get("cfg", 4.0), denoise=rf.get("denoise", 0.3),
                      tile=rf.get("tile", 1536), overlap=rf.get("overlap", 192))
    api[noise_node]["inputs"]["image"] = last
    api.pop(blend_node, None)
    api.pop(lanczos_node, None)
    return True



SEAM_TRIGGER = ("Fill the green spaces according to the image. This is a 360 equirectangular panorama: "
                "the left and right sides of the green band are the same continuous place.")


def green_seam_band(api: dict, ids: dict, st: dict) -> bool:
    """test.json 이음새 패스: 가운데로 옮긴 이미지(99)의 띠(마스크 101/102 위치)를 초록으로 칠해 인코드(79)에 넣음."""
    shifted, enc, band, place, seam_pos = "99", "79", "101", "102", str(ids.get("seam_positive"))
    if not all(k in api for k in (shifted, enc, band, place)):
        return False
    bw = int(api[band]["inputs"].get("width", 512))
    bh = int(api[band]["inputs"].get("height", 1024))
    bx = int(api[place]["inputs"].get("x", 768))
    api["901"] = {"class_type": "EmptyImage", "inputs": {"width": bw, "height": bh, "batch_size": 1, "color": 0x00FF00},
                  "_meta": {"title": "이음새 띠 초록"}}
    api["902"] = {"class_type": "ImageCompositeMasked",
                  "inputs": {"destination": [shifted, 0], "source": ["901", 0], "x": bx, "y": 0, "resize_source": False},
                  "_meta": {"title": "이음새 띠를 초록으로"}}
    api[enc]["inputs"]["pixels"] = ["902", 0]
    if seam_pos in api:
        t = api[seam_pos]["inputs"].get("text", "")
        if not t.startswith("Fill the green"):
            api[seam_pos]["inputs"]["text"] = SEAM_TRIGGER + " " + t
    if str(ids.get("sampler_seam")) in api:
        api[str(ids["sampler_seam"])]["inputs"]["denoise"] = 1.0        # 초록을 완전히 채워야 함
    return True



def seam_second_pass(api: dict, ids: dict, st: dict, seed: int, band=192, feather=64, denoise=0.6) -> bool:
    """test.json 이음새 패스 뒤(85 = 이음새 복원 합성, 아직 가운데로 옮긴 상태)에 좁은 띠를 약하게 한 번 더.
    85 → 인코드 → 가운데 band 만 노이즈 → 같은 이음새 샘플러 설정(LoRA·스텝·cfg)·denoise 0.6 → 부드러운 마스크로 합성
    → 원래대로 되돌리는 노드(105·106·107)가 이 결과를 쓰게 바꿈."""
    comp, seam, vae_enc = "85", str(ids.get("sampler_seam")), "79"
    if not all(k in api for k in (comp, seam, vae_enc, "105", "106", "107", "100")):
        return False
    W = int(api["100"]["inputs"].get("width", 2048))
    H = int(api["100"]["inputs"].get("height", 1024))
    x = W // 2 - band // 2
    vae = api[vae_enc]["inputs"]["vae"]
    sm = api[seam]["inputs"]
    pos_cond = api.get("80", {}).get("inputs", {}).get("conditioning", sm["positive"])
    N = lambda t, i, title: {"class_type": t, "inputs": i, "_meta": {"title": title}}
    api["911"] = N("VAEEncode", {"pixels": [comp, 0], "vae": vae}, "이음새 2차 · 인코드")
    api["912"] = N("SolidMask", {"value": 1.0, "width": band, "height": H}, "이음새 2차 · 좁은 띠")
    api["913"] = N("MaskComposite", {"destination": ["100", 0], "source": ["912", 0], "x": x, "y": 0, "operation": "add"},
                   "이음새 2차 · 띠 위치(가운데)")
    api["914"] = N("SetLatentNoiseMask", {"samples": ["911", 0], "mask": ["913", 0]}, "이음새 2차 · 띠만 다시")
    api["915"] = N("ReferenceLatent", {"conditioning": pos_cond, "latent": ["911", 0]}, "이음새 2차 · 참조")
    api["916"] = N("KSampler", {"model": sm["model"], "positive": ["915", 0], "negative": sm["negative"],
                                "latent_image": ["914", 0], "seed": int(seed) + 1, "steps": sm.get("steps", 20),
                                "cfg": sm.get("cfg", 4.0), "sampler_name": sm.get("sampler_name", "euler"),
                                "scheduler": sm.get("scheduler", "simple"), "denoise": float(denoise)},
                   "이음새 2차 · 약하게 다시 그림")
    api["917"] = N("VAEDecode", {"samples": ["916", 0], "vae": vae}, "이음새 2차 · 디코드")
    api["918"] = N("FeatherMask", {"mask": ["912", 0], "left": feather, "top": 0, "right": feather, "bottom": 0},
                   "이음새 2차 · 띠 가장자리 부드럽게")
    api["919"] = N("MaskComposite", {"destination": ["100", 0], "source": ["918", 0], "x": x, "y": 0, "operation": "add"},
                   "이음새 2차 · 부드러운 띠")
    api["920"] = N("ImageCompositeMasked", {"destination": [comp, 0], "source": ["917", 0], "mask": ["919", 0],
                                            "x": 0, "y": 0, "resize_source": False}, "이음새 2차 · 합성")
    for k in ("105", "106"):
        api[k]["inputs"]["image"] = ["920", 0]
    api["107"]["inputs"]["destination"] = ["920", 0]
    return True


def lock_photo_region(api: dict, ids: dict, mask_image: str) -> bool:
    """1차 샘플러(31)의 latent 에 노이즈 마스크를 걸어 사진 부분 latent 를 그대로 유지."""
    smp = str(ids.get("sampler_main"))
    if smp not in api:
        return False
    lat = api[smp]["inputs"].get("latent_image")
    if not isinstance(lat, list):
        return False
    api["950"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_image, "channel": "red"},
                  "_meta": {"title": "생성할 곳 마스크 (사진 영역 유지)"}}
    api["951"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": lat, "mask": ["950", 0]},
                  "_meta": {"title": "사진 영역은 다시 그리지 않음"}}
    api[smp]["inputs"]["latent_image"] = ["951", 0]
    return True


# 360 전체 영상 프롬프트 (LTX-2.3 + 360 LoRA 에 붙여 넣는 용 — 생성은 ComfyUI 에서)
LTX360_POS = ("Equirectangular 360 degree panorama, immersive photorealistic VR video. Locked camera and fixed horizon. "
              "The video starts calm and exactly like the input panorama. {event} "
              "Then the water calms down and the scene settles, ending exactly like the input panorama. "
              "{keep} remain solid and perfectly still. Continuous, natural motion.")
LTX360_EVENT = ("A massive flood of water surges from the waterfall through the scene: waves crash and foam over the rocks, "
                "spray and mist rise toward the viewer.")
LTX360_NEG = ("camera movement, camera pan, zoom, dolly, rotation, shaking, moving bridge, moving rocks, morphing stonework, "
              "warped architecture, new objects, people, animals in view, flickering, boiling textures, visible seam, "
              "non-equirectangular perspective, scene cut, text, subtitles, watermark, music, speech, voices")


# ----------------------------------------------------------------------------- 360 영상: 사용자 노드 파일에 값만 넣어 실행
# 사용자가 ComfyUI 에서 쓰는 노드 파일(video_nodes/*.json)을 그대로 쓰고, 웹 값만 넣음. 노드 id 로 찾음:
#   1131 원본 360 · 1128/1112 Positive/Negative · 1102 길이(초) · 1114 fps · 1132 360 LoRA · 1100 시드 · SaveVideo 전부
#   마스크: 1162 LoadImage(칠한 마스크 그림) 가 있으면 그 그림을 넣고,
#          없으면 사각 마스크(1134 SolidMask 크기 + 1136 MaskComposite 위치, 1133 캔버스 기준)에 칠한 범위의 좌표를 넣음.
#   계산 노드(ComfyMathExpression)는 숫자로 바꿔 넣음.
VIDEO_IDS = {"image": 1131, "positive": 1128, "negative": 1112, "seconds": 1102, "fps": 1114, "lora": 1132, "seed": 1100,
             "mask_image": 1162, "rect": 1134, "rect_pos": 1136, "rect_canvas": 1133, "rect_feather": 1135}


def _ui_node(ui, nid):
    return next((n for n in ui["nodes"] if n["id"] == int(nid)), None)


def ui_set(ui, nid, key, value):
    """UI 노드의 위젯 값 바꾸기 (widgets_values 와 widgets_values_named 둘 다)."""
    n = _ui_node(ui, nid)
    if n is None:
        raise ValueError(f"노드 {nid} 가 없어요")
    named = n.get("widgets_values_named") or {}
    keys = list(named)
    if key not in keys:
        raise ValueError(f"노드 {nid} ({n['type']}) 에 '{key}' 값이 없어요")
    n["widgets_values"][keys.index(key)] = value
    named[key] = value


def video_mask_mode(ui) -> str:
    if (n := _ui_node(ui, VIDEO_IDS["mask_image"])) is not None and n["type"] == "LoadImage":
        return "image"
    if _ui_node(ui, VIDEO_IDS["rect"]) is not None and _ui_node(ui, VIDEO_IDS["rect_pos"]) is not None:
        return "rect"
    return "none"


def video_rect_canvas(ui) -> tuple[int, int]:
    c = _ui_node(ui, VIDEO_IDS["rect_canvas"])
    nm = (c or {}).get("widgets_values_named") or {}
    return int(nm.get("width", 2048)), int(nm.get("height", 1024))


def fill_video_node(template_ui: dict, *, image: str, positive: str, negative: str, seconds=4, lora_strength=None,
                    seed=None, prefix="VR360/flood", mask_image: str | None = None, rect: tuple | None = None,
                    crop: dict | None = None):
    """노드 파일(UI)에 값 넣기 → (채운 UI, 시드, 정보). rect = (x, y, w, h) — 1133 캔버스 좌표.
    crop = crop_plan() 결과 — 크롭 생성 노드 파일(video_has_crop)일 때 필요."""
    ui = copy.deepcopy(template_ui)
    ids = VIDEO_IDS
    seed = int(seed if seed is not None else random.randint(0, 2**31 - 1))
    ui_set(ui, ids["image"], "image", image)
    ui_set(ui, ids["positive"], "text", positive)
    ui_set(ui, ids["negative"], "text", negative)
    secs = max(1, int(round(float(seconds))))
    ui_set(ui, ids["seconds"], "value", secs)
    if _ui_node(ui, ids["seed"]) is not None:
        ui_set(ui, ids["seed"], "noise_seed", seed)
    if lora_strength is not None and _ui_node(ui, ids["lora"]) is not None:
        ui_set(ui, ids["lora"], "strength_model", float(lora_strength))
    mode = video_mask_mode(ui)
    if mode == "image":
        if not mask_image:
            raise ValueError("마스크 그림이 없어요")
        ui_set(ui, ids["mask_image"], "image", mask_image)
    elif mode == "rect":
        if not rect:
            raise ValueError("마스크 범위가 없어요")
        x, y, w, h = (int(v) for v in rect)
        ui_set(ui, ids["rect"], "width", w)
        ui_set(ui, ids["rect"], "height", h)
        ui_set(ui, ids["rect_pos"], "x", x)
        ui_set(ui, ids["rect_pos"], "y", y)
        if (f := _ui_node(ui, ids["rect_feather"])) is not None:   # 경계 흐림이 사각형보다 크면 줄임
            lim = max(0, min(w, h) // 2 - 1)
            for k in ("left", "top", "right", "bottom"):
                if int(f["widgets_values_named"].get(k, 0)) > lim:
                    ui_set(ui, ids["rect_feather"], k, lim)
    else:
        raise ValueError("노드 파일에서 마스크 노드(1162 LoadImage 또는 1134 SolidMask + 1136 MaskComposite)를 못 찾았어요")
    if video_has_crop(ui):
        if not crop:
            raise ValueError("크롭 생성 노드 파일에는 잘라낼 범위(crop)가 필요해요")
        fill_crop(ui, crop)
    k = 0
    for n in ui["nodes"]:
        if n["type"] == "SaveVideo":
            k += 1
            ui_set(ui, n["id"], "filename_prefix", f"{prefix}_{k}")
    fps_n = _ui_node(ui, ids["fps"])
    fps = int((fps_n or {}).get("widgets_values_named", {}).get("value", 24))
    return ui, seed, {"mode": mode, "seconds": secs, "fps": fps, "frames": secs * fps + 1, "rect": rect, "crop": crop}


def video_node_api(filled_ui: dict) -> tuple[dict, list]:
    """채운 UI → API. 계산 노드는 숫자로. 반환 (그래프, 저장 노드 id 목록)."""
    api = ui_to_api(filled_ui)
    ids = VIDEO_IDS
    vals = {str(ids["seconds"]): int(api[str(ids["seconds"])]["inputs"]["value"]),
            str(ids["fps"]): int(api[str(ids["fps"])]["inputs"]["value"])} if str(ids["seconds"]) in api else {}
    links = {l[0]: l for l in filled_ui["links"]}

    def src_val(nid, slot):                      # 링크 출처 값 (PrimitiveInt · 계산 노드)
        nid = str(nid)
        if nid in vals:
            return vals[nid]
        n = api.get(nid)
        if n and n["class_type"] == "ComfyMathExpression":
            env = {}
            for k, v in n["inputs"].items():
                if k.startswith("values."):
                    env[k.split(".", 1)[1]] = src_val(v[0], v[1]) if isinstance(v, list) else v
            r = eval(n["inputs"]["expression"], {"__builtins__": {}}, env)   # 노드 파일의 간단한 산술식
            return float(r) if slot == 0 else int(r) if slot == 1 else bool(r)
        return None

    calc = [k for k, n in api.items() if n["class_type"] in ("ComfyMathExpression", "PrimitiveInt", "PrimitiveFloat")]
    for n in api.values():
        for k, v in list(n["inputs"].items()):
            if isinstance(v, list) and len(v) == 2 and v[0] in calc:
                r = src_val(v[0], v[1])
                if r is None:
                    raise ValueError(f"계산 노드 {v[0]} 값을 못 구했어요")
                n["inputs"][k] = r
    for k in calc:
        api.pop(k, None)
    saves = []
    for k, n in api.items():
        if n["class_type"] == "SaveVideo":
            src = api.get(str((n["inputs"].get("video") or [None])[0]), {})
            saves.append({"node": k, "audio": "audio" in src.get("inputs", {})})
    saves.sort(key=lambda s: (not s["audio"], s["node"]))          # 소리 있는 것 먼저
    return api, saves


# ----------------------------------------------------------------------------- 크롭 생성 노드 파일 (event_cropgen_*.json)
# 칠한 범위만 원본 360(8K)에서 잘라 일반 영상으로 고해상도 생성(2단계) → 칠한 곳 밖은 사진 그대로 → 4K 360 원래 자리에 붙임.
# 장면마다 다른 값(자를 범위·생성 크기·붙일 자리)은 칠한 마스크로 계산해서 아래 노드에 넣음 (장면 값은 노드 파일에 두지 않음).
#   1202 ImageCrop(원본) · 1203/1204 생성 크기(2단계/1단계) · 1208 CropMask(마스크 2048 좌표)
#   1210/1212 마스크 크기(1단계/2단계) · 1219 4K 크기 · 1221 4K 붙일 자리 · 1218 4K 360 (4096x2048 고정)
VIDEO_CROP_IDS = {"src_crop": 1202, "gen_hi": 1203, "gen_lo": 1204, "mask_crop": 1208, "mask_lo": 1210,
                  "mask_hi": 1212, "back_scale": 1219, "paste": 1221, "full": 1218}
CROP_MAX_PIXELS = 2560 * 1088          # 2단계 생성 크기 상한 (GPU 부담 기준)
OUT_W, OUT_H = 4096, 2048              # 최종 360 영상

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


def event_timeline(seconds) -> str:
    """영상 길이에 맞춘 사건 시간표 (장면과 무관한 공통 문장). 길이를 늘리지 않고 사건의 크기·속도를 길이에 맞춰
    시작 → 절정 → 끝(첫 화면과 같음)이 반드시 다 나오게 한다. 노드의 마지막 프레임 = 사진(1226)과 함께 씀."""
    s = float(seconds)
    a, b, c = round(s * 0.15, 1), round(s * 0.5, 1), round(s * 0.85, 1)
    return (f"Timing for this {s:g}-second clip: 0-{a:g}s only the scene's ordinary ongoing motion "
            "(any stream or waterfall already in the photo keeps flowing naturally, leaves move slightly); "
            f"{a:g}-{b:g}s the event appears and builds to its peak; {b:g}-{c:g}s it passes and subsides; "
            f"{c:g}-{s:g}s everything is back to that ordinary motion and looks like the first frame. "
            "The whole event happens and completely finishes inside the clip, sized to fit this short time; "
            "nothing is cut off mid-action.")


AMBIENT_EVENT = ("Only the scene's ordinary everyday motion: any stream, waterfall or water already in the photo keeps "
                 "flowing naturally and continuously at its normal rate with moving ripples, glints and foam; leaves move "
                 "slightly. Nothing new appears, no surge, no event.")


def ambient_timeline(seconds) -> str:
    """평소 움직임(반복용) 시간표: 처음부터 끝까지 같은 세기, 끝 = 처음 (뷰어에서 끊김 없이 반복)."""
    s = float(seconds)
    return (f"For the whole {s:g}-second clip the motion stays steady at the same strength from the first frame to the "
            "last, nothing builds up or calms down, and the last frame looks like the first frame so the clip loops "
            "seamlessly.")


def video_has_crop(ui) -> bool:
    n = _ui_node(ui, VIDEO_CROP_IDS["src_crop"])
    return n is not None and n["type"] == "ImageCrop" and n.get("mode", 0) not in (2, 4)


def crop_plan(mask, src_size, *, margin=0.12, max_pixels=CROP_MAX_PIXELS, min_aspect=0.6, max_aspect=3.0):
    """칠한 마스크(PIL L, 보통 2048x1024) → 자를 범위·생성 크기·4K 붙일 자리.
    마스크 범위에 여유를 붙이고, 생성 크기(64 배수, 상한 max_pixels)와 같은 비율로 맞춤."""
    import numpy as np
    MW, MH = mask.size
    m = np.asarray(mask.convert("L")) > 127
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    if x0 <= 2 and x1 >= MW - 2:
        raise ValueError("칠한 범위가 360 좌우 끝에 걸쳐 있어요 — 한쪽으로 모아서 칠해 주세요")
    pad = int(round(margin * max(x1 - x0, y1 - y0)))
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(MW, x1 + pad), min(MH, y1 + pad)
    w, h = x1 - x0, y1 - y0
    aspect = min(max_aspect, max(min_aspect, w / h))
    gw = max(256, int((max_pixels * aspect) ** 0.5) // 64 * 64)
    gh = max(256, int(round(gw / aspect / 64)) * 64)
    while gw * gh > max_pixels:
        gw -= 64
    aspect = gw / gh                                   # 생성 크기 비율에 정확히 맞춤 (짧은 쪽을 넓힘)
    if w / h < aspect:
        w = min(MW, int(round(h * aspect)))
        h = min(MH, int(round(w / aspect)))
    else:
        h = min(MH, int(round(w / aspect)))
        w = min(MW, int(round(h * aspect)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x = int(round(min(max(0, cx - w / 2), MW - w)))
    y = int(round(min(max(0, cy - h / 2), MH - h)))
    sx, sy = src_size[0] / MW, src_size[1] / MH
    ox, oy = OUT_W / MW, OUT_H / MH
    return {"mask": [x, y, w, h],
            "src": [int(round(x * sx)), int(round(y * sy)), int(round(w * sx)), int(round(h * sy))],
            "gen": [gw, gh], "gen_lo": [gw // 2, gh // 2],
            "out": [int(round(x * ox)), int(round(y * oy)), int(round(w * ox)), int(round(h * oy))]}


def fill_crop(ui, plan):
    ids = VIDEO_CROP_IDS
    x, y, w, h = plan["src"]
    for k, v in (("x", x), ("y", y), ("width", w), ("height", h)):
        ui_set(ui, ids["src_crop"], k, v)
    x, y, w, h = plan["mask"]
    for k, v in (("x", x), ("y", y), ("width", w), ("height", h)):
        ui_set(ui, ids["mask_crop"], k, v)
    for nid, (w, h) in ((ids["gen_hi"], plan["gen"]), (ids["mask_hi"], plan["gen"]),
                        (ids["gen_lo"], plan["gen_lo"]), (ids["mask_lo"], plan["gen_lo"]),
                        (ids["back_scale"], plan["out"][2:])):
        ui_set(ui, nid, "width", int(w))
        ui_set(ui, nid, "height", int(h))
    ui_set(ui, ids["paste"], "x", int(plan["out"][0]))
    ui_set(ui, ids["paste"], "y", int(plan["out"][1]))
