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

    # 디테일 리파인 (LoRA 없이) — 업스케일 블러/뭉개짐 대신 실제 질감
    rf = st.get("refine")
    if rf and rf.get("enabled", True):
        attach_refine_to_generate(api, ids, {**rf, "seed": s1}, prompts)
    return api, {"seed1": s1, "seed2": s2}


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


# ----------------------------------------------------------------------------- 영상화 (LTX-2.3 + Cinemagraph LoRA)
# ComfyUI 기본 블루프린트 "Image to Video (LTX-2.3)" 을 API 포맷으로 풀어 쓴 것 (서브그래프·동적 노드 없음 → 서버/Comfy Cloud 공통).
# 1단계: 절반 크기(=LoRA 학습 704×512)에서 8스텝 → 2배 잠재 업스케일 → 2단계 3스텝. 오디오 잠재는 모델 구조상 같이 돌지만 저장하지 않음.
LTX_SIGMAS_1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
LTX_SIGMAS_2 = "0.85, 0.7250, 0.4219, 0.0"


def ltx_length(duration_s: float, fps: int) -> int:
    """프레임 수는 8n+1 이어야 함 (24fps·정수 초면 그대로 맞음)."""
    n = max(1, round((duration_s * fps) / 8))
    return 8 * n + 1


def build_cinemagraph(image: str, *, positive: str, negative: str, width: int, height: int,
                      duration=1, fps=24, seed=None, prefix="VR360/motion", m: dict | None = None,
                      lora_strength=1.2, range_mask: str | None = None) -> tuple[dict, int]:
    """영역 화면 한 장 → 프레임(PNG) 들. 반환 (그래프, seed).

    range_mask(업로드한 흑백 PNG, 흰색 = 움직여도 되는 범위)가 있으면 '범위 고정 생성':
    모든 프레임을 원본 화면으로 채운 잠재에서 시작해 범위 밖은 매 단계 원본으로 되돌림(노이즈 마스크).
    → 범위 밖·첫 프레임은 원본 그대로, 카메라 흔들림 없음, 범위 경계는 모델이 주변과 어울리게 만듦."""
    m = m or {}
    ck = m.get("ckpt", "ltx-2.3-22b-dev-fp8.safetensors")
    seed = int(seed if seed is not None else random.randint(0, 2**48))
    L = ltx_length(duration, fps)
    w2, h2 = (width // 2) // 32 * 32, (height // 2) // 32 * 32       # 1단계 크기 (32 배수)
    wf = {
        "1": node("CheckpointLoaderSimple", {"ckpt_name": ck}, "LTX-2.3"),
        "2": node("LTXAVTextEncoderLoader", {"text_encoder": m.get("text_encoder", "gemma_3_12B_it_fp4_mixed.safetensors"),
                                             "ckpt_name": ck, "device": "default"}, "Gemma 텍스트 인코더"),
        "3": node("LTXVAudioVAELoader", {"ckpt_name": ck}, "오디오 VAE (모델 구조상 필요)"),
        "4": node("LoraLoaderModelOnly", {"model": ["1", 0], "lora_name": m.get("distilled_lora", "ltx-2.3-22b-distilled-lora-384.safetensors"),
                                          "strength_model": float(m.get("distilled_strength", 0.5))}, "증류 LoRA"),
        "6": node("CLIPTextEncode", {"clip": ["2", 0], "text": positive}, "Positive (일부만 움직임)"),
        "7": node("CLIPTextEncode", {"clip": ["2", 0], "text": negative}, "Negative"),
        "8": node("LTXVConditioning", {"positive": ["6", 0], "negative": ["7", 0], "frame_rate": float(fps)}, "프레임레이트"),
        "9": node("LoadImage", {"image": image}, "영역 화면"),
        "10": node("LTXVPreprocess", {"image": ["9", 0], "img_compression": int(m.get("img_compression", 18))}, "첫 프레임 전처리"),
        "11": node("EmptyLTXVLatentVideo", {"width": w2, "height": h2, "length": L, "batch_size": 1}, "1단계 빈 영상"),
        "12": node("LTXVImgToVideoInplace", {"vae": ["1", 2], "image": ["10", 0], "latent": ["11", 0],
                                             "strength": 0.7, "bypass": False}, "첫 프레임 넣기 (1단계)"),
        "13": node("LTXVEmptyLatentAudio", {"audio_vae": ["3", 0], "frames_number": L, "frame_rate": int(fps), "batch_size": 1}, "빈 오디오"),
        "14": node("LTXVConcatAVLatent", {"video_latent": ["12", 0], "audio_latent": ["13", 0]}, "영상+오디오"),
        "15": node("RandomNoise", {"noise_seed": seed}, "노이즈"),
        "16": node("KSamplerSelect", {"sampler_name": "euler_ancestral_cfg_pp"}, "샘플러 1"),
        "17": node("ManualSigmas", {"sigmas": LTX_SIGMAS_1}, "시그마 1"),
        "18": node("CFGGuider", {"model": ["5", 0], "positive": ["8", 0], "negative": ["8", 1], "cfg": 1.0}, "가이더 1"),
        "19": node("SamplerCustomAdvanced", {"noise": ["15", 0], "guider": ["18", 0], "sampler": ["16", 0],
                                             "sigmas": ["17", 0], "latent_image": ["14", 0]}, "1단계 생성"),
        "20": node("LTXVSeparateAVLatent", {"av_latent": ["19", 0]}, "분리 1"),
        "21": node("LTXVCropGuides", {"positive": ["8", 0], "negative": ["8", 1], "latent": ["20", 0]}, "가이드 정리"),
        "22": node("LatentUpscaleModelLoader", {"model_name": m.get("upscaler", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")}, "잠재 업스케일러"),
        "23": node("LTXVLatentUpsampler", {"samples": ["20", 0], "upscale_model": ["22", 0], "vae": ["1", 2]}, "2배 업스케일"),
        "24": node("LTXVImgToVideoInplace", {"vae": ["1", 2], "image": ["10", 0], "latent": ["23", 0],
                                             "strength": 1.0, "bypass": False}, "첫 프레임 넣기 (2단계)"),
        "25": node("LTXVConcatAVLatent", {"video_latent": ["24", 0], "audio_latent": ["20", 1]}, "영상+오디오 2"),
        "26": node("RandomNoise", {"noise_seed": 42}, "노이즈 2"),
        "27": node("KSamplerSelect", {"sampler_name": "euler_cfg_pp"}, "샘플러 2"),
        "28": node("ManualSigmas", {"sigmas": LTX_SIGMAS_2}, "시그마 2"),
        "29": node("CFGGuider", {"model": ["5", 0], "positive": ["21", 0], "negative": ["21", 1], "cfg": 1.0}, "가이더 2"),
        "30": node("SamplerCustomAdvanced", {"noise": ["26", 0], "guider": ["29", 0], "sampler": ["27", 0],
                                             "sigmas": ["28", 0], "latent_image": ["25", 0]}, "2단계 생성"),
        "31": node("LTXVSeparateAVLatent", {"av_latent": ["30", 0]}, "분리 2"),
        "32": node("VAEDecodeTiled", {"samples": ["31", 0], "vae": ["1", 2], "tile_size": 768, "overlap": 64,
                                      "temporal_size": 4096, "temporal_overlap": 4}, "디코드"),
        "33": node("SaveImage", {"images": ["32", 0], "filename_prefix": prefix + "_frames"}, "프레임 저장"),
    }
    if range_mask:
        Lt = (L - 1) // 8 + 1                                         # 잠재 프레임 수
        wf.update({
            "34": node("RepeatImageBatch", {"image": ["10", 0], "amount": L}, "원본을 모든 프레임에"),
            "35": node("ImageScale", {"image": ["34", 0], "upscale_method": "lanczos", "width": w2, "height": h2,
                                      "crop": "disabled"}, "1단계 크기"),
            "36": node("VAEEncode", {"pixels": ["35", 0], "vae": ["1", 2]}, "원본 잠재 (멈춘 영상)"),
            "37": node("LoadImage", {"image": range_mask}, "움직임 범위 (흰색만 움직임)"),
            "38": node("RepeatImageBatch", {"image": ["37", 0], "amount": max(1, Lt - 1)}, "범위 × 프레임"),
            "39": node("EmptyImage", {"width": int(width), "height": int(height), "batch_size": 1, "color": 0},
                       "첫 프레임 = 전부 고정"),
            "40": node("ImageBatch", {"image1": ["39", 0], "image2": ["38", 0]}, "프레임별 범위"),
            "41": node("ImageToMask", {"image": ["40", 0], "channel": "red"}, "범위 마스크"),
            "42": node("SetLatentNoiseMask", {"samples": ["36", 0], "mask": ["41", 0]}, "범위 밖은 원본 고정 (1단계)"),
            "43": node("SetLatentNoiseMask", {"samples": ["23", 0], "mask": ["41", 0]}, "범위 밖은 고정 (2단계)"),
        })
        wf["14"]["inputs"]["video_latent"] = ["42", 0]
        wf["25"]["inputs"]["video_latent"] = ["43", 0]
        for k in ("11", "12", "24"):
            del wf[k]
    if m.get("cinemagraph_lora") and float(lora_strength) > 0:
        wf["5"] = node("LoraLoaderModelOnly", {"model": ["4", 0], "lora_name": m["cinemagraph_lora"],
                                               "strength_model": float(lora_strength)}, "Cinemagraph LoRA (일부만 움직임)")
    else:                                         # LoRA 없이 (프롬프트 + 합성 마스크로만 고정)
        for k in ("18", "29"):
            wf[k]["inputs"]["model"] = ["4", 0]
    return wf, seed


def build_sam_mask(image: str, text: str, *, ckpt="sam3.1_multiplex_fp16.safetensors", threshold=0.4,
                   prefix="VR360/motion") -> dict:
    """영역 화면에서 움직일 요소(text, 예: 'pond water surface') 모양 마스크 (SAM 3)."""
    return {
        "1": node("CheckpointLoaderSimple", {"ckpt_name": ckpt}, "SAM 3"),
        "2": node("CLIPTextEncode", {"clip": ["1", 1], "text": text}, "찾을 것"),
        "3": node("LoadImage", {"image": image}, "영역 화면"),
        "4": node("SAM3_Detect", {"model": ["1", 0], "image": ["3", 0], "conditioning": ["2", 0], "threshold": float(threshold),
                                  "refine_iterations": 2, "individual_masks": False}, "마스크 찾기"),
        "5": node("MaskToImage", {"mask": ["4", 0]}, "마스크 → 이미지"),
        "6": node("SaveImage", {"images": ["5", 0], "filename_prefix": prefix + "_mask"}, "마스크 저장"),
    }
