"""
이미 있는 360 이미지를 다루는 그래프: 8K 화질 개선 · 극점(발밑·하늘) 다시 그리기 · 타일 디테일 리파인
"""

import copy

from .core import node, ksampler, random_seed, ui_to_api


def _loaders(template_ui: dict, keys) -> dict:
    """생성 워크플로(template)의 모델 로더 노드를 그대로 가져옴."""
    base = ui_to_api(copy.deepcopy(template_ui))
    return {k: base[k] for k in keys}


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
        wf.update(_loaders(template_ui, [str(ids[k]) for k in ("unet", "clip", "vae")]))
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
    unet, lora, clip, vae = loader_ids
    wf = _loaders(template_ui, loader_ids)
    wf[lora]["inputs"]["strength_model"] = float(lora_strength)
    s = random_seed(seed)
    wf.update({
        "201": node("LoadImage", {"image": image}, "돌린 ERP"),
        "202": node("LoadImageMask", {"image": mask, "channel": "red"}, "극점 마스크"),
        "203": node("CLIPTextEncode", {"clip": [clip, 0], "text": positive}, "극점 Positive"),
        "204": node("CLIPTextEncode", {"clip": [clip, 0], "text": negative}, "극점 Negative"),
        "205": node("VAEEncode", {"pixels": ["201", 0], "vae": [vae, 0]}, "인코드"),
        "206": node("ReferenceLatent", {"conditioning": ["203", 0], "latent": ["205", 0]}, "주변 참조"),
        "207": node("SetLatentNoiseMask", {"samples": ["205", 0], "mask": ["202", 0]}, "극점만 다시"),
        "208": ksampler([lora, 0], ["206", 0], ["204", 0], ["207", 0], seed=s, steps=steps, cfg=cfg,
                        denoise=denoise, title="극점 인페인트"),
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
            wf[smp] = ksampler([unet, 0], [pos, 0], [neg, 0], [enc, 0], seed=seed, steps=steps, cfg=cfg,
                               denoise=denoise, title="리파인 샘플러")
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
