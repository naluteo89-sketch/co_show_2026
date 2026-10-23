"""
360 이미지 생성 그래프 (comfy_nodes/image/test_quest3.json 에 값 주입 + 이음새·사진 보존·리파인 노드 추가)
"""

import copy

from .core import node, ksampler, random_seed, ui_to_api
from .enhance import add_refine

SEAM_TRIGGER = ("Fill the green spaces according to the image. This is a 360 equirectangular panorama: "
                "the left and right sides of the green band are the same continuous place.")


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
    s1, s2 = random_seed(seed1), random_seed(seed2)
    set_("sampler_main", "seed", s1)
    set_("sampler_seam", "seed", s2)
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
    green_seam_band(api, ids)

    # 이음새 복원은 LoRA 를 약하게 (별도 LoRA 노드 복제)
    lora_id, seam_id = str(ids.get("lora")), str(ids.get("sampler_seam"))
    if st.get("seam_lora_strength") is not None and lora_id in api and seam_id in api:
        api[lora_id + "0"] = copy.deepcopy(api[lora_id])
        api[lora_id + "0"]["inputs"]["strength_model"] = float(st["seam_lora_strength"])
        api[lora_id + "0"]["_meta"] = {"title": "LoRA (이음새용, 약하게)"}
        api[seam_id]["inputs"]["model"] = [lora_id + "0", 0]

    # 이음새 2차 다듬기: 1차(초록 띠 채우기)가 띠 가운데서 좌우를 따로 그려 '세로 선'이 남는 경우가 있어
    # (특히 4스텝 distilled 모델) 1차 결과의 가운데 좁은 띠만 약하게 다시 그려 양쪽을 이어 줌
    seam_second_pass(api, ids, s2)

    # 디테일 리파인 (LoRA 없이) — 업스케일 블러/뭉개짐 대신 실제 질감
    rf = st.get("refine")
    if rf and rf.get("enabled", True):
        attach_refine_to_generate(api, ids, {**rf, "seed": s1}, prompts)
    return api, {"seed1": s1, "seed2": s2}


def lock_photo_region(api: dict, ids: dict, mask_image: str) -> bool:
    """1차 샘플러(31)의 latent 에 노이즈 마스크를 걸어 사진 부분 latent 를 그대로 유지."""
    smp = str(ids.get("sampler_main"))
    if smp not in api:
        return False
    lat = api[smp]["inputs"].get("latent_image")
    if not isinstance(lat, list):
        return False
    api["950"] = node("LoadImageMask", {"image": mask_image, "channel": "red"}, "생성할 곳 마스크 (사진 영역 유지)")
    api["951"] = node("SetLatentNoiseMask", {"samples": lat, "mask": ["950", 0]}, "사진 영역은 다시 그리지 않음")
    api[smp]["inputs"]["latent_image"] = ["951", 0]
    return True


def green_seam_band(api: dict, ids: dict) -> bool:
    """이음새 패스: 가운데로 옮긴 이미지(99)의 띠(마스크 101/102 위치)를 초록으로 칠해 인코드(79)에 넣음."""
    shifted, enc, band, place, seam_pos = "99", "79", "101", "102", str(ids.get("seam_positive"))
    if not all(k in api for k in (shifted, enc, band, place)):
        return False
    bw = int(api[band]["inputs"].get("width", 512))
    bh = int(api[band]["inputs"].get("height", 1024))
    bx = int(api[place]["inputs"].get("x", 768))
    api["901"] = node("EmptyImage", {"width": bw, "height": bh, "batch_size": 1, "color": 0x00FF00}, "이음새 띠 초록")
    api["902"] = node("ImageCompositeMasked", {"destination": [shifted, 0], "source": ["901", 0], "x": bx, "y": 0,
                                               "resize_source": False}, "이음새 띠를 초록으로")
    api[enc]["inputs"]["pixels"] = ["902", 0]
    if seam_pos in api:
        t = api[seam_pos]["inputs"].get("text", "")
        if not t.startswith("Fill the green"):
            api[seam_pos]["inputs"]["text"] = SEAM_TRIGGER + " " + t
    if str(ids.get("sampler_seam")) in api:
        api[str(ids["sampler_seam"])]["inputs"]["denoise"] = 1.0        # 초록을 완전히 채워야 함
    return True


def seam_second_pass(api: dict, ids: dict, seed: int, band=192, feather=64, denoise=0.6) -> bool:
    """이음새 패스 뒤(85 = 이음새 복원 합성, 아직 가운데로 옮긴 상태)에 좁은 띠를 약하게 한 번 더.
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
    api["911"] = node("VAEEncode", {"pixels": [comp, 0], "vae": vae}, "이음새 2차 · 인코드")
    api["912"] = node("SolidMask", {"value": 1.0, "width": band, "height": H}, "이음새 2차 · 좁은 띠")
    api["913"] = node("MaskComposite", {"destination": ["100", 0], "source": ["912", 0], "x": x, "y": 0, "operation": "add"},
                      "이음새 2차 · 띠 위치(가운데)")
    api["914"] = node("SetLatentNoiseMask", {"samples": ["911", 0], "mask": ["913", 0]}, "이음새 2차 · 띠만 다시")
    api["915"] = node("ReferenceLatent", {"conditioning": pos_cond, "latent": ["911", 0]}, "이음새 2차 · 참조")
    api["916"] = ksampler(sm["model"], ["915", 0], sm["negative"], ["914", 0], seed=int(seed) + 1,
                          steps=sm.get("steps", 20), cfg=sm.get("cfg", 4.0), denoise=denoise,
                          sampler_name=sm.get("sampler_name", "euler"), scheduler=sm.get("scheduler", "simple"),
                          title="이음새 2차 · 약하게 다시 그림")
    api["917"] = node("VAEDecode", {"samples": ["916", 0], "vae": vae}, "이음새 2차 · 디코드")
    api["918"] = node("FeatherMask", {"mask": ["912", 0], "left": feather, "top": 0, "right": feather, "bottom": 0},
                      "이음새 2차 · 띠 가장자리 부드럽게")
    api["919"] = node("MaskComposite", {"destination": ["100", 0], "source": ["918", 0], "x": x, "y": 0, "operation": "add"},
                      "이음새 2차 · 부드러운 띠")
    api["920"] = node("ImageCompositeMasked", {"destination": [comp, 0], "source": ["917", 0], "mask": ["919", 0],
                                               "x": 0, "y": 0, "resize_source": False}, "이음새 2차 · 합성")
    for k in ("105", "106"):
        api[k]["inputs"]["image"] = ["920", 0]
    api["107"]["inputs"]["destination"] = ["920", 0]
    return True


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
