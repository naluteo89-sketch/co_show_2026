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
          keep_mask: str | None = None, wrap: dict | None = None) -> tuple[dict, dict]:
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

    if st.get("cylinder", False) and not wrap:
        # 원통 생성: 다 그린 뒤 띠를 붙이지 않고, 생성 뒷부분을 이음새가 가운데 오게 돌려서 그림
        # → 0°/360° 가 캔버스 안쪽에서 양쪽을 보며 한 장으로 그려짐 (초록 띠·섞기·2차 패스 없음)
        cylinder_pass(api, ids, s1, denoise=float(st.get("cylinder_denoise", 0.6)),
                      keep=float(st.get("cylinder_keep", 0.5)), feather=int(st.get("cylinder_feather", 128)),
                      back_text=prompts.get("back_positive") or "")
        rf = st.get("refine")
        if rf and rf.get("enabled", True):
            attach_refine_to_generate(api, ids, {**rf, "seed": s1}, prompts)
        return api, {"seed1": s1, "seed2": s2}

    # 이음새 복원: 띠를 초록(#00ff00)으로 칠해 아웃페인트 LoRA 방식으로 '채우기'
    # (기존: 이음새가 그대로 있는 그림을 참조로 줘서, 편집 모델이 이음새까지 그대로 다시 그림)
    green_seam_band(api, ids)

    # 이음새 띠 가장자리에서 나무·구조물이 투명해지다 끊기는 문제: 띠 전체를 지우지 않고 가운데(core)만 새로 그림
    # → 가장자리의 원래 그림을 모델이 보고 물체를 이어 그림 → 합성 시 섞는 폭도 좁게 (내용이 같아 겹침 없음)
    if int(st.get("seam_core", 0) or 0) > 0:
        seam_core_only(api, int(st["seam_core"]), int(st.get("seam_core_feather", 32)))

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

    # 이음새 마무리 편집: 이음새 둘레를 편집 모델로 '한 장의 자연스러운 사진'으로 다시 그려 겹침·선·밝기 단차 정리
    if st.get("seam_edit", False) and not wrap:
        seam_edit_pass(api, ids, s2)

    if wrap:
        # 사진을 0°/360° 에 걸쳐 놓고 생성한 경우: 이음새는 사진 위(하늘)·아래(바닥)에만 있음
        # → 사진 줄은 이음새 복원에서 빼고, 끝에서 원래 방향으로 돌림
        seam_skip_rows(api, int(wrap["top"]), int(wrap["bottom"]))
        wrap_roll(api, int(wrap["shift"]))

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


def wrap_shift_for(yaw: float, W: int) -> tuple[int, float]:
    """사진을 생성 캔버스 양끝(0°/360°)에 걸쳐 놓기: 최종 = np.roll(생성, s).
    반환 (s, 생성용 yaw). 사진 가운데가 캔버스 끝(x=0/W)에 오고, 결과는 s 만큼 돌려 원래 yaw 로."""
    xf = W * (0.5 + float(yaw) / 360.0)               # 최종에서 사진 가운데 x
    s = int(round(xf)) % W                           # 생성 캔버스에서는 x=0(=W) 에 있게
    yaw_gen = ((xf - s) / W - 0.5) * 360.0
    return s, yaw_gen


def _hole(api: dict, nid: str, width: int, top: int, bottom: int, feather: int):
    """사진 줄(top~bottom) 구멍 마스크 (가장자리 feather px 부드럽게)."""
    api[nid] = node("SolidMask", {"value": 1.0, "width": width, "height": bottom - top}, "사진 걸침 · 사진 줄")
    if feather > 0:
        api[nid + "f"] = node("FeatherMask", {"mask": [nid, 0], "left": 0, "top": feather, "right": 0, "bottom": feather},
                              "사진 걸침 · 사진 줄 가장자리")
        return [nid + "f", 0]
    return [nid, 0]


def seam_skip_rows(api: dict, top: int, bottom: int, feather: int = 16) -> bool:
    """이음새 복원(1차·2차)을 사진 줄(top~bottom)에선 하지 않음: 초록은 위·아래만, 마스크는 사진 줄을 뺌."""
    if not all(k in api for k in ("901", "902", "102", "104", "101", "100")) or bottom - top < 8:
        return False
    H = int(api["100"]["inputs"].get("height", 1024))
    bw, bx = int(api["101"]["inputs"]["width"]), int(api["902"]["inputs"]["x"])
    # 초록: 사진 위(0~top)·아래(bottom~H)만
    api["901"]["inputs"]["height"] = max(8, top)
    api["903"] = node("EmptyImage", {"width": bw, "height": max(8, H - bottom), "batch_size": 1, "color": 0x00FF00},
                      "이음새 띠 초록 (사진 아래)")
    api["904"] = node("ImageCompositeMasked", {"destination": ["902", 0], "source": ["903", 0], "x": bx, "y": bottom,
                                               "resize_source": False}, "이음새 띠를 초록으로 (사진 아래)")
    api["79"]["inputs"]["pixels"] = ["904", 0]
    # 1차 노이즈 마스크(102)·합성 마스크(104), 2차(913·919)에서 사진 줄 빼기
    pairs = [("102", bx, bw, 0, "940"), ("104", bx, bw, feather, "942")]
    if "913" in api:
        bx2, bw2 = int(api["913"]["inputs"]["x"]), int(api["912"]["inputs"]["width"])
        pairs += [("913", bx2, bw2, 0, "944"), ("919", bx2, bw2, feather, "946")]
    for mid, x, w, f, nid in pairs:
        hole = _hole(api, nid, w, top, bottom, f)
        sub = nid + "s"
        api[sub] = node("MaskComposite", {"destination": [mid, 0], "source": hole, "x": x, "y": top,
                                          "operation": "subtract"}, "사진 걸침 · 사진 줄 빼기")
        for k, n in api.items():                      # mid 를 쓰던 곳 → sub
            if k == sub:
                continue
            for field, v in n["inputs"].items():
                if v == [mid, 0]:
                    n["inputs"][field] = [sub, 0]
    return True


def wrap_roll(api: dict, s: int) -> bool:
    """이음새 복원 결과(108)를 s 만큼 가로로 돌려 원래 방향으로 → 업스케일 패딩(110·111·112)이 이것을 씀."""
    if not all(k in api for k in ("108", "110", "111", "112", "100")):
        return False
    W = int(api["100"]["inputs"].get("width", 2048))
    s %= W
    if s == 0:
        return True
    api["960"] = node("ImageCrop", {"image": ["108", 0], "width": s, "height": int(api["100"]["inputs"]["height"]),
                                    "x": W - s, "y": 0}, "사진 걸침 → 원래 방향: 오른쪽 끝")
    api["961"] = node("ImageCrop", {"image": ["108", 0], "width": W - s, "height": int(api["100"]["inputs"]["height"]),
                                    "x": 0, "y": 0}, "사진 걸침 → 원래 방향: 나머지")
    api["962"] = node("ImageCompositeMasked", {"destination": ["108", 0], "source": ["960", 0], "x": 0, "y": 0,
                                               "resize_source": False}, "사진 걸침 → 원래 방향: 앞")
    api["963"] = node("ImageCompositeMasked", {"destination": ["962", 0], "source": ["961", 0], "x": s, "y": 0,
                                               "resize_source": False}, "사진 걸침 → 원래 방향: 뒤")
    for k in ("110", "111"):
        api[k]["inputs"]["image"] = ["963", 0]
    api["112"]["inputs"]["source"] = ["963", 0]
    return True


def _roll_half(api: dict, src: list, W: int, H: int, base: int, title: str) -> list:
    """그림을 가로로 반 바퀴 돌림 (오른쪽 반 → 앞, 왼쪽 반 → 뒤). 결과 노드 [id, 0]."""
    a, b, c, d = (str(base + i) for i in range(4))
    h = W // 2
    api[a] = node("ImageCrop", {"image": src, "width": W - h, "height": H, "x": h, "y": 0}, f"{title} · 오른쪽 반")
    api[b] = node("ImageCrop", {"image": src, "width": h, "height": H, "x": 0, "y": 0}, f"{title} · 왼쪽 반")
    api[c] = node("ImageCompositeMasked", {"destination": src, "source": [a, 0], "x": 0, "y": 0, "resize_source": False},
                  f"{title} · 앞")
    api[d] = node("ImageCompositeMasked", {"destination": [c, 0], "source": [b, 0], "x": W - h, "y": 0,
                                           "resize_source": False}, f"{title} · 뒤")
    return [d, 0]


def cylinder_pass(api: dict, ids: dict, seed: int, denoise: float = 0.6, keep: float = 0.5, feather: int = 128,
                  back_text: str = "") -> bool:
    """원통 생성 (캔버스 양 끝이 붙어 있다는 걸 알고 그리게).
    1차 생성(8) → 반 바퀴 돌림(99, 0°/360° 가 가운데) → 같은 모델·프롬프트로 뒷부분(denoise)을 다시 그림
    (참조 control 도 똑같이 돌림) → 가운데 keep 폭만 새 그림으로, 경계는 feather 로 (경계 = ±90°, 두 그림 내용이 같은 곳)
    → 다시 반 바퀴 (105~108) → 8K 단계.
    denoise 0.6 = 4스텝 중 뒤 절반 남짓을 이음새가 가운데인 상태로 그리는 것과 같음 (구도는 1차에서 정해짐).
    back_text: 뒤쪽 전용 프롬프트 (좌우 가장자리가 뒤에서 어떻게 만나는지 · 주인공 문장 없음). 없으면 본 프롬프트."""
    main, vae_id = str(ids["sampler_main"]), str(ids["vae"])
    need = ("8", "99", "100", "105", "106", "107", "71", str(ids["positive"]), str(ids["negative"]), main)
    if not all(k in api for k in need):
        return False
    W = int(api["100"]["inputs"].get("width", 2048))
    H = int(api["100"]["inputs"].get("height", 1024))
    sm = api[main]["inputs"]
    vae = [vae_id, 0]
    ctrl = _roll_half(api, ["71", 0], W, H, 1001, "원통 · 참조 control 돌림")
    api["1005"] = node("VAEEncode", {"pixels": ctrl, "vae": vae}, "원통 · 돌린 control 인코드")
    cond = [str(ids["positive"]), 0]
    clip = api[str(ids["positive"])]["inputs"].get("clip")
    if back_text.strip() and clip:
        api["1014"] = node("CLIPTextEncode", {"text": back_text, "clip": clip}, "원통 · 뒤쪽 프롬프트")
        cond = ["1014", 0]
    api["1006"] = node("ReferenceLatent", {"conditioning": cond, "latent": ["1005", 0]},
                       "원통 · 참조 (돌린 control)")
    api["1007"] = node("VAEEncode", {"pixels": ["99", 0], "vae": vae}, "원통 · 돌린 1차 결과 인코드")
    api["1008"] = ksampler(sm["model"], ["1006", 0], [str(ids["negative"]), 0], ["1007", 0], seed=int(seed) + 7,
                           steps=sm.get("steps", 20), cfg=sm.get("cfg", 4.0), denoise=float(denoise),
                           sampler_name=sm.get("sampler_name", "euler"), scheduler=sm.get("scheduler", "simple"),
                           title="원통 · 이음새가 가운데인 상태로 뒷부분 다시 그림")
    api["1009"] = node("VAEDecode", {"samples": ["1008", 0], "vae": vae}, "원통 · 디코드")
    kw = max(2 * feather + 64, min(W - 2 * feather, int(W * keep)))
    x = (W - kw) // 2
    api["1010"] = node("SolidMask", {"value": 1.0, "width": kw, "height": H}, "원통 · 새 그림 쓸 범위 (가운데)")
    api["1011"] = node("FeatherMask", {"mask": ["1010", 0], "left": feather, "top": 0, "right": feather, "bottom": 0},
                       "원통 · 경계 부드럽게")
    api["1012"] = node("MaskComposite", {"destination": ["100", 0], "source": ["1011", 0], "x": x, "y": 0,
                                         "operation": "add"}, "원통 · 합성 마스크")
    api["1013"] = node("ImageCompositeMasked", {"destination": ["99", 0], "source": ["1009", 0], "mask": ["1012", 0],
                                                "x": 0, "y": 0, "resize_source": False}, "원통 · 합성")
    for k in ("105", "106"):
        api[k]["inputs"]["image"] = ["1013", 0]
    api["107"]["inputs"]["destination"] = ["1013", 0]
    # 쓰지 않는 이음새 복원 노드는 빼서 실행되지 않게
    for k in ("77", "78", "79", "80", "82", "83", "84", "85", "101", "102", "103", "104", "630"):
        api.pop(k, None)
    return True


def seam_core_only(api: dict, core: int = 256, feather: int = 32) -> bool:
    """1차 이음새 복원(초록 띠 채우기)을 띠 가운데 core px 로 좁힘.
    - 초록(901/902)·노이즈 마스크(82)는 가운데 core 만 (딱 끊김 — 바깥 latent 는 그대로라 모델이 경계에서 이어 그림)
    - 합성 마스크(85)는 core 안쪽 feather px 만 부드럽게 (예전: 512 띠 안쪽 128px 를 투명도로 섞어 물체가 반투명해지다 끊김)
    띠 바깥과 core 바깥 ~ 띠 끝 사이는 원래 그림 그대로."""
    if not all(k in api for k in ("901", "902", "82", "85", "100", "101", "102")):
        return False
    W = int(api["100"]["inputs"].get("width", 2048))
    H = int(api["100"]["inputs"].get("height", 1024))
    bw = int(api["101"]["inputs"].get("width", 512))
    core = max(64, min(core, bw))
    feather = max(0, min(feather, core // 2 - 1))
    x = W // 2 - core // 2
    api["901"]["inputs"]["width"] = core
    api["902"]["inputs"]["x"] = x
    api["905"] = node("SolidMask", {"value": 1.0, "width": core, "height": H}, "이음새 가운데만 (core)")
    api["906"] = node("MaskComposite", {"destination": ["100", 0], "source": ["905", 0], "x": x, "y": 0, "operation": "add"},
                      "이음새 가운데만 · 노이즈 마스크 (딱 끊김)")
    api["82"]["inputs"]["mask"] = ["906", 0]
    api["907"] = node("FeatherMask", {"mask": ["905", 0], "left": feather, "top": 0, "right": feather, "bottom": 0},
                      "이음새 가운데만 · 합성 경계 좁게")
    api["908"] = node("MaskComposite", {"destination": ["100", 0], "source": ["907", 0], "x": x, "y": 0, "operation": "add"},
                      "이음새 가운데만 · 합성 마스크")
    api["85"]["inputs"]["mask"] = ["908", 0]
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


SEAM_EDIT_PROMPT = ("Edit this photo into one natural, continuous photograph. Remove any semi-transparent ghosted or "
                    "doubled shapes, vertical lines, seams and brightness steps. Keep the same scene, objects, layout, "
                    "colors, lighting and sharpness. Photorealistic.")


def seam_edit_pass(api: dict, ids: dict, seed: int, size: int = 1024, feather: int = 256) -> bool:
    """이음새 복원이 끝난 그림(920, 이음새가 가운데)에서 가운데 size×H 를 잘라 편집 모델(LoRA 없음)로 다시 그림
    → 가운데는 편집 결과, 양옆 feather px 동안 원래 그림으로 서서히 → 되돌리기(105~107)가 이 결과를 씀."""
    src = "920" if "920" in api else "85"
    seam = str(ids.get("sampler_seam"))
    if not all(k in api for k in (src, seam, "100", "105", "106", "107", "79")):
        return False
    W = int(api["100"]["inputs"].get("width", 2048))
    H = int(api["100"]["inputs"].get("height", 1024))
    x = (W - size) // 2
    vae = api["79"]["inputs"]["vae"]
    sm = api[seam]["inputs"]
    clip = api[str(ids["positive"])]["inputs"]["clip"]
    api["980"] = node("ImageCrop", {"image": [src, 0], "width": size, "height": H, "x": x, "y": 0}, "이음새 편집 · 가운데 자르기")
    api["981"] = node("VAEEncode", {"pixels": ["980", 0], "vae": vae}, "이음새 편집 · 인코드")
    api["982"] = node("CLIPTextEncode", {"text": SEAM_EDIT_PROMPT, "clip": clip}, "이음새 편집 · 지시문")
    api["983"] = node("ReferenceLatent", {"conditioning": ["982", 0], "latent": ["981", 0]}, "이음새 편집 · 참조(원래 그림)")
    api["984"] = ksampler([str(ids["unet"]), 0], ["983", 0], sm["negative"], ["981", 0], seed=int(seed) + 2,
                          steps=sm.get("steps", 20), cfg=sm.get("cfg", 4.0), denoise=1.0,
                          sampler_name=sm.get("sampler_name", "euler"), scheduler=sm.get("scheduler", "simple"),
                          title="이음새 편집 · 다시 그림 (LoRA 없음)")
    api["985"] = node("VAEDecode", {"samples": ["984", 0], "vae": vae}, "이음새 편집 · 디코드")
    api["986"] = node("SolidMask", {"value": 1.0, "width": size, "height": H}, "이음새 편집 · 마스크")
    api["987"] = node("FeatherMask", {"mask": ["986", 0], "left": feather, "top": 0, "right": feather, "bottom": 0},
                      "이음새 편집 · 양옆 부드럽게")
    api["988"] = node("ImageCompositeMasked", {"destination": [src, 0], "source": ["985", 0], "mask": ["987", 0],
                                               "x": x, "y": 0, "resize_source": False}, "이음새 편집 · 합성")
    for k in ("105", "106"):
        api[k]["inputs"]["image"] = ["988", 0]
    api["107"]["inputs"]["destination"] = ["988", 0]
    return True


def draft_only(api: dict, ids: dict, prefix: str) -> str:
    """2K 초안까지만: 이음새 복원 결과(108)만 저장하고 8K 단계(최종 저장)는 실행하지 않음. 저장 노드 id 반환."""
    api.pop(str(ids["save_final"]), None)
    api["973"] = node("SaveImage", {"images": ["108", 0], "filename_prefix": f"{prefix}_draft_2k"}, "2K 초안 (검수용)")
    return "973"


def final_from_draft(api: dict, ids: dict, image: str) -> str:
    """검수 통과한 2K 초안(LoadImage) → 같은 8K 단계(패딩 → 업스케일 → 리파인 → 디더 → 저장). 저장 노드 id 반환."""
    api["970"] = node("LoadImage", {"image": image}, "검수 통과 2K 초안")
    for k in ("110", "111"):
        api[k]["inputs"]["image"] = ["970", 0]
    api["112"]["inputs"]["source"] = ["970", 0]
    api.pop("973", None)
    return str(ids["save_final"])


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
