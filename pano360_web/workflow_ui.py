"""
ComfyUI 화면용 워크플로(.json, 끌어다 놓으면 열림) 만들기 — 사건 영상용 이미지→영상.

  wan14: Wan 2.2 I2V 14B fp8 + 4스텝 LoRA (24GB 에서 720p, 품질·움직임 좋음)
  wan5:  Wan 2.2 TI2V 5B (8~16GB 에서도, 가벼움)
기본 노드만 사용 (커스텀 노드 없음) → 내 PC · 서버 · Comfy Cloud 같은 파일.
"""


class _G:
    def __init__(self):
        self.nodes, self.links = [], []

    def node(self, id, type, pos, inputs, outputs, widgets, size=(320, 120), title=None, color=None):
        n = {"id": id, "type": type, "pos": list(pos), "size": list(size), "flags": {}, "order": id - 1, "mode": 0,
             "inputs": [{"name": a, "type": t, "link": None} for a, t in inputs],
             "outputs": [{"name": a, "type": t, "links": [], "slot_index": i} for i, (a, t) in enumerate(outputs)],
             "properties": {"Node name for S&R": type}, "widgets_values": widgets}
        if title:
            n["title"] = title
        if color:
            n["color"] = color
        self.nodes.append(n)

    def link(self, a, ao, b, bi):
        l = len(self.links) + 1
        A = next(n for n in self.nodes if n["id"] == a)
        B = next(n for n in self.nodes if n["id"] == b)
        A["outputs"][ao]["links"].append(l)
        B["inputs"][bi]["link"] = l
        self.links.append([l, a, ao, b, bi, A["outputs"][ao]["type"]])

    def done(self):
        for n in self.nodes:
            for o in n["outputs"]:
                if not o["links"]:
                    o["links"] = None
        return {"last_node_id": max(n["id"] for n in self.nodes), "last_link_id": len(self.links),
                "nodes": self.nodes, "links": self.links, "groups": [], "config": {}, "extra": {}, "version": 0.4}


NOTE = ("사건 영상 (한 번 일어나는 연출): 잔잔 → 물이 넘쳐 화면을 덮음 → 다시 잔잔.\n"
        "- ① 에 웹앱에서 받은 '사건 화면' PNG 를 넣고 실행.\n"
        "- 결과 MP4 를 웹앱 6단계 '생성 영상 올리기'에 올리면 360 에 합쳐 줘요.\n"
        "- 원본에서 달라진 곳만 영상이 쓰이고, 나머지는 8K 원본이에요.\n"
        "- 크기는 16 배수(5B 는 32), 길이는 4n+1. 메모리가 모자라면 길이·크기를 줄이세요.")


def wan_event_workflow(positive: str, negative: str, image: str = "event_view.png", variant="wan14",
                       width=1280, height=720, seconds=5.0) -> dict:
    g = _G()
    g.node(1, "LoadImage", (0, 0), [], [("IMAGE", "IMAGE"), ("MASK", "MASK")], [image, "image"], (340, 340),
           "① 사건 화면 (웹앱에서 받은 PNG)")
    g.node(4, "CLIPLoader", (0, 380), [], [("CLIP", "CLIP")], ["umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default"],
           (340, 110), "텍스트 인코더 (umT5)")
    g.node(6, "CLIPTextEncode", (380, 0), [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")], [positive], (520, 240),
           "② Positive (웹앱 '사건 프롬프트')", "#232")
    g.node(7, "CLIPTextEncode", (380, 270), [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")], [negative], (520, 140),
           "Negative", "#322")
    g.node(20, "Note", (380, 440), [], [], [NOTE], (520, 200), "메모")
    if variant == "wan5":
        fps = 24
        L = int(round(seconds * fps)) // 4 * 4 + 1
        g.node(2, "UNETLoader", (0, 530), [], [("MODEL", "MODEL")], ["wan2.2_ti2v_5B_fp16.safetensors", "default"],
               (340, 90), "Wan 2.2 TI2V 5B")
        g.node(3, "ModelSamplingSD3", (0, 660), [("model", "MODEL")], [("MODEL", "MODEL")], [8.0], (260, 60), "shift 8")
        g.node(5, "VAELoader", (0, 760), [], [("VAE", "VAE")], ["wan2.2_vae.safetensors"], (340, 60), "Wan 2.2 VAE")
        g.node(8, "Wan22ImageToVideoLatent", (940, 0), [("vae", "VAE"), ("start_image", "IMAGE")], [("LATENT", "LATENT")],
               [width // 32 * 32, height // 32 * 32, L, 1], (320, 150), f"③ 크기·길이 ({L}프레임 = {L / fps:.1f}초)")
        g.node(9, "KSampler", (940, 190), [("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                                          ("latent_image", "LATENT")], [("LATENT", "LATENT")],
               [0, "randomize", 30, 5.0, "uni_pc", "simple", 1.0], (320, 270), "④ 생성")
        g.link(2, 0, 3, 0); g.link(3, 0, 9, 0); g.link(4, 0, 6, 0); g.link(4, 0, 7, 0)
        g.link(5, 0, 8, 0); g.link(1, 0, 8, 1); g.link(6, 0, 9, 1); g.link(7, 0, 9, 2); g.link(8, 0, 9, 3)
        last, vae = 9, 5
    else:                                                    # Wan 2.2 I2V 14B + 4스텝 LoRA (ComfyUI 기본 템플릿 구성)
        fps = 16
        L = int(round(seconds * fps)) // 4 * 4 + 1
        g.node(2, "UNETLoader", (0, 530), [], [("MODEL", "MODEL")], ["wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "default"],
               (340, 90), "14B 고노이즈")
        g.node(12, "UNETLoader", (0, 640), [], [("MODEL", "MODEL")], ["wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "default"],
               (340, 90), "14B 저노이즈")
        g.node(13, "LoraLoaderModelOnly", (0, 750), [("model", "MODEL")], [("MODEL", "MODEL")],
               ["wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", 1.0], (340, 90), "4스텝 LoRA (고)")
        g.node(14, "LoraLoaderModelOnly", (0, 860), [("model", "MODEL")], [("MODEL", "MODEL")],
               ["wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", 1.0], (340, 90), "4스텝 LoRA (저)")
        g.node(3, "ModelSamplingSD3", (380, 680), [("model", "MODEL")], [("MODEL", "MODEL")], [5.0], (260, 60), "shift 5 (고)")
        g.node(15, "ModelSamplingSD3", (380, 770), [("model", "MODEL")], [("MODEL", "MODEL")], [5.0], (260, 60), "shift 5 (저)")
        g.node(5, "VAELoader", (0, 970), [], [("VAE", "VAE")], ["wan_2.1_vae.safetensors"], (340, 60), "Wan VAE")
        g.node(8, "WanImageToVideo", (940, 0), [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("vae", "VAE"),
                                                ("clip_vision_output", "CLIP_VISION_OUTPUT"), ("start_image", "IMAGE")],
               [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")],
               [width // 16 * 16, height // 16 * 16, L, 1], (320, 170), f"③ 크기·길이 ({L}프레임 = {L / fps:.1f}초)")
        g.node(9, "KSamplerAdvanced", (940, 210), [("model", "MODEL"), ("positive", "CONDITIONING"),
                                                  ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
               [("LATENT", "LATENT")], ["enable", 0, "randomize", 4, 1, "euler", "simple", 0, 2, "enable"],
               (320, 330), "④ 생성 1/2 (고노이즈)")
        g.node(16, "KSamplerAdvanced", (1280, 210), [("model", "MODEL"), ("positive", "CONDITIONING"),
                                                    ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
               [("LATENT", "LATENT")], ["disable", 0, "fixed", 4, 1, "euler", "simple", 2, 4, "disable"],
               (320, 330), "④ 생성 2/2 (저노이즈)")
        g.link(2, 0, 13, 0); g.link(13, 0, 3, 0); g.link(12, 0, 14, 0); g.link(14, 0, 15, 0)
        g.link(4, 0, 6, 0); g.link(4, 0, 7, 0)
        g.link(6, 0, 8, 0); g.link(7, 0, 8, 1); g.link(5, 0, 8, 2); g.link(1, 0, 8, 4)
        g.link(3, 0, 9, 0); g.link(8, 0, 9, 1); g.link(8, 1, 9, 2); g.link(8, 2, 9, 3)
        g.link(15, 0, 16, 0); g.link(8, 0, 16, 1); g.link(8, 1, 16, 2); g.link(9, 0, 16, 3)
        last, vae = 16, 5
    g.node(10, "VAEDecode", (1640, 0), [("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")], [], (220, 50))
    g.node(11, "CreateVideo", (1640, 90), [("images", "IMAGE")], [("VIDEO", "VIDEO")], [fps], (220, 80), f"{fps}fps")
    g.node(17, "SaveVideo", (1640, 210), [("video", "VIDEO")], [], ["video/pano360_event", "auto", "auto"], (420, 400),
           "⑤ 저장 → 웹앱에 올리기")
    g.link(last, 0, 10, 0); g.link(vae, 0, 10, 1); g.link(10, 0, 11, 0); g.link(11, 0, 17, 0)
    return g.done()
