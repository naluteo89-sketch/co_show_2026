"""
ERP(등장방형 360) 이미지 계산

    projection  투영 공통 (방향·카메라 평면·샘플링)
    control     컨트롤 이미지 · 원본 보존 마스크 · Quest JPG
    pole        발밑·하늘 보정
    paste       원본 사진 다시 붙이기 · 톤 맞춤
    seam        좌우 이음새 후처리
    regions     영상화·사건 영역 화면 · 미리보기 · 마스크
"""

from .projection import GREEN, load_upright, exif_hfov, pitch_from_horizon, roll_array
from .control import make_control, keep_mask, to_2to1, save_quest_jpg
from .pole import pole_cap_weight, pole_prepare, pole_merge, pole_converge
from .paste import photo_composite, tone_match, photo_paste_mode, harmonize_paste, paste_photo
from .seam import wrap_line_columns, fix_wrap_line, match_wrap_tone, wrap_seam_score, blend_wrap_band, bridge_wrap_sky, continuity_report
from .regions import (ASPECT_MIN, ASPECT_MAX, MOTION_COLORS, region_aspect, region_size, perspective, grid_overlay,
                      region_outline, region_mask, motion_overlay)
