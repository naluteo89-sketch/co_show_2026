"""
요청 처리 — 기능별 mixin 을 Studio 한 클래스로 합친다.

    base      공통 (응답 · 잠금 · 실행 위치 · 경로 · meta.json 갱신)
    info      화면 시작 정보 · 모델 점검 · 작업 목록
    generate  업로드 · 분석 · 컨트롤 이미지 · 생성 · 진행/결과 후처리
    pole      발밑·하늘 보정
    photo     원본 사진 다시 붙이기
    motion    영상화 (일부만 움직임)
    event     사건 연출 · 마스크 · 360 영상
    viewer    Unity 뷰어로 보내기
"""

from .base import StudioBase
from .event import EventMixin
from .generate import GenerateMixin
from .info import InfoMixin
from .motion import MotionMixin
from .photo import PhotoMixin
from .pole import PoleMixin
from .viewer import ViewerMixin


class Studio(InfoMixin, GenerateMixin, PoleMixin, PhotoMixin, MotionMixin, EventMixin, ViewerMixin, StudioBase):
    pass
