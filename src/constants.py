"""프로젝트 전역 상수 — 단일 소스."""

GAIT_CLASS_NAMES: dict[int, tuple[str, str]] = {
    0: ("normal",      "정상 보행"),
    1: ("antalgic",    "절뚝거림"),
    2: ("ataxic",      "운동실조"),
    3: ("parkinsonian","파킨슨"),
}
