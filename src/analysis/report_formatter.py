"""한국어 보고서 포매터.

모든 분석 모듈의 보고서 생성에서 공통으로 사용되는 포매팅 유틸리티.
"""

from .common import severity_label


# ── 공통 포매팅 상수 ──────────────────────────────────────────────────
DIVIDER = "─" * 65
HEADER_DIVIDER = "=" * 65


def header(title: str) -> str:
    """보고서 헤더 블록."""
    return f"{HEADER_DIVIDER}\n  {title}\n{HEADER_DIVIDER}"


def section(title: str) -> str:
    """섹션 구분선 + 제목."""
    return f"{DIVIDER}\n  [{title}]"


def risk_bar(score: float, width: int = 15) -> str:
    """위험도 막대 문자열. score: 0~1, width: 막대 길이."""
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def risk_line(
    label: str,
    score: float,
    bar_width: int = 15,
    extra: str = "",
) -> str:
    """위험도 한 줄 포매팅: 라벨 [████░░░] 42% (주의) extra."""
    bar = risk_bar(score, bar_width)
    grade = severity_label(score)
    line = f"  {label:16s} [{bar}] {score:.0%} ({grade})"
    if extra:
        line += f" {extra}"
    return line


def marker_line(
    label: str,
    score: float,
    bar_width: int = 15,
    extra: str = "",
) -> str:
    """마커 포함 위험도 줄: ▲/○ 라벨 [████░░░] 42% (주의)."""
    marker = "▲" if score > 0 else "○"
    bar = risk_bar(score, bar_width)
    grade = severity_label(score)
    line = f"  {marker} {label:16s} [{bar}] {score:.0%} ({grade})"
    if extra:
        line += f" {extra}"
    return line


def ranked_line(
    rank: int,
    label: str,
    score: float,
    bar_width: int = 20,
) -> str:
    """순위 포함 줄: 1. 라벨 [████████░░░░] 65.2%."""
    bar = risk_bar(score, bar_width)
    return f"  {rank}. {label:14s} [{bar}] {score:.1%}"


def overall_summary_line(health_score: float) -> tuple[str, str]:
    """건강/위험 점수에 따른 등급과 코멘트.

    Returns: (등급, 설명)
    """
    if health_score >= 85:
        return "양호", "양호한 보행 패턴입니다."
    elif health_score >= 70:
        return "보통", "일부 항목에서 주의가 필요합니다."
    elif health_score >= 50:
        return "주의", "여러 항목에서 이상 소견이 있습니다."
    else:
        return "경고", "전문가 상담이 필요합니다."
