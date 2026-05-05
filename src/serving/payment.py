"""결제 수단 관리 API 모듈.

보행 분석 SaaS 서비스를 위한 결제 수단 CRUD API.
SQLite 기반 경량 데이터베이스를 사용하며, 구독 플랜 관리를 지원합니다.

엔드포인트:
    POST   /payments/methods          - 결제 수단 등록
    GET    /payments/methods           - 결제 수단 목록 조회
    GET    /payments/methods/{id}      - 결제 수단 상세 조회
    PUT    /payments/methods/{id}      - 결제 수단 수정
    DELETE /payments/methods/{id}      - 결제 수단 삭제
    POST   /payments/methods/{id}/default - 기본 결제 수단 설정
    GET    /payments/plans             - 구독 플랜 목록
    POST   /payments/subscribe         - 구독 신청
"""

import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_DB_PATH: str = "data/payments.db"


def set_db_path(path: str) -> None:
    """테스트 등에서 DB 경로를 변경할 때 사용."""
    global _DB_PATH
    _DB_PATH = path


@contextmanager
def get_db():
    """SQLite 연결 컨텍스트 매니저."""
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """데이터베이스 테이블 초기화 (마이그레이션 v1 적용)."""
    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.executescript(_MIGRATION_V1)
        conn.executescript(_MIGRATION_V2_SEED_PLANS)
    logger.info("Payment database initialized: %s", _DB_PATH)


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------

_MIGRATION_V1 = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS payment_methods (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    method_type TEXT NOT NULL CHECK(method_type IN ('card', 'bank_transfer', 'kakao_pay', 'naver_pay', 'toss_pay')),
    label TEXT NOT NULL,
    card_last4 TEXT,
    card_brand TEXT,
    bank_name TEXT,
    account_last4 TEXT,
    is_default INTEGER NOT NULL DEFAULT 0,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_payment_methods_user
    ON payment_methods(user_id, is_active);

CREATE TABLE IF NOT EXISTS subscription_plans (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    name_kr TEXT NOT NULL,
    price_krw INTEGER NOT NULL,
    interval TEXT NOT NULL CHECK(interval IN ('monthly', 'yearly')),
    max_analyses_per_month INTEGER NOT NULL,
    features TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS subscriptions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    plan_id TEXT NOT NULL REFERENCES subscription_plans(id),
    payment_method_id TEXT NOT NULL REFERENCES payment_methods(id),
    status TEXT NOT NULL CHECK(status IN ('active', 'cancelled', 'past_due', 'trialing')),
    current_period_start TEXT NOT NULL,
    current_period_end TEXT NOT NULL,
    created_at TEXT NOT NULL,
    cancelled_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_subscriptions_user
    ON subscriptions(user_id, status);

CREATE TABLE IF NOT EXISTS payment_history (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    subscription_id TEXT REFERENCES subscriptions(id),
    payment_method_id TEXT NOT NULL REFERENCES payment_methods(id),
    amount_krw INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'pending', 'refunded')),
    description TEXT,
    created_at TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_version (version, applied_at)
    VALUES (1, datetime('now'));
"""

_MIGRATION_V2_SEED_PLANS = """
INSERT OR IGNORE INTO subscription_plans (id, name, name_kr, price_krw, interval, max_analyses_per_month, features)
VALUES
    ('plan_free', 'Free', '무료', 0, 'monthly', 5,
     '기본 보행 분류,월 5회 분석'),
    ('plan_basic', 'Basic', '베이직', 29000, 'monthly', 50,
     '보행 분류,질병 위험도,교정 피드백,월 50회 분석'),
    ('plan_pro', 'Professional', '프로페셔널', 79000, 'monthly', 500,
     '전체 기능,API 접근,대시보드,우선 지원,월 500회 분석'),
    ('plan_enterprise', 'Enterprise', '엔터프라이즈', 290000, 'monthly', -1,
     '전체 기능,무제한 분석,전용 서버,SLA 보장,맞춤 모델 학습');

INSERT OR IGNORE INTO schema_version (version, applied_at)
    VALUES (2, datetime('now'));
"""


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MethodType(str, Enum):
    CARD = "card"
    BANK_TRANSFER = "bank_transfer"
    KAKAO_PAY = "kakao_pay"
    NAVER_PAY = "naver_pay"
    TOSS_PAY = "toss_pay"


class PaymentMethodCreate(BaseModel):
    """결제 수단 등록 요청."""
    user_id: str = Field(..., min_length=1, max_length=64)
    method_type: MethodType
    label: str = Field(..., min_length=1, max_length=100, description="표시 이름 (예: '내 신한카드')")
    card_last4: Optional[str] = Field(None, pattern=r"^\d{4}$")
    card_brand: Optional[str] = Field(None, max_length=20)
    bank_name: Optional[str] = Field(None, max_length=30)
    account_last4: Optional[str] = Field(None, pattern=r"^\d{4}$")

    @field_validator("card_last4")
    @classmethod
    def card_required_for_card_type(cls, v, info):
        if info.data.get("method_type") == MethodType.CARD and v is None:
            raise ValueError("card_last4 is required for card payment method")
        return v


class PaymentMethodUpdate(BaseModel):
    """결제 수단 수정 요청."""
    label: Optional[str] = Field(None, min_length=1, max_length=100)
    is_active: Optional[bool] = None


class PaymentMethodResponse(BaseModel):
    """결제 수단 응답."""
    id: str
    user_id: str
    method_type: str
    label: str
    card_last4: Optional[str] = None
    card_brand: Optional[str] = None
    bank_name: Optional[str] = None
    account_last4: Optional[str] = None
    is_default: bool
    is_active: bool
    created_at: str
    updated_at: str


class PlanResponse(BaseModel):
    """구독 플랜 응답."""
    id: str
    name: str
    name_kr: str
    price_krw: int
    interval: str
    max_analyses_per_month: int
    features: list[str]


class SubscribeRequest(BaseModel):
    """구독 신청 요청."""
    user_id: str = Field(..., min_length=1)
    plan_id: str = Field(..., min_length=1)
    payment_method_id: str = Field(..., min_length=1)


class SubscriptionResponse(BaseModel):
    """구독 응답."""
    id: str
    user_id: str
    plan_id: str
    payment_method_id: str
    status: str
    current_period_start: str
    current_period_end: str
    created_at: str


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_method(row: sqlite3.Row) -> PaymentMethodResponse:
    return PaymentMethodResponse(
        id=row["id"],
        user_id=row["user_id"],
        method_type=row["method_type"],
        label=row["label"],
        card_last4=row["card_last4"],
        card_brand=row["card_brand"],
        bank_name=row["bank_name"],
        account_last4=row["account_last4"],
        is_default=bool(row["is_default"]),
        is_active=bool(row["is_active"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def create_payment_method(data: PaymentMethodCreate) -> PaymentMethodResponse:
    """결제 수단 DB 삽입."""
    method_id = f"pm_{uuid.uuid4().hex[:16]}"
    now = _now_iso()

    with get_db() as conn:
        # 첫 번째 결제 수단이면 기본으로 설정
        existing = conn.execute(
            "SELECT COUNT(*) AS cnt FROM payment_methods WHERE user_id=? AND is_active=1",
            (data.user_id,),
        ).fetchone()
        is_default = 1 if existing["cnt"] == 0 else 0

        conn.execute(
            """INSERT INTO payment_methods
               (id, user_id, method_type, label, card_last4, card_brand,
                bank_name, account_last4, is_default, is_active, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (method_id, data.user_id, data.method_type.value, data.label,
             data.card_last4, data.card_brand, data.bank_name, data.account_last4,
             is_default, now, now),
        )

        row = conn.execute(
            "SELECT * FROM payment_methods WHERE id=?", (method_id,),
        ).fetchone()

    return _row_to_method(row)


def get_payment_methods(user_id: str, include_inactive: bool = False) -> list[PaymentMethodResponse]:
    """사용자의 결제 수단 목록 조회."""
    with get_db() as conn:
        if include_inactive:
            rows = conn.execute(
                "SELECT * FROM payment_methods WHERE user_id=? ORDER BY is_default DESC, created_at DESC",
                (user_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM payment_methods WHERE user_id=? AND is_active=1 ORDER BY is_default DESC, created_at DESC",
                (user_id,),
            ).fetchall()
    return [_row_to_method(r) for r in rows]


def get_payment_method(method_id: str) -> Optional[PaymentMethodResponse]:
    """단일 결제 수단 조회."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM payment_methods WHERE id=?", (method_id,),
        ).fetchone()
    return _row_to_method(row) if row else None


def update_payment_method(method_id: str, data: PaymentMethodUpdate) -> Optional[PaymentMethodResponse]:
    """결제 수단 수정."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM payment_methods WHERE id=?", (method_id,)).fetchone()
        if not row:
            return None

        updates = []
        params = []
        if data.label is not None:
            updates.append("label=?")
            params.append(data.label)
        if data.is_active is not None:
            updates.append("is_active=?")
            params.append(int(data.is_active))

        if not updates:
            return _row_to_method(row)

        updates.append("updated_at=?")
        params.append(_now_iso())
        params.append(method_id)

        conn.execute(
            f"UPDATE payment_methods SET {', '.join(updates)} WHERE id=?",
            params,
        )
        updated = conn.execute("SELECT * FROM payment_methods WHERE id=?", (method_id,)).fetchone()
    return _row_to_method(updated)


def delete_payment_method(method_id: str) -> bool:
    """결제 수단 비활성화 (소프트 삭제)."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM payment_methods WHERE id=?", (method_id,)).fetchone()
        if not row:
            return False

        # 활성 구독에 연결된 결제 수단은 삭제 불가
        active_sub = conn.execute(
            "SELECT COUNT(*) AS cnt FROM subscriptions WHERE payment_method_id=? AND status='active'",
            (method_id,),
        ).fetchone()
        if active_sub["cnt"] > 0:
            raise ValueError("Cannot delete payment method with active subscription")

        conn.execute(
            "UPDATE payment_methods SET is_active=0, is_default=0, updated_at=? WHERE id=?",
            (_now_iso(), method_id),
        )
    return True


def set_default_payment_method(method_id: str) -> Optional[PaymentMethodResponse]:
    """기본 결제 수단 설정."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM payment_methods WHERE id=? AND is_active=1",
            (method_id,),
        ).fetchone()
        if not row:
            return None

        user_id = row["user_id"]
        now = _now_iso()
        # 기존 기본 해제
        conn.execute(
            "UPDATE payment_methods SET is_default=0, updated_at=? WHERE user_id=? AND is_default=1",
            (now, user_id),
        )
        # 새 기본 설정
        conn.execute(
            "UPDATE payment_methods SET is_default=1, updated_at=? WHERE id=?",
            (now, method_id),
        )
        updated = conn.execute("SELECT * FROM payment_methods WHERE id=?", (method_id,)).fetchone()
    return _row_to_method(updated)


def get_plans() -> list[PlanResponse]:
    """활성 구독 플랜 목록."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM subscription_plans WHERE is_active=1 ORDER BY price_krw",
        ).fetchall()
    return [
        PlanResponse(
            id=r["id"], name=r["name"], name_kr=r["name_kr"],
            price_krw=r["price_krw"], interval=r["interval"],
            max_analyses_per_month=r["max_analyses_per_month"],
            features=r["features"].split(","),
        )
        for r in rows
    ]


def create_subscription(req: SubscribeRequest) -> SubscriptionResponse:
    """구독 생성."""
    with get_db() as conn:
        # 플랜 검증
        plan = conn.execute("SELECT * FROM subscription_plans WHERE id=? AND is_active=1", (req.plan_id,)).fetchone()
        if not plan:
            raise ValueError(f"Plan not found: {req.plan_id}")

        # 결제 수단 검증
        pm = conn.execute(
            "SELECT * FROM payment_methods WHERE id=? AND user_id=? AND is_active=1",
            (req.payment_method_id, req.user_id),
        ).fetchone()
        if not pm:
            raise ValueError("Payment method not found or inactive")

        # 기존 활성 구독 취소
        conn.execute(
            "UPDATE subscriptions SET status='cancelled', cancelled_at=? WHERE user_id=? AND status='active'",
            (_now_iso(), req.user_id),
        )

        sub_id = f"sub_{uuid.uuid4().hex[:16]}"
        now = _now_iso()
        # 간이 기간 계산 (실제 서비스에서는 dateutil 사용)
        period_end = datetime.now(timezone.utc).replace(
            month=datetime.now(timezone.utc).month % 12 + 1,
        ).isoformat()

        conn.execute(
            """INSERT INTO subscriptions
               (id, user_id, plan_id, payment_method_id, status,
                current_period_start, current_period_end, created_at)
               VALUES (?, ?, ?, ?, 'active', ?, ?, ?)""",
            (sub_id, req.user_id, req.plan_id, req.payment_method_id, now, period_end, now),
        )

        # 결제 이력
        conn.execute(
            """INSERT INTO payment_history
               (id, user_id, subscription_id, payment_method_id, amount_krw, status, description, created_at)
               VALUES (?, ?, ?, ?, ?, 'success', ?, ?)""",
            (f"pay_{uuid.uuid4().hex[:16]}", req.user_id, sub_id,
             req.payment_method_id, plan["price_krw"],
             f"{plan['name_kr']} 구독", now),
        )

        row = conn.execute("SELECT * FROM subscriptions WHERE id=?", (sub_id,)).fetchone()

    return SubscriptionResponse(
        id=row["id"], user_id=row["user_id"], plan_id=row["plan_id"],
        payment_method_id=row["payment_method_id"], status=row["status"],
        current_period_start=row["current_period_start"],
        current_period_end=row["current_period_end"],
        created_at=row["created_at"],
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/payments", tags=["payments"])


@router.on_event("startup")
async def _startup():
    init_db()


@router.post("/methods", response_model=PaymentMethodResponse, status_code=201)
async def api_create_method(data: PaymentMethodCreate):
    """결제 수단 등록."""
    return create_payment_method(data)


@router.get("/methods", response_model=list[PaymentMethodResponse])
async def api_list_methods(user_id: str):
    """결제 수단 목록 조회."""
    return get_payment_methods(user_id)


@router.get("/methods/{method_id}", response_model=PaymentMethodResponse)
async def api_get_method(method_id: str):
    """결제 수단 상세 조회."""
    result = get_payment_method(method_id)
    if not result:
        raise HTTPException(status_code=404, detail="Payment method not found")
    return result


@router.put("/methods/{method_id}", response_model=PaymentMethodResponse)
async def api_update_method(method_id: str, data: PaymentMethodUpdate):
    """결제 수단 수정."""
    result = update_payment_method(method_id, data)
    if not result:
        raise HTTPException(status_code=404, detail="Payment method not found")
    return result


@router.delete("/methods/{method_id}", status_code=204)
async def api_delete_method(method_id: str):
    """결제 수단 삭제 (비활성화)."""
    try:
        deleted = delete_payment_method(method_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail="Payment method not found")


@router.post("/methods/{method_id}/default", response_model=PaymentMethodResponse)
async def api_set_default(method_id: str):
    """기본 결제 수단 설정."""
    result = set_default_payment_method(method_id)
    if not result:
        raise HTTPException(status_code=404, detail="Payment method not found or inactive")
    return result


@router.get("/plans", response_model=list[PlanResponse])
async def api_list_plans():
    """구독 플랜 목록."""
    return get_plans()


@router.post("/subscribe", response_model=SubscriptionResponse, status_code=201)
async def api_subscribe(req: SubscribeRequest):
    """구독 신청."""
    try:
        return create_subscription(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
