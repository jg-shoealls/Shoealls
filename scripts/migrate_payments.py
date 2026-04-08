"""결제 데이터베이스 마이그레이션 스크립트.

Usage:
    python scripts/migrate_payments.py                    # 마이그레이션 실행
    python scripts/migrate_payments.py --status           # 현재 버전 확인
    python scripts/migrate_payments.py --rollback 1       # 버전 1로 롤백
    python scripts/migrate_payments.py --db data/test.db  # DB 경로 지정
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# 마이그레이션 정의 (순서대로 적용)
MIGRATIONS = {
    1: {
        "description": "결제 수단, 구독 플랜, 구독, 결제 이력 테이블 생성",
        "up": """
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
        """,
        "down": """
DROP TABLE IF EXISTS payment_history;
DROP TABLE IF EXISTS subscriptions;
DROP TABLE IF EXISTS subscription_plans;
DROP TABLE IF EXISTS payment_methods;
DROP TABLE IF EXISTS schema_version;
        """,
    },
    2: {
        "description": "기본 구독 플랜 시드 데이터 삽입",
        "up": """
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
        """,
        "down": """
DELETE FROM subscription_plans WHERE id IN ('plan_free', 'plan_basic', 'plan_pro', 'plan_enterprise');
DELETE FROM schema_version WHERE version = 2;
        """,
    },
    3: {
        "description": "결제 수단에 만료일, 결제 이력에 영수증 URL 컬럼 추가",
        "up": """
ALTER TABLE payment_methods ADD COLUMN expires_at TEXT;
ALTER TABLE payment_history ADD COLUMN receipt_url TEXT;

INSERT OR IGNORE INTO schema_version (version, applied_at)
    VALUES (3, datetime('now'));
        """,
        "down": """
-- SQLite는 ALTER TABLE DROP COLUMN을 3.35.0+에서만 지원
-- 안전하게 무시 (컬럼은 남아있지만 사용하지 않음)
DELETE FROM schema_version WHERE version = 3;
        """,
    },
}


def get_current_version(conn: sqlite3.Connection) -> int:
    """현재 스키마 버전 조회."""
    try:
        row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        return row[0] if row and row[0] else 0
    except sqlite3.OperationalError:
        return 0


def migrate(db_path: str, target_version: int | None = None) -> int:
    """마이그레이션 실행.

    Args:
        db_path: SQLite DB 파일 경로.
        target_version: 목표 버전. None이면 최신까지 적용.

    Returns:
        적용된 마이그레이션 수.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")

    current = get_current_version(conn)
    max_version = max(MIGRATIONS.keys())
    target = target_version if target_version is not None else max_version

    if target < 0 or target > max_version:
        conn.close()
        raise ValueError(f"Invalid target version: {target} (max: {max_version})")

    applied = 0

    if target > current:
        # Forward migration
        for v in range(current + 1, target + 1):
            if v not in MIGRATIONS:
                continue
            print(f"  Applying migration v{v}: {MIGRATIONS[v]['description']}")
            conn.executescript(MIGRATIONS[v]["up"])
            applied += 1

    elif target < current:
        # Rollback
        for v in range(current, target, -1):
            if v not in MIGRATIONS:
                continue
            print(f"  Rolling back migration v{v}: {MIGRATIONS[v]['description']}")
            conn.executescript(MIGRATIONS[v]["down"])
            applied += 1

    conn.close()
    return applied


def show_status(db_path: str) -> None:
    """현재 마이그레이션 상태 출력."""
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        print(f"Current version: 0 (not initialized)")
        print(f"Latest available: {max(MIGRATIONS.keys())}")
        return

    conn = sqlite3.connect(db_path)
    current = get_current_version(conn)
    max_ver = max(MIGRATIONS.keys())
    print(f"Database: {db_path}")
    print(f"Current version: {current}")
    print(f"Latest available: {max_ver}")

    if current < max_ver:
        print(f"\nPending migrations:")
        for v in range(current + 1, max_ver + 1):
            print(f"  v{v}: {MIGRATIONS[v]['description']}")
    else:
        print("\nAll migrations applied.")

    # 적용 이력
    try:
        rows = conn.execute("SELECT * FROM schema_version ORDER BY version").fetchall()
        if rows:
            print("\nApplied versions:")
            for r in rows:
                print(f"  v{r[0]}: applied at {r[1]}")
    except sqlite3.OperationalError:
        pass

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="결제 DB 마이그레이션")
    parser.add_argument("--db", type=str, default="data/payments.db", help="DB 파일 경로")
    parser.add_argument("--status", action="store_true", help="현재 상태 확인")
    parser.add_argument("--rollback", type=int, default=None, help="목표 버전으로 롤백")
    parser.add_argument("--target", type=int, default=None, help="목표 버전까지 마이그레이션")
    args = parser.parse_args()

    if args.status:
        show_status(args.db)
        return

    target = args.rollback if args.rollback is not None else args.target
    print(f"Database: {args.db}")
    print(f"Target version: {target or 'latest'}")
    print()

    applied = migrate(args.db, target)

    if applied > 0:
        print(f"\n{applied} migration(s) applied successfully.")
    else:
        print("\nNo migrations to apply.")

    show_status(args.db)


if __name__ == "__main__":
    main()
