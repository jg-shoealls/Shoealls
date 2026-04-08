"""결제 수단 API 및 DB 마이그레이션 테스트."""

import os
import tempfile

import pytest

from src.serving.payment import (
    PaymentMethodCreate,
    PaymentMethodUpdate,
    MethodType,
    SubscribeRequest,
    create_payment_method,
    get_payment_methods,
    get_payment_method,
    update_payment_method,
    delete_payment_method,
    set_default_payment_method,
    get_plans,
    create_subscription,
    init_db,
    set_db_path,
)
from scripts.migrate_payments import migrate, get_current_version, MIGRATIONS

import sqlite3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def tmp_db(tmp_path):
    """각 테스트마다 임시 DB 사용."""
    db_path = str(tmp_path / "test_payments.db")
    set_db_path(db_path)
    init_db()
    yield db_path


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------

class TestMigrations:
    def test_migrate_creates_tables(self, tmp_path):
        db_path = str(tmp_path / "migrate_test.db")
        applied = migrate(db_path, target_version=1)
        assert applied == 1

        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()

        assert "payment_methods" in tables
        assert "subscription_plans" in tables
        assert "subscriptions" in tables
        assert "payment_history" in tables
        assert "schema_version" in tables

    def test_migrate_to_latest(self, tmp_path):
        db_path = str(tmp_path / "migrate_latest.db")
        max_ver = max(MIGRATIONS.keys())
        applied = migrate(db_path)
        assert applied == max_ver

        conn = sqlite3.connect(db_path)
        version = get_current_version(conn)
        conn.close()
        assert version == max_ver

    def test_migrate_seed_plans(self, tmp_path):
        db_path = str(tmp_path / "migrate_plans.db")
        migrate(db_path, target_version=2)

        conn = sqlite3.connect(db_path)
        plans = conn.execute("SELECT COUNT(*) FROM subscription_plans").fetchone()[0]
        conn.close()
        assert plans == 4

    def test_migrate_v3_adds_columns(self, tmp_path):
        db_path = str(tmp_path / "migrate_v3.db")
        migrate(db_path, target_version=3)

        conn = sqlite3.connect(db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(payment_methods)").fetchall()]
        hist_cols = [r[1] for r in conn.execute("PRAGMA table_info(payment_history)").fetchall()]
        conn.close()

        assert "expires_at" in cols
        assert "receipt_url" in hist_cols

    def test_migrate_idempotent(self, tmp_path):
        db_path = str(tmp_path / "migrate_idem.db")
        migrate(db_path)
        applied = migrate(db_path)
        assert applied == 0

    def test_rollback(self, tmp_path):
        db_path = str(tmp_path / "migrate_rollback.db")
        migrate(db_path, target_version=2)

        conn = sqlite3.connect(db_path)
        assert get_current_version(conn) == 2
        conn.close()

        applied = migrate(db_path, target_version=1)
        assert applied == 1

    def test_invalid_target_raises(self, tmp_path):
        db_path = str(tmp_path / "migrate_bad.db")
        with pytest.raises(ValueError, match="Invalid target version"):
            migrate(db_path, target_version=999)


# ---------------------------------------------------------------------------
# Payment method CRUD tests
# ---------------------------------------------------------------------------

class TestPaymentMethodCRUD:
    def _create_card(self, user_id: str = "user_001", label: str = "내 신한카드") -> str:
        result = create_payment_method(PaymentMethodCreate(
            user_id=user_id,
            method_type=MethodType.CARD,
            label=label,
            card_last4="1234",
            card_brand="Shinhan",
        ))
        return result.id

    def test_create_card(self):
        result = create_payment_method(PaymentMethodCreate(
            user_id="user_001",
            method_type=MethodType.CARD,
            label="내 신한카드",
            card_last4="1234",
            card_brand="Shinhan",
        ))
        assert result.id.startswith("pm_")
        assert result.user_id == "user_001"
        assert result.method_type == "card"
        assert result.card_last4 == "1234"
        assert result.card_brand == "Shinhan"
        assert result.is_active is True

    def test_create_kakao_pay(self):
        result = create_payment_method(PaymentMethodCreate(
            user_id="user_001",
            method_type=MethodType.KAKAO_PAY,
            label="카카오페이",
        ))
        assert result.method_type == "kakao_pay"
        assert result.card_last4 is None

    def test_create_bank_transfer(self):
        result = create_payment_method(PaymentMethodCreate(
            user_id="user_001",
            method_type=MethodType.BANK_TRANSFER,
            label="국민은행",
            bank_name="KB국민은행",
            account_last4="5678",
        ))
        assert result.method_type == "bank_transfer"
        assert result.bank_name == "KB국민은행"
        assert result.account_last4 == "5678"

    def test_first_method_is_default(self):
        result = create_payment_method(PaymentMethodCreate(
            user_id="user_new",
            method_type=MethodType.CARD,
            label="첫 카드",
            card_last4="1111",
        ))
        assert result.is_default is True

    def test_second_method_not_default(self):
        self._create_card("user_002", "카드1")
        result = create_payment_method(PaymentMethodCreate(
            user_id="user_002",
            method_type=MethodType.CARD,
            label="카드2",
            card_last4="5678",
        ))
        assert result.is_default is False

    def test_list_methods(self):
        self._create_card("user_003", "카드A")
        self._create_card("user_003", "카드B")
        self._create_card("user_other", "다른사용자")

        methods = get_payment_methods("user_003")
        assert len(methods) == 2
        assert all(m.user_id == "user_003" for m in methods)

    def test_get_method(self):
        method_id = self._create_card()
        result = get_payment_method(method_id)
        assert result is not None
        assert result.id == method_id

    def test_get_nonexistent_method(self):
        result = get_payment_method("pm_nonexistent")
        assert result is None

    def test_update_label(self):
        method_id = self._create_card()
        result = update_payment_method(method_id, PaymentMethodUpdate(label="새 이름"))
        assert result is not None
        assert result.label == "새 이름"

    def test_update_nonexistent(self):
        result = update_payment_method("pm_nonexistent", PaymentMethodUpdate(label="x"))
        assert result is None

    def test_delete_method(self):
        method_id = self._create_card()
        deleted = delete_payment_method(method_id)
        assert deleted is True

        result = get_payment_method(method_id)
        assert result.is_active is False

    def test_delete_hides_from_list(self):
        method_id = self._create_card("user_del")
        delete_payment_method(method_id)
        methods = get_payment_methods("user_del")
        assert len(methods) == 0

    def test_delete_nonexistent(self):
        deleted = delete_payment_method("pm_nonexistent")
        assert deleted is False

    def test_set_default(self):
        id1 = self._create_card("user_def", "카드1")
        id2 = self._create_card("user_def", "카드2")

        # id1이 기본
        assert get_payment_method(id1).is_default is True
        assert get_payment_method(id2).is_default is False

        # id2를 기본으로 변경
        result = set_default_payment_method(id2)
        assert result.is_default is True
        assert get_payment_method(id1).is_default is False

    def test_set_default_nonexistent(self):
        result = set_default_payment_method("pm_nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Subscription plan tests
# ---------------------------------------------------------------------------

class TestPlans:
    def test_list_plans(self):
        plans = get_plans()
        assert len(plans) == 4

    def test_plan_fields(self):
        plans = get_plans()
        free = next(p for p in plans if p.id == "plan_free")
        assert free.name == "Free"
        assert free.name_kr == "무료"
        assert free.price_krw == 0
        assert free.max_analyses_per_month == 5
        assert isinstance(free.features, list)
        assert len(free.features) > 0

    def test_plans_ordered_by_price(self):
        plans = get_plans()
        prices = [p.price_krw for p in plans]
        assert prices == sorted(prices)


# ---------------------------------------------------------------------------
# Subscription tests
# ---------------------------------------------------------------------------

class TestSubscription:
    def _setup_user(self, user_id: str = "user_sub") -> str:
        result = create_payment_method(PaymentMethodCreate(
            user_id=user_id,
            method_type=MethodType.CARD,
            label="테스트카드",
            card_last4="9999",
        ))
        return result.id

    def test_subscribe(self):
        pm_id = self._setup_user()
        sub = create_subscription(SubscribeRequest(
            user_id="user_sub",
            plan_id="plan_basic",
            payment_method_id=pm_id,
        ))
        assert sub.id.startswith("sub_")
        assert sub.status == "active"
        assert sub.plan_id == "plan_basic"

    def test_subscribe_invalid_plan(self):
        pm_id = self._setup_user("user_bad_plan")
        with pytest.raises(ValueError, match="Plan not found"):
            create_subscription(SubscribeRequest(
                user_id="user_bad_plan",
                plan_id="plan_nonexistent",
                payment_method_id=pm_id,
            ))

    def test_subscribe_invalid_payment_method(self):
        with pytest.raises(ValueError, match="Payment method not found"):
            create_subscription(SubscribeRequest(
                user_id="user_no_pm",
                plan_id="plan_free",
                payment_method_id="pm_nonexistent",
            ))

    def test_resubscribe_cancels_previous(self):
        pm_id = self._setup_user("user_resub")
        sub1 = create_subscription(SubscribeRequest(
            user_id="user_resub",
            plan_id="plan_basic",
            payment_method_id=pm_id,
        ))
        sub2 = create_subscription(SubscribeRequest(
            user_id="user_resub",
            plan_id="plan_pro",
            payment_method_id=pm_id,
        ))
        assert sub2.status == "active"
        assert sub1.id != sub2.id

    def test_cannot_delete_method_with_active_sub(self):
        pm_id = self._setup_user("user_nodelete")
        create_subscription(SubscribeRequest(
            user_id="user_nodelete",
            plan_id="plan_basic",
            payment_method_id=pm_id,
        ))
        with pytest.raises(ValueError, match="Cannot delete.*active subscription"):
            delete_payment_method(pm_id)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_card_requires_last4(self):
        with pytest.raises(Exception):
            PaymentMethodCreate(
                user_id="user_v",
                method_type=MethodType.CARD,
                label="카드",
                card_last4=None,
            )

    def test_invalid_card_last4_format(self):
        with pytest.raises(Exception):
            PaymentMethodCreate(
                user_id="user_v",
                method_type=MethodType.CARD,
                label="카드",
                card_last4="abc",
            )

    def test_empty_user_id_rejected(self):
        with pytest.raises(Exception):
            PaymentMethodCreate(
                user_id="",
                method_type=MethodType.CARD,
                label="카드",
                card_last4="1234",
            )

    def test_empty_label_rejected(self):
        with pytest.raises(Exception):
            PaymentMethodCreate(
                user_id="user_v",
                method_type=MethodType.CARD,
                label="",
                card_last4="1234",
            )
