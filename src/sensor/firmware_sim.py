"""BLE 펌웨어 시뮬레이터.

실제 하드웨어(nRF52840 신발 센서) 없이 BLE 패킷 스트림을 생성한다.
테스트·개발 환경에서 bleak 수신 코드를 그대로 검증할 수 있다.

사용 예:
    sim = FirmwareSimulator(gait_class=1, session_id=0x0042)
    for pkt_bytes in sim.stream():
        pkt = BLEPacket.from_bytes(pkt_bytes)
        ...  # 수신 처리
"""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from .ble_protocol import (
    BLEPacket, PacketType,
    encode_imu, encode_pressure, encode_skeleton,
    session_id_new,
)

# ── 하드웨어 스펙 ──────────────────────────────────────────────────────────────
SENSOR_SPECS = {
    "imu": {
        "model":       "ICM-42688-P",
        "accel_range": "±8 g",
        "gyro_range":  "±1000 dps",
        "sample_rate": 128,       # Hz
        "resolution":  16,        # bits
    },
    "pressure": {
        "model":       "Pedar-X FSR array",
        "grid":        [16, 8],
        "cells":       128,
        "sample_rate": 100,       # Hz
        "resolution":  12,        # bits
    },
    "ble": {
        "soc":         "nRF52840",
        "version":     "BT 5.0",
        "mtu":         244,
        "tx_power":    "+4 dBm",
    },
}

# 보행 클래스별 생성 파라미터
_GAIT_PARAMS: dict[str | int, dict] = {
    0: {"freq": 1.8, "noise": 0.10, "amp": 1.0, "label": "normal"},
    1: {"freq": 1.2, "noise": 0.15, "amp": 0.5, "label": "parkinsons"},   # 느린 보행, 진전
    2: {"freq": 1.5, "noise": 0.20, "amp": 0.9, "label": "stroke"},       # 비대칭
    3: {"freq": 1.4, "noise": 0.35, "amp": 0.8, "label": "fall_risk"},    # 불규칙
}
# 문자열 alias
_GAIT_ALIAS: dict[str, int] = {
    "normal":     0,
    "parkinsons": 1,
    "stroke":     2,
    "fall_risk":  3,
    "antalgic":   2,
    "ataxic":     3,
}


class FirmwareSimulator:
    """단일 보행 세션의 BLE 패킷 스트림을 생성하는 시뮬레이터.

    Args:
        gait_class : int(0-3) 또는 str 보행 프로파일
        session_id : BLE session_id (None이면 무작위)
        seq_len    : IMU 시퀀스 길이 (프레임)
        seed       : 재현성용 numpy random seed
    """

    def __init__(
        self,
        gait_class: int | str = 0,
        session_id: int | None = None,
        seq_len: int = 128,
        seed: int = 42,
    ):
        if isinstance(gait_class, str):
            gait_class = _GAIT_ALIAS.get(gait_class, 0)
        self.gait_class = int(gait_class)
        self.session_id = session_id if session_id is not None else session_id_new()
        self.seq_len    = seq_len
        self._rng       = np.random.default_rng(seed)
        self._params    = _GAIT_PARAMS.get(self.gait_class, _GAIT_PARAMS[0])

    # ── 센서 데이터 생성 ──────────────────────────────────────────────────────

    def generate_imu(self) -> list[list[float]]:
        """[seq_len, 6] — (ax,ay,az,gx,gy,gz) m/s² / rad/s."""
        T  = self.seq_len
        p  = self._params
        t  = np.linspace(0, T / 128, T)
        f  = p["freq"]
        a  = p["amp"]
        n  = p["noise"]

        ax = a * 0.3 * np.sin(2 * np.pi * f * t + 0.2)
        ay = a * 1.0 * np.sin(2 * np.pi * f * t)        # 주 수직 성분
        az = a * 0.2 * np.cos(2 * np.pi * f * t)

        # 파킨슨: 4–6 Hz 진전 추가
        if self.gait_class == 1:
            tremor = 0.4 * np.sin(2 * np.pi * 5 * t)
            ax += tremor
            ay += tremor * 0.5

        gx = a * 0.5 * np.cos(2 * np.pi * f * t)
        gy = a * 0.1 * np.ones(T)
        gz = a * 0.3 * np.sin(2 * np.pi * f * t + np.pi / 3)

        noise = self._rng.normal(0, n, (T, 6))
        imu   = np.stack([ax, ay, az, gx, gy, gz], axis=1) + noise
        return imu.tolist()

    def generate_pressure(self) -> list[list[float]]:
        """[16, 8] — 0–1 정규화 족저압 그리드."""
        H, W = 16, 8
        grid = self._rng.uniform(0.05, 0.4, (H, W))

        # 뒤꿈치 강조
        grid[11:, :] += 0.2
        # 전족부 강조
        grid[2:6, :] += 0.15

        # 뇌졸중/낙상: 한쪽 편중
        if self.gait_class in (2, 3):
            grid[:, :4] *= 0.5

        grid = np.clip(grid / grid.max(), 0.0, 1.0)
        return grid.tolist()

    def generate_skeleton(self) -> list[list[list[float]]]:
        """[seq_len, 17, 3] — COCO 17관절 3D 좌표."""
        T = self.seq_len
        p = self._params
        t = np.linspace(0, T / 128, T)
        f = p["freq"]

        # COCO 기준 정적 자세 (발뒤꿈치 원점)
        base = np.array([
            [0.0,  1.0,  0.0],   # 0  nose
            [0.1,  1.0,  0.0],   # 1  left_eye
            [-0.1, 1.0,  0.0],   # 2  right_eye
            [0.2,  0.95, 0.0],   # 3  left_ear
            [-0.2, 0.95, 0.0],   # 4  right_ear
            [0.2,  0.75, 0.0],   # 5  left_shoulder
            [-0.2, 0.75, 0.0],   # 6  right_shoulder
            [0.35, 0.45, 0.0],   # 7  left_elbow
            [-0.35,0.45, 0.0],   # 8  right_elbow
            [0.4,  0.2,  0.0],   # 9  left_wrist
            [-0.4, 0.2,  0.0],   # 10 right_wrist
            [0.1,  0.55, 0.0],   # 11 left_hip
            [-0.1, 0.55, 0.0],   # 12 right_hip
            [0.15, 0.3,  0.0],   # 13 left_knee
            [-0.15,0.3,  0.0],   # 14 right_knee
            [0.15, 0.05, 0.0],   # 15 left_ankle
            [-0.15,0.05, 0.0],   # 16 right_ankle
        ], dtype=np.float64)

        # 보행 애니메이션: 좌우 교대 레그 모션
        skel = np.tile(base, (T, 1, 1))
        swing = 0.12 * np.sin(2 * np.pi * f * t)
        skel[:, 13, 2] += swing        # left_knee z
        skel[:, 14, 2] -= swing        # right_knee z (반대)
        skel[:, 15, 2] += swing * 1.5  # left_ankle
        skel[:, 16, 2] -= swing * 1.5

        # 몸통 흔들림
        trunk_sway = p["noise"] * 0.5 * np.sin(2 * np.pi * f * t)
        skel[:, 5, 0] += trunk_sway   # left_shoulder x
        skel[:, 6, 0] += trunk_sway

        noise = self._rng.normal(0, p["noise"] * 0.02, skel.shape)
        return (skel + noise).tolist()

    # ── 패킷 스트림 ───────────────────────────────────────────────────────────

    def stream(self, delay_ms: float = 0.0) -> Iterator[bytes]:
        """전체 세션 데이터를 BLE 청크 패킷으로 순서대로 반환.

        Args:
            delay_ms : 패킷 간 지연 (ms). 0이면 즉시 반환.
        """
        imu      = self.generate_imu()
        pressure = self.generate_pressure()
        skeleton = self.generate_skeleton()

        packets  = (
            encode_imu(imu, self.session_id)
            + encode_pressure(pressure, self.session_id)
            + encode_skeleton(skeleton, self.session_id)
        )
        packets.append(
            BLEPacket(PacketType.END, self.session_id, 0, 1, b"").to_bytes()
        )

        for pkt in packets:
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
            yield pkt

    def snapshot(self) -> dict:
        """단번에 모든 센서 데이터를 dict로 반환 (BLE 없이 직접 사용)."""
        return {
            "session_id":  self.session_id,
            "gait_class":  self.gait_class,
            "gait_label":  self._params["label"],
            "imu":         self.generate_imu(),
            "pressure":    self.generate_pressure(),
            "skeleton":    self.generate_skeleton(),
        }
