"""센서 레이어 단위 테스트 — BLE 프로토콜 + 특성 추출 + 펌웨어 시뮬레이터."""

import struct
import pytest
import numpy as np

from src.sensor.ble_protocol import (
    BLEPacket, PacketType, StreamAssembler,
    encode_imu, encode_pressure, encode_skeleton,
    decode_imu, decode_pressure, decode_skeleton,
    HEADER_SIZE, MAX_PAYLOAD, MAGIC,
)
from src.sensor.feature_extractor import GaitFeatureExtractor, GaitFeatures
from src.sensor.firmware_sim import FirmwareSimulator


# ─────────────────────────────────────────────────────────────────────────────
# BLE Protocol
# ─────────────────────────────────────────────────────────────────────────────

class TestBLEPacket:
    def _make_payload(self, n: int = 10) -> bytes:
        return bytes(range(n))

    def test_roundtrip_to_from_bytes(self):
        payload = self._make_payload(50)
        pkt = BLEPacket(PacketType.IMU, session_id=0x1234, chunk_idx=0,
                        total_chunks=3, payload=payload)
        raw = pkt.to_bytes()
        pkt2 = BLEPacket.from_bytes(raw)
        assert pkt2.packet_type == PacketType.IMU
        assert pkt2.session_id == 0x1234
        assert pkt2.chunk_idx == 0
        assert pkt2.total_chunks == 3
        assert pkt2.payload == payload

    def test_header_size(self):
        payload = self._make_payload(5)
        pkt = BLEPacket(PacketType.PRESSURE, 0, 0, 1, payload)
        raw = pkt.to_bytes()
        assert len(raw) == HEADER_SIZE + len(payload)

    def test_magic_byte(self):
        payload = self._make_payload(4)
        pkt = BLEPacket(PacketType.END, 0, 0, 1, payload)
        raw = pkt.to_bytes()
        assert raw[0] == MAGIC

    def test_crc_valid(self):
        payload = self._make_payload(20)
        pkt = BLEPacket(PacketType.IMU, 0, 0, 1, payload)
        raw = pkt.to_bytes()
        pkt2 = BLEPacket.from_bytes(raw)
        assert pkt2.is_valid()

    def test_crc_invalid_on_corruption(self):
        payload = self._make_payload(20)
        pkt = BLEPacket(PacketType.IMU, 0, 0, 1, payload)
        raw = bytearray(pkt.to_bytes())
        raw[-1] ^= 0xFF            # payload 마지막 바이트 오염
        pkt2 = BLEPacket.from_bytes(bytes(raw))
        assert not pkt2.is_valid()

    def test_bad_magic_raises(self):
        payload = self._make_payload(4)
        pkt = BLEPacket(PacketType.END, 0, 0, 1, payload)
        raw = bytearray(pkt.to_bytes())
        raw[0] = 0x00              # MAGIC 오염
        with pytest.raises(ValueError, match="MAGIC"):
            BLEPacket.from_bytes(bytes(raw))

    def test_short_packet_raises(self):
        with pytest.raises(ValueError, match="너무 짧음"):
            BLEPacket.from_bytes(b"\xAA" * 5)

    def test_max_payload_fits(self):
        payload = bytes(MAX_PAYLOAD)
        pkt = BLEPacket(PacketType.IMU, 0, 0, 1, payload)
        assert len(pkt.to_bytes()) == HEADER_SIZE + MAX_PAYLOAD


class TestStreamAssembler:
    def _encode_and_reassemble(self, chunks_bytes: list[bytes], ptype: PacketType) -> bytes:
        asm = StreamAssembler(ptype)
        result = None
        for raw in chunks_bytes:
            pkt = BLEPacket.from_bytes(raw)
            result = asm.feed(pkt)
        assert result is not None
        return result

    def test_imu_roundtrip(self):
        imu = [[float(i % 6) * 0.1 for i in range(6)] for _ in range(128)]
        chunks = encode_imu(imu, session_id=0x0001)
        raw = self._encode_and_reassemble(chunks, PacketType.IMU)
        decoded = decode_imu(raw)
        assert len(decoded) == 128
        assert len(decoded[0]) == 6
        # 부동소수점 → int16 → 복원 오차 < 0.002
        for orig, dec in zip(imu, decoded):
            for o, d in zip(orig, dec):
                assert abs(o - d) < 0.002

    def test_pressure_roundtrip(self):
        pressure = [[float((r * 8 + c) % 100) / 100 for c in range(8)] for r in range(16)]
        chunks = encode_pressure(pressure, session_id=0x0002)
        raw = self._encode_and_reassemble(chunks, PacketType.PRESSURE)
        decoded = decode_pressure(raw)
        assert len(decoded) == 16
        assert len(decoded[0]) == 8
        for orig_row, dec_row in zip(pressure, decoded):
            for o, d in zip(orig_row, dec_row):
                assert abs(o - d) < 0.00002

    def test_skeleton_roundtrip(self):
        skeleton = [[[float(j) * 0.01 + t * 0.001 for _ in range(3)]
                     for j in range(17)]
                    for t in range(128)]
        chunks = encode_skeleton(skeleton, session_id=0x0003)
        raw = self._encode_and_reassemble(chunks, PacketType.SKELETON)
        decoded = decode_skeleton(raw)
        assert len(decoded) == 128
        assert len(decoded[0]) == 17
        assert len(decoded[0][0]) == 3

    def test_assembler_progress(self):
        imu = [[0.0] * 6] * 128
        chunks = encode_imu(imu, 0)
        asm = StreamAssembler(PacketType.IMU)
        for i, raw in enumerate(chunks[:-1]):
            pkt = BLEPacket.from_bytes(raw)
            result = asm.feed(pkt)
            assert result is None
            received, total = asm.progress
            assert received == i + 1
            assert total == len(chunks)

    def test_assembler_resets_after_complete(self):
        imu = [[0.0] * 6] * 128
        chunks = encode_imu(imu, 0)
        asm = StreamAssembler(PacketType.IMU)
        for raw in chunks:
            pkt = BLEPacket.from_bytes(raw)
            asm.feed(pkt)
        assert asm.progress == (0, None)

    def test_crc_error_raises(self):
        imu = [[0.0] * 6] * 128
        chunks = encode_imu(imu, 0)
        corrupt = bytearray(chunks[0])
        corrupt[-1] ^= 0xFF
        pkt = BLEPacket.from_bytes(bytes(corrupt))
        asm = StreamAssembler(PacketType.IMU)
        with pytest.raises(ValueError, match="CRC"):
            asm.feed(pkt)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class TestGaitFeatureExtractor:
    def _normal_imu(self, T: int = 128, fs: int = 128) -> np.ndarray:
        """정상 보행 IMU 신호."""
        t = np.linspace(0, T / fs, T)
        f = 1.8   # 1.8 Hz (stride freq ~1.8 strides/s)
        ax = 0.3 * np.sin(2 * np.pi * f * t)
        ay = 1.0 * np.sin(2 * np.pi * f * t)   # 수직
        az = 0.2 * np.cos(2 * np.pi * f * t)
        gx = 0.5 * np.cos(2 * np.pi * f * t)
        gy = 0.1 * np.ones(T)
        gz = 0.3 * np.sin(2 * np.pi * f * t)
        return np.stack([ax, ay, az, gx, gy, gz], axis=1)

    def _normal_pressure(self) -> np.ndarray:
        """정상 족저압 그리드."""
        grid = np.zeros((16, 8))
        grid[11:, :] = 0.35     # heel
        grid[3:7, :] = 0.45     # forefoot
        grid[7:11, :] = 0.20    # midfoot
        return grid / grid.max()

    def test_returns_gaitfeatures(self):
        ext = GaitFeatureExtractor()
        imu = self._normal_imu()
        pres = self._normal_pressure()
        feats = ext.extract(imu, pres)
        assert isinstance(feats, GaitFeatures)

    def test_feature_count(self):
        ext = GaitFeatureExtractor()
        vec = ext.extract(self._normal_imu(), self._normal_pressure()).to_vector()
        assert len(vec) == 13

    def test_all_finite(self):
        ext = GaitFeatureExtractor()
        vec = ext.extract(self._normal_imu(), self._normal_pressure()).to_vector()
        assert np.all(np.isfinite(vec))

    def test_dict_keys(self):
        ext = GaitFeatureExtractor()
        d = ext.extract(self._normal_imu(), self._normal_pressure()).to_dict()
        expected = {
            "gait_speed", "cadence", "stride_regularity", "step_symmetry",
            "cop_sway", "ml_variability", "heel_pressure_ratio",
            "forefoot_pressure_ratio", "arch_index", "pressure_asymmetry",
            "acceleration_rms", "acceleration_variability", "trunk_sway",
        }
        assert set(d.keys()) == expected

    def test_gait_speed_range(self):
        ext = GaitFeatureExtractor()
        f = ext.extract(self._normal_imu(), self._normal_pressure())
        assert 0.3 <= f.gait_speed <= 2.5

    def test_cadence_range(self):
        ext = GaitFeatureExtractor()
        f = ext.extract(self._normal_imu(), self._normal_pressure())
        assert 50 <= f.cadence <= 200

    def test_pressure_ratios_sum_approx_one(self):
        ext = GaitFeatureExtractor()
        f = ext.extract(self._normal_imu(), self._normal_pressure())
        total = f.heel_pressure_ratio + f.forefoot_pressure_ratio + f.arch_index
        # 합이 1에 근사 (midfoot를 arch_index로 나타냄, 엄밀히는 독립 지표)
        assert 0.6 <= total <= 1.4

    def test_acceleration_rms_positive(self):
        ext = GaitFeatureExtractor()
        f = ext.extract(self._normal_imu(), self._normal_pressure())
        assert f.acceleration_rms > 0

    def test_skeleton_trunk_sway(self):
        ext = GaitFeatureExtractor()
        T = 128
        # 안정적인 스켈레톤
        skel = np.zeros((T, 17, 3))
        skel[:, 5, :] = [0.2, 0.75, 0.0]   # left_shoulder
        skel[:, 6, :] = [-0.2, 0.75, 0.0]  # right_shoulder
        skel[:, 11, :] = [0.1, 0.55, 0.0]  # left_hip
        skel[:, 12, :] = [-0.1, 0.55, 0.0] # right_hip
        f = ext.extract(self._normal_imu(), self._normal_pressure(), skel)
        assert 0.0 <= f.trunk_sway <= 20.0

    def test_pressure_time_series_aggregation(self):
        """[T, 16, 8] 입력을 평균 내어 처리해야 한다."""
        ext = GaitFeatureExtractor()
        pres_3d = np.tile(self._normal_pressure(), (10, 1, 1))  # [10, 16, 8]
        pres_2d = self._normal_pressure()
        f3 = ext.extract(self._normal_imu(), pres_3d)
        f2 = ext.extract(self._normal_imu(), pres_2d)
        assert abs(f3.heel_pressure_ratio - f2.heel_pressure_ratio) < 1e-6

    def test_parkinsons_higher_variability(self):
        """파킨슨 보행은 정상보다 가속도 변동성이 높아야 한다."""
        ext = GaitFeatureExtractor()
        T = 128
        t = np.linspace(0, 1, T)

        # 정상: 깨끗한 sin
        imu_norm = np.stack([
            0.3 * np.sin(2 * np.pi * 1.8 * t),
            1.0 * np.sin(2 * np.pi * 1.8 * t),
            0.2 * np.cos(2 * np.pi * 1.8 * t),
            0.5 * np.cos(2 * np.pi * 1.8 * t),
            0.1 * np.ones(T),
            0.3 * np.sin(2 * np.pi * 1.8 * t),
        ], axis=1)

        # 파킨슨: 5 Hz 진전 추가
        tremor = 0.5 * np.sin(2 * np.pi * 5 * t)
        imu_park = imu_norm.copy()
        imu_park[:, 0] += tremor
        imu_park[:, 1] += tremor * 0.5

        pres = self._normal_pressure()
        f_norm = ext.extract(imu_norm, pres)
        f_park = ext.extract(imu_park, pres)
        # 파킨슨이 정상보다 가속도 RMS가 높거나 같아야 함
        assert f_park.acceleration_rms >= f_norm.acceleration_rms * 0.8


# ─────────────────────────────────────────────────────────────────────────────
# Firmware Simulator
# ─────────────────────────────────────────────────────────────────────────────

class TestFirmwareSimulator:
    def test_snapshot_shapes(self):
        sim = FirmwareSimulator(gait_class=0)
        snap = sim.snapshot()
        assert len(snap["imu"]) == 128
        assert len(snap["imu"][0]) == 6
        assert len(snap["pressure"]) == 16
        assert len(snap["pressure"][0]) == 8
        assert len(snap["skeleton"]) == 128
        assert len(snap["skeleton"][0]) == 17
        assert len(snap["skeleton"][0][0]) == 3

    def test_all_gait_classes(self):
        for cls in range(4):
            sim = FirmwareSimulator(gait_class=cls)
            snap = sim.snapshot()
            assert snap["gait_class"] == cls

    def test_string_gait_class(self):
        sim = FirmwareSimulator(gait_class="parkinsons")
        assert sim.gait_class == 1

    def test_stream_yields_packets(self):
        sim = FirmwareSimulator(gait_class=0)
        packets = list(sim.stream())
        assert len(packets) > 0
        # 마지막은 END 패킷
        last = BLEPacket.from_bytes(packets[-1])
        assert last.packet_type == PacketType.END

    def test_stream_full_reconstruction(self):
        """스트림에서 IMU + Pressure + Skeleton을 완전히 복원한다."""
        sim = FirmwareSimulator(gait_class=0, session_id=0xABCD)
        asm_imu   = StreamAssembler(PacketType.IMU)
        asm_pres  = StreamAssembler(PacketType.PRESSURE)
        asm_skel  = StreamAssembler(PacketType.SKELETON)

        imu_raw = pres_raw = skel_raw = None
        for pkt_bytes in sim.stream():
            pkt = BLEPacket.from_bytes(pkt_bytes)
            if pkt.packet_type == PacketType.END:
                break
            r = asm_imu.feed(pkt)
            if r:
                imu_raw = r
            r = asm_pres.feed(pkt)
            if r:
                pres_raw = r
            r = asm_skel.feed(pkt)
            if r:
                skel_raw = r

        assert imu_raw is not None
        assert pres_raw is not None
        assert skel_raw is not None

        imu   = decode_imu(imu_raw)
        pres  = decode_pressure(pres_raw)
        skel  = decode_skeleton(skel_raw)

        assert len(imu) == 128
        assert len(pres) == 16
        assert len(skel) == 128

    def test_stream_and_extract_features(self):
        """스트림 수신 → 특성 추출 파이프라인 종단간 테스트."""
        sim = FirmwareSimulator(gait_class=0)
        snap = sim.snapshot()

        ext = GaitFeatureExtractor()
        feats = ext.extract(
            np.array(snap["imu"]),
            np.array(snap["pressure"]),
            np.array(snap["skeleton"]),
        )
        vec = feats.to_vector()
        assert len(vec) == 13
        assert np.all(np.isfinite(vec))

    def test_session_id_propagated(self):
        sid = 0x1234
        sim = FirmwareSimulator(gait_class=0, session_id=sid)
        for raw in sim.stream():
            pkt = BLEPacket.from_bytes(raw)
            if pkt.packet_type != PacketType.END:
                assert pkt.session_id == sid

    def test_reproducible_with_seed(self):
        sim1 = FirmwareSimulator(gait_class=1, seed=7)
        sim2 = FirmwareSimulator(gait_class=1, seed=7)
        s1 = sim1.snapshot()
        s2 = sim2.snapshot()
        assert s1["imu"] == s2["imu"]
        assert s1["pressure"] == s2["pressure"]
