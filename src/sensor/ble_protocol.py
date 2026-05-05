"""BLE 데이터 전송 프로토콜 — GATT 정의 + 이진 패킷 인코더/디코더.

패킷 구조 (12 byte 헤더):
  [0]    MAGIC   : 0xAA
  [1]    type    : PacketType (uint8)
  [2:4]  session : uint16 LE
  [4:6]  chunk   : uint16 LE  (0-based)
  [6:8]  total   : uint16 LE
  [8:10] plen    : uint16 LE  (payload 길이)
  [10:12] crc    : uint16 LE  (CRC16 of payload)
  [12:]  payload : bytes

BLE MTU 244 → 최대 페이로드 232 bytes/패킷.
IMU  : 1536 bytes → 7 패킷
Pressure: 256 bytes → 2 패킷
Skeleton: 26112 bytes → 113 패킷 (스트리밍)
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum

# ── GATT UUID ─────────────────────────────────────────────────────────────────
SHOEALLS_SERVICE_UUID  = "f000aa00-0451-4000-b000-000000000000"
IMU_CHAR_UUID          = "f000aa01-0451-4000-b000-000000000000"
PRESSURE_CHAR_UUID     = "f000aa02-0451-4000-b000-000000000000"
SKELETON_CHAR_UUID     = "f000aa03-0451-4000-b000-000000000000"
CONTROL_CHAR_UUID      = "f000aa04-0451-4000-b000-000000000000"
STATUS_CHAR_UUID       = "f000aa05-0451-4000-b000-000000000000"

# ── 상수 ──────────────────────────────────────────────────────────────────────
MAGIC          = 0xAA
HEADER_SIZE    = 12
MAX_PAYLOAD    = 232   # 244 MTU - 12 header
IMU_SCALE      = 1000.0   # float → int16  (±32.767 m/s² 또는 rad/s)
PRESSURE_SCALE = 65535.0  # float → uint16 (0–1 → 0–65535)

_HDR_FMT = "<BBHHHHH"  # 12 bytes: MAGIC(B) type(B) session(H) chunk(H) total(H) plen(H) crc(H)


# ── 열거형 ────────────────────────────────────────────────────────────────────

class PacketType(IntEnum):
    IMU      = 0x01
    PRESSURE = 0x02
    SKELETON = 0x03
    FEATURE  = 0x04
    SYNC     = 0xFE
    END      = 0xFF


class ControlCmd(IntEnum):
    START_SESSION   = 0x01
    STOP_SESSION    = 0x02
    RESET           = 0x03
    REQUEST_RESEND  = 0x04
    SET_SAMPLE_RATE = 0x10


# ── 패킷 클래스 ───────────────────────────────────────────────────────────────

@dataclass
class BLEPacket:
    packet_type:  PacketType
    session_id:   int
    chunk_idx:    int
    total_chunks: int
    payload:      bytes
    crc:          int = 0

    def is_valid(self) -> bool:
        return self.crc == _crc16(self.payload)

    def to_bytes(self) -> bytes:
        crc = _crc16(self.payload)
        header = struct.pack(
            _HDR_FMT,
            MAGIC,
            int(self.packet_type),
            self.session_id & 0xFFFF,
            self.chunk_idx & 0xFFFF,
            self.total_chunks & 0xFFFF,
            len(self.payload) & 0xFFFF,
            crc,
        )
        return header + self.payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "BLEPacket":
        if len(data) < HEADER_SIZE:
            raise ValueError(f"패킷 너무 짧음: {len(data)} bytes")
        magic, ptype, sid, chunk, total, plen, crc = struct.unpack_from(_HDR_FMT, data)
        if magic != MAGIC:
            raise ValueError(f"MAGIC 불일치: 0x{magic:02X} (expected 0x{MAGIC:02X})")
        payload = data[HEADER_SIZE: HEADER_SIZE + plen]
        return cls(
            packet_type  = PacketType(ptype),
            session_id   = sid,
            chunk_idx    = chunk,
            total_chunks = total,
            payload      = payload,
            crc          = crc,
        )


# ── 인코더 ────────────────────────────────────────────────────────────────────

def encode_imu(imu: list[list[float]], session_id: int) -> list[bytes]:
    """[T, 6] float → BLE 청크 bytes 리스트 (int16 × 6 × T)."""
    raw = bytearray()
    for frame in imu:
        for v in frame:
            clamped = max(-32768, min(32767, int(v * IMU_SCALE)))
            raw += struct.pack("<h", clamped)
    return _chunked(bytes(raw), PacketType.IMU, session_id)


def encode_pressure(pressure: list[list[float]], session_id: int) -> list[bytes]:
    """[16, 8] float 0–1 → BLE 청크 bytes 리스트 (uint16 × 128)."""
    raw = bytearray()
    for row in pressure:
        for v in row:
            clamped = max(0, min(65535, int(v * PRESSURE_SCALE)))
            raw += struct.pack("<H", clamped)
    return _chunked(bytes(raw), PacketType.PRESSURE, session_id)


def encode_skeleton(skeleton: list[list[list[float]]], session_id: int) -> list[bytes]:
    """[T, 17, 3] float → BLE 청크 bytes 리스트 (float32 × 17 × 3 × T)."""
    raw = bytearray()
    for frame in skeleton:
        for joint in frame:
            for v in joint:
                raw += struct.pack("<f", v)
    return _chunked(bytes(raw), PacketType.SKELETON, session_id)


def _chunked(payload: bytes, ptype: PacketType, session_id: int) -> list[bytes]:
    chunks = [payload[i: i + MAX_PAYLOAD] for i in range(0, len(payload), MAX_PAYLOAD)]
    total = len(chunks)
    return [
        BLEPacket(ptype, session_id, i, total, chunk).to_bytes()
        for i, chunk in enumerate(chunks)
    ]


# ── 디코더 ────────────────────────────────────────────────────────────────────

class StreamAssembler:
    """BLE 청크를 수신하면서 완성된 payload를 반환."""

    def __init__(self, expected_type: PacketType):
        self.expected_type = expected_type
        self._chunks: dict[int, bytes] = {}
        self._total: int | None = None

    def feed(self, packet: BLEPacket) -> bytes | None:
        if packet.packet_type != self.expected_type:
            return None
        if not packet.is_valid():
            raise ValueError(f"CRC 오류: chunk {packet.chunk_idx} (got 0x{packet.crc:04X})")
        self._total = packet.total_chunks
        self._chunks[packet.chunk_idx] = packet.payload
        if len(self._chunks) == self._total:
            result = b"".join(self._chunks[i] for i in range(self._total))
            self.reset()
            return result
        return None

    @property
    def progress(self) -> tuple[int, int | None]:
        return len(self._chunks), self._total

    def missing_chunks(self) -> list[int]:
        if self._total is None:
            return []
        return [i for i in range(self._total) if i not in self._chunks]

    def reset(self) -> None:
        self._chunks.clear()
        self._total = None


def decode_imu(raw: bytes) -> list[list[float]]:
    n = len(raw) // 2
    values = struct.unpack_from(f"<{n}h", raw)
    return [[values[i * 6 + j] / IMU_SCALE for j in range(6)] for i in range(n // 6)]


def decode_pressure(raw: bytes) -> list[list[float]]:
    n = len(raw) // 2
    values = struct.unpack_from(f"<{n}H", raw)
    return [[values[i * 8 + j] / PRESSURE_SCALE for j in range(8)] for i in range(n // 8)]


def decode_skeleton(raw: bytes) -> list[list[list[float]]]:
    n = len(raw) // 4
    values = struct.unpack_from(f"<{n}f", raw)
    s = 17 * 3  # frame stride
    return [
        [[values[i * s + j * 3 + d] for d in range(3)] for j in range(17)]
        for i in range(n // s)
    ]


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _crc16(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFF


def session_id_new() -> int:
    import random
    return random.randint(0, 0xFFFF)
