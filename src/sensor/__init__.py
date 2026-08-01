"""Sensor layer: BLE protocol, feature extraction, firmware simulator."""

from .ble_protocol import (
    BLEPacket, PacketType, ControlCmd, StreamAssembler,
    encode_imu, encode_pressure, encode_skeleton,
    decode_imu, decode_pressure, decode_skeleton,
    session_id_new,
    SHOEALLS_SERVICE_UUID, IMU_CHAR_UUID, PRESSURE_CHAR_UUID,
    SKELETON_CHAR_UUID, CONTROL_CHAR_UUID, STATUS_CHAR_UUID,
)
from .feature_extractor import GaitFeatures, GaitFeatureExtractor
from .firmware_sim import FirmwareSimulator, SENSOR_SPECS

__all__ = [
    "BLEPacket", "PacketType", "ControlCmd", "StreamAssembler",
    "encode_imu", "encode_pressure", "encode_skeleton",
    "decode_imu", "decode_pressure", "decode_skeleton",
    "session_id_new",
    "SHOEALLS_SERVICE_UUID", "IMU_CHAR_UUID", "PRESSURE_CHAR_UUID",
    "SKELETON_CHAR_UUID", "CONTROL_CHAR_UUID", "STATUS_CHAR_UUID",
    "GaitFeatures", "GaitFeatureExtractor",
    "FirmwareSimulator", "SENSOR_SPECS",
]
