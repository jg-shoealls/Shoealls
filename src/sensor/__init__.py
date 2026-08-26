"""Sensor layer: BLE protocol, feature extraction, firmware simulator."""

from .ble_protocol import (
    CONTROL_CHAR_UUID,
    IMU_CHAR_UUID,
    PRESSURE_CHAR_UUID,
    SHOEALLS_SERVICE_UUID,
    SKELETON_CHAR_UUID,
    STATUS_CHAR_UUID,
    BLEPacket,
    ControlCmd,
    PacketType,
    StreamAssembler,
    decode_imu,
    decode_pressure,
    decode_skeleton,
    encode_imu,
    encode_pressure,
    encode_skeleton,
    session_id_new,
)
from .feature_extractor import GaitFeatureExtractor, GaitFeatures
from .firmware_sim import SENSOR_SPECS, FirmwareSimulator

__all__ = [
    "CONTROL_CHAR_UUID",
    "IMU_CHAR_UUID",
    "PRESSURE_CHAR_UUID",
    "SENSOR_SPECS",
    "SHOEALLS_SERVICE_UUID",
    "SKELETON_CHAR_UUID",
    "STATUS_CHAR_UUID",
    "BLEPacket",
    "ControlCmd",
    "FirmwareSimulator",
    "GaitFeatureExtractor",
    "GaitFeatures",
    "PacketType",
    "StreamAssembler",
    "decode_imu",
    "decode_pressure",
    "decode_skeleton",
    "encode_imu",
    "encode_pressure",
    "encode_skeleton",
    "session_id_new",
]
