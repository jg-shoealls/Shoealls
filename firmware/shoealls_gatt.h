/**
 * shoealls_gatt.h — Shoealls BLE GATT 서비스 정의
 *
 * 대상 플랫폼: nRF52840 (Nordic SDK 17.x / Zephyr RTOS)
 * BLE 버전   : BT 5.0 (LE 2M PHY, ATT MTU 244)
 *
 * 사용 예 (nRF5 SDK):
 *   ble_uuid128_t svc_uuid = SHOEALLS_SERVICE_UUID_INIT;
 *   err = sd_ble_gatts_service_add(BLE_GATTS_SRVC_TYPE_PRIMARY,
 *                                  &svc_uuid, &svc_handle);
 */

#ifndef SHOEALLS_GATT_H
#define SHOEALLS_GATT_H

#include <stdint.h>
#include <stdbool.h>

/* ─────────────────────────────────────────────────────────────────────────────
 * GATT 서비스 UUID  (128-bit, Little-Endian byte order)
 * ─────────────────────────────────────────────────────────────────────────── */
/* f000aa00-0451-4000-b000-000000000000 */
#define SHOEALLS_SERVICE_UUID_BASE \
    { 0x00,0x00,0x00,0x00, 0x00,0x00, 0x00,0xb0, \
      0x00,0x40, 0x51,0x04, 0x00,0xaa,0x00,0xf0 }

#define SHOEALLS_SERVICE_UUID_16    0xAA00

/* ─────────────────────────────────────────────────────────────────────────────
 * Characteristic UUIDs (16-bit 하위 부분만)
 * ─────────────────────────────────────────────────────────────────────────── */
#define SHOEALLS_IMU_CHAR_UUID      0xAA01  /**< IMU 데이터 (NOTIFY) */
#define SHOEALLS_PRESSURE_CHAR_UUID 0xAA02  /**< 족저압 데이터 (NOTIFY) */
#define SHOEALLS_SKELETON_CHAR_UUID 0xAA03  /**< 스켈레톤 데이터 (NOTIFY) */
#define SHOEALLS_CONTROL_CHAR_UUID  0xAA04  /**< 제어 명령 (WRITE) */
#define SHOEALLS_STATUS_CHAR_UUID   0xAA05  /**< 디바이스 상태 (READ | NOTIFY) */

/* ─────────────────────────────────────────────────────────────────────────────
 * 패킷 상수
 * ─────────────────────────────────────────────────────────────────────────── */
#define SHOEALLS_MAGIC              0xAAU
#define SHOEALLS_HEADER_SIZE        12U
#define SHOEALLS_MAX_PAYLOAD        232U    /* MTU 244 − header 12 */

/** 패킷 타입 */
typedef enum {
    PKT_IMU       = 0x01,
    PKT_PRESSURE  = 0x02,
    PKT_SKELETON  = 0x03,
    PKT_FEATURE   = 0x04,
    PKT_SYNC      = 0xFE,
    PKT_END       = 0xFF,
} shoealls_pkt_type_t;

/** 제어 명령 */
typedef enum {
    CMD_START_SESSION   = 0x01,
    CMD_STOP_SESSION    = 0x02,
    CMD_RESET           = 0x03,
    CMD_REQUEST_RESEND  = 0x04,
    CMD_SET_SAMPLE_RATE = 0x10,
} shoealls_ctrl_cmd_t;

/* ─────────────────────────────────────────────────────────────────────────────
 * 패킷 헤더 구조체  (packed, Little-Endian)
 *
 *  Offset | Size | Field
 *  -------|------|------
 *     0   |  1   | magic      (0xAA)
 *     1   |  1   | type       (shoealls_pkt_type_t)
 *     2   |  2   | session_id (uint16 LE)
 *     4   |  2   | chunk_idx  (uint16 LE)
 *     6   |  2   | total_chunks (uint16 LE)
 *     8   |  2   | payload_len  (uint16 LE)
 *    10   |  2   | crc16       (CRC16 of payload only)
 *    12   |  N   | payload
 * ─────────────────────────────────────────────────────────────────────────── */
#pragma pack(push, 1)
typedef struct {
    uint8_t  magic;
    uint8_t  type;
    uint16_t session_id;
    uint16_t chunk_idx;
    uint16_t total_chunks;
    uint16_t payload_len;
    uint16_t crc16;
    uint8_t  payload[SHOEALLS_MAX_PAYLOAD];
} shoealls_packet_t;
#pragma pack(pop)

/* ─────────────────────────────────────────────────────────────────────────────
 * 센서 데이터 스케일 상수
 * ─────────────────────────────────────────────────────────────────────────── */
/** IMU: float(m/s² or rad/s) × IMU_SCALE → int16 */
#define SHOEALLS_IMU_SCALE          1000

/** Pressure: float(0–1) × PRESSURE_SCALE → uint16 */
#define SHOEALLS_PRESSURE_SCALE     65535

/* ─────────────────────────────────────────────────────────────────────────────
 * 센서 하드웨어 스펙
 * ─────────────────────────────────────────────────────────────────────────── */
#define SHOEALLS_IMU_SAMPLE_RATE    128     /* Hz */
#define SHOEALLS_PRESSURE_SAMPLE_RATE 100  /* Hz */
#define SHOEALLS_SEQUENCE_LENGTH    128     /* frames per session */
#define SHOEALLS_IMU_CHANNELS       6       /* ax,ay,az,gx,gy,gz */
#define SHOEALLS_PRESSURE_ROWS      16
#define SHOEALLS_PRESSURE_COLS      8
#define SHOEALLS_SKELETON_JOINTS    17      /* COCO format */

/* ─────────────────────────────────────────────────────────────────────────────
 * 상태 레지스터 (STATUS characteristic)
 * ─────────────────────────────────────────────────────────────────────────── */
#pragma pack(push, 1)
typedef struct {
    uint8_t  battery_pct;       /**< 0–100 % */
    uint8_t  session_active;    /**< 0=idle, 1=capturing */
    uint16_t session_id;
    uint8_t  imu_ok;
    uint8_t  pressure_ok;
    uint8_t  fw_major;
    uint8_t  fw_minor;
} shoealls_status_t;
#pragma pack(pop)

/* ─────────────────────────────────────────────────────────────────────────────
 * CRC16 (zlib/CRC-32 하위 16비트 — Python 측과 동일)
 * ─────────────────────────────────────────────────────────────────────────── */
static inline uint16_t shoealls_crc16(const uint8_t *data, uint16_t len) {
    /* zlib CRC-32 하위 16비트 (Python: zlib.crc32(data) & 0xFFFF) */
    uint32_t crc = 0xFFFFFFFFU;
    for (uint16_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0xEDB88320U & -(crc & 1));
        }
    }
    return (uint16_t)((crc ^ 0xFFFFFFFFU) & 0xFFFF);
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 패킷 빌더 매크로
 * ─────────────────────────────────────────────────────────────────────────── */
/** 패킷 헤더를 초기화한다 (payload는 별도 복사 필요). */
#define SHOEALLS_PKT_INIT(pkt, _type, _sid, _cidx, _total, _plen) \
    do {                                          \
        (pkt).magic        = SHOEALLS_MAGIC;      \
        (pkt).type         = (_type);             \
        (pkt).session_id   = (_sid);              \
        (pkt).chunk_idx    = (_cidx);             \
        (pkt).total_chunks = (_total);            \
        (pkt).payload_len  = (_plen);             \
        (pkt).crc16 = shoealls_crc16((pkt).payload, (_plen)); \
    } while (0)

#endif /* SHOEALLS_GATT_H */
