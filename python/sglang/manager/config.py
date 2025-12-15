import os
import struct

# 端口与地址 (Socket IPC)
SCHEDULER_PORT = 9999
LOCALHOST = "127.0.0.1"

# 队列配置
SPSC_QUEUE_SIZE = 1024
SPSC_MSG_SIZE = 256
CACHE_LINE_SIZE = 64

# SHM 名称
SHM_NAME_SCHEDULER_PREFIX = "/kernel_scheduler_registry"
SHM_NAME_PREFIX_PYTORCH = "/ks_pytorch_"
SHM_NAME_PREFIX_SGLANG = "/ks_sglang_"

# 客户端限制
MAX_REGISTERED_CLIENTS = 64

# 环境变量
USER = os.getenv("USER", "nouser")
UNIQUE_ID = os.getenv("UNIQUE_ID", "")
CLIENT_ID = "sglang"

# ============================================================
# 内存布局计算 (对应 C++ struct 布局)
# ============================================================

# --- SPSCQueue ---
# layout: head(64) + tail(64) + buffer(...)
OFFSET_Q_HEAD = 0
OFFSET_Q_TAIL = CACHE_LINE_SIZE
OFFSET_Q_BUFFER = 2 * CACHE_LINE_SIZE
# Size = 64 + 64 + (1024 * 256)
SPSC_QUEUE_BYTES = 2 * CACHE_LINE_SIZE + SPSC_QUEUE_SIZE * SPSC_MSG_SIZE

# --- ClientChannelStruct ---
# layout: req_q + resp_q + client_connected(64) + scheduler_ready(64)
OFFSET_CHANNEL_REQ = 0
OFFSET_CHANNEL_RESP = SPSC_QUEUE_BYTES
OFFSET_CHANNEL_CONNECTED = 2 * SPSC_QUEUE_BYTES
OFFSET_CHANNEL_READY = OFFSET_CHANNEL_CONNECTED + CACHE_LINE_SIZE
# Total size
CHANNEL_STRUCT_SIZE = OFFSET_CHANNEL_READY + CACHE_LINE_SIZE

# --- ClientRegistryEntry ---
# layout (alignas 64):
# 0: active (atomic bool)
# 1: shm_name[64]
# 65: client_type[16]
# 81: unique_id[64]
# 192 (align 64): client_pid (int64)
# 256 (align 64): last_heartbeat (uint64)
OFFSET_ENTRY_ACTIVE = 0
OFFSET_ENTRY_NAME = 1
OFFSET_ENTRY_TYPE = 65
OFFSET_ENTRY_UID = 81
OFFSET_ENTRY_PID = 192
OFFSET_ENTRY_HB = 256
ENTRY_SIZE = 5 * CACHE_LINE_SIZE  # 320 bytes

# --- ClientRegistry ---
# layout: scheduler_ready(64) + version(64) + entries[...]
OFFSET_REG_READY = 0
OFFSET_REG_VERSION = CACHE_LINE_SIZE
OFFSET_REG_ENTRIES = 2 * CACHE_LINE_SIZE
REGISTRY_SIZE = OFFSET_REG_ENTRIES + (MAX_REGISTERED_CLIENTS * ENTRY_SIZE)


def get_user_suffix_client():
    return f"_{USER}" if USER else "_nouser"