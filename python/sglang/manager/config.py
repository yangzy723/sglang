import os

# 基础配置（与 server 端保持一致）
SHM_NAME_PREFIX_SGLANG = "/ks_sglang_"
USER_NAME = os.getenv("USER", "nouser")
SHM_NAME_REGISTRY = f"/kernel_scheduler_registry_{USER_NAME}"
CLIENT_ID = "sglang"
UNIQUE_ID = os.getenv("UNIQUE_ID", "")
MAX_REGISTERED_CLIENTS = 64

# SPSC 队列配置（与 C++ 端一致）
SPSC_QUEUE_SIZE = 1024
SPSC_MSG_SIZE = 256
CACHE_LINE_SIZE = 64


def generate_shm_name() -> str:
    """生成唯一的共享内存通道名"""
    suffix = UNIQUE_ID if UNIQUE_ID else str(os.getpid())
    return f"{SHM_NAME_PREFIX_SGLANG}{USER_NAME}_{suffix}"


def create_request_message(req_id: str, kernel_type: str) -> bytes:
    """
    构建符合协议的消息: {kernel_type}|{req_id}|{client_id}|{unique_id}
    """
    return f"{kernel_type}|{req_id}|{CLIENT_ID}|{UNIQUE_ID}\n".encode("utf-8")

