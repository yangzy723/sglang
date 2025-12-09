import struct
import time
from .config import CACHE_LINE_SIZE, MAX_REGISTERED_CLIENTS


class ClientRegistry:
    """
    客户端注册表视图，需与 C++ 端布局一致。
    """

    SCHEDULER_READY_OFFSET = 0
    VERSION_OFFSET = CACHE_LINE_SIZE
    ENTRIES_OFFSET = 2 * CACHE_LINE_SIZE

    ENTRY_ACTIVE_OFFSET = 0
    ENTRY_SHM_NAME_OFFSET = 1
    ENTRY_CLIENT_TYPE_OFFSET = ENTRY_SHM_NAME_OFFSET + 64
    ENTRY_UNIQUE_ID_OFFSET = ENTRY_CLIENT_TYPE_OFFSET + 16
    ENTRY_CLIENT_PID_OFFSET = 3 * CACHE_LINE_SIZE
    ENTRY_HEARTBEAT_OFFSET = 4 * CACHE_LINE_SIZE

    ENTRY_SIZE = 5 * CACHE_LINE_SIZE  # 320 字节
    TOTAL_SIZE = ENTRIES_OFFSET + ENTRY_SIZE * MAX_REGISTERED_CLIENTS

    def __init__(self, shm_view: memoryview):
        self.shm = shm_view

    def is_scheduler_ready(self) -> bool:
        return struct.unpack_from("<?", self.shm, self.SCHEDULER_READY_OFFSET)[0]

    def register_client(self, shm_name: str, client_type: str, unique_id: str, pid: int = 0) -> int:
        shm_name_bytes = shm_name.encode("utf-8")[:63]
        client_type_bytes = client_type.encode("utf-8")[:15]
        unique_id_bytes = unique_id.encode("utf-8")[:63]

        for i in range(MAX_REGISTERED_CLIENTS):
            entry_base = self.ENTRIES_OFFSET + i * self.ENTRY_SIZE
            active_offset = entry_base + self.ENTRY_ACTIVE_OFFSET

            current = struct.unpack_from("<?", self.shm, active_offset)[0]
            if not current:
                struct.pack_into("<?", self.shm, active_offset, True)

                shm_name_offset = entry_base + self.ENTRY_SHM_NAME_OFFSET
                client_type_offset = entry_base + self.ENTRY_CLIENT_TYPE_OFFSET
                unique_id_offset = entry_base + self.ENTRY_UNIQUE_ID_OFFSET
                client_pid_offset = entry_base + self.ENTRY_CLIENT_PID_OFFSET
                heartbeat_offset = entry_base + self.ENTRY_HEARTBEAT_OFFSET

                self.shm[shm_name_offset : shm_name_offset + 64] = b"\x00" * 64
                self.shm[shm_name_offset : shm_name_offset + len(shm_name_bytes)] = shm_name_bytes

                self.shm[client_type_offset : client_type_offset + 16] = b"\x00" * 16
                self.shm[client_type_offset : client_type_offset + len(client_type_bytes)] = client_type_bytes

                self.shm[unique_id_offset : unique_id_offset + 64] = b"\x00" * 64
                self.shm[unique_id_offset : unique_id_offset + len(unique_id_bytes)] = unique_id_bytes

                struct.pack_into("<q", self.shm, client_pid_offset, pid)
                timestamp = int(time.perf_counter() * 1000)
                struct.pack_into("<Q", self.shm, heartbeat_offset, timestamp)

                version_val = struct.unpack_from("<I", self.shm, self.VERSION_OFFSET)[0]
                struct.pack_into("<I", self.shm, self.VERSION_OFFSET, version_val + 1)

                return i
        return -1

    def unregister_client(self, slot: int):
        if 0 <= slot < MAX_REGISTERED_CLIENTS:
            entry_base = self.ENTRIES_OFFSET + slot * self.ENTRY_SIZE
            struct.pack_into("<?", self.shm, entry_base, False)

            version_val = struct.unpack_from("<I", self.shm, self.VERSION_OFFSET)[0]
            struct.pack_into("<I", self.shm, self.VERSION_OFFSET, version_val + 1)

    def update_heartbeat(self, slot: int):
        if 0 <= slot < MAX_REGISTERED_CLIENTS:
            entry_base = self.ENTRIES_OFFSET + slot * self.ENTRY_SIZE
            heartbeat_offset = entry_base + self.ENTRY_HEARTBEAT_OFFSET
            timestamp = int(time.perf_counter() * 1000)
            struct.pack_into("<Q", self.shm, heartbeat_offset, timestamp)

