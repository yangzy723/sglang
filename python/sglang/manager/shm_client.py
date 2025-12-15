import os
import time
import mmap
import struct
import posix_ipc

from .config import *

class ShmClient:
    def __init__(self, client_type: str, unique_id: str):
        self.client_type = client_type
        self.unique_id = unique_id
        
        # 构造 SHM 名称 (对应 C++ 构造函数)
        prefix = SHM_NAME_PREFIX_SGLANG if client_type == "sglang" else SHM_NAME_PREFIX_PYTORCH
        self.my_shm_name = f"{prefix}{get_user_suffix_client()}_{unique_id}"
        
        self.connected = False
        self.my_slot = -1
        
        # 资源句柄
        self.registry_shm = None
        self.registry_mm = None
        self.channel_shm = None
        self.channel_mm = None

    def __del__(self):
        self.disconnect()

    def connect(self) -> bool:
        if not self._init_registry():
            return False
        if not self._create_channel_shm():
            return False
        if not self._register_to_server():
            return False
        
        self._wait_for_scheduler()
        
        self.connected = True
        return True

    def disconnect(self):
        self.connected = False
        
        # 1. 从 Registry 注销
        if self.registry_mm and self.my_slot >= 0:
            entry_base = OFFSET_REG_ENTRIES + (self.my_slot * ENTRY_SIZE)
            # active = false
            struct.pack_into("<?", self.registry_mm, entry_base + OFFSET_ENTRY_ACTIVE, False)
            # version++
            self._inc_registry_version()

        # 2. 关闭通道
        if self.channel_mm:
            # client_connected = false
            struct.pack_into("<?", self.channel_mm, OFFSET_CHANNEL_CONNECTED, False)
            self.channel_mm.close()
            self.channel_mm = None
            
        if self.channel_shm:
            try:
                self.channel_shm.unlink()
            except posix_ipc.ExistentialError:
                pass
            self.channel_shm = None

        # 3. 关闭 Registry 映射
        if self.registry_mm:
            self.registry_mm.close()
            self.registry_mm = None

    def is_connected(self) -> bool:
        if not self.connected or not self.channel_mm:
            return False
        # 检查 scheduler_ready 标志
        return struct.unpack_from("<?", self.channel_mm, OFFSET_CHANNEL_READY)[0]

    # ======================= 内部初始化逻辑 =======================

    def _init_registry(self) -> bool:
        reg_name = SHM_NAME_SCHEDULER_PREFIX + get_user_suffix_client()
        try:
            # 客户端只读写，不创建
            self.registry_shm = posix_ipc.SharedMemory(reg_name)
            self.registry_mm = mmap.mmap(self.registry_shm.fd, REGISTRY_SIZE)
            self.registry_shm.close_fd()
            
            # 等待 Registry 初始化完成 (scheduler_ready)
            while not struct.unpack_from("<?", self.registry_mm, OFFSET_REG_READY)[0]:
                time.sleep(0.1)
            return True
        except posix_ipc.ExistentialError:
            print(f"[ShmClient] Registry not found: {reg_name}")
            return False
        except Exception as e:
            print(f"[ShmClient] Init registry error: {e}")
            return False

    def _create_channel_shm(self) -> bool:
        try:
            try:
                posix_ipc.unlink_shared_memory(self.my_shm_name)
            except posix_ipc.ExistentialError:
                pass

            # 创建新的 (O_CREAT | O_RDWR)
            self.channel_shm = posix_ipc.SharedMemory(self.my_shm_name, flags=posix_ipc.O_CREAT | posix_ipc.O_RDWR, size=CHANNEL_STRUCT_SIZE)
            self.channel_mm = mmap.mmap(self.channel_shm.fd, CHANNEL_STRUCT_SIZE)
            self.channel_shm.close_fd()

            # 初始化内存清零
            self.channel_mm.seek(0)
            self.channel_mm.write(b'\x00' * CHANNEL_STRUCT_SIZE)

            # 初始化队列状态
            # (head/tail 默认为 0，buffer 默认为 0，无需额外操作)
            
            # client_connected = true
            struct.pack_into("<?", self.channel_mm, OFFSET_CHANNEL_CONNECTED, True)
            # scheduler_ready = false (等待服务端置位)
            struct.pack_into("<?", self.channel_mm, OFFSET_CHANNEL_READY, False)
            
            return True
        except Exception as e:
            print(f"[ShmClient] Create channel error: {e}")
            return False

    def _register_to_server(self) -> bool:
        # 寻找空闲槽位
        for i in range(MAX_REGISTERED_CLIENTS):
            entry_base = OFFSET_REG_ENTRIES + (i * ENTRY_SIZE)
            active_offset = entry_base + OFFSET_ENTRY_ACTIVE
            
            # 检查 active 状态
            if not struct.unpack_from("<?", self.registry_mm, active_offset)[0]:
                # 抢占槽位 (Python GIL 某种程度上保证了这里的安全性，虽不如 atomic CAS 严谨)
                struct.pack_into("<?", self.registry_mm, active_offset, True)
                self.my_slot = i
                
                # 写入元数据
                self._write_str(entry_base + OFFSET_ENTRY_NAME, self.my_shm_name, 64)
                self._write_str(entry_base + OFFSET_ENTRY_TYPE, self.client_type, 16)
                self._write_str(entry_base + OFFSET_ENTRY_UID, self.unique_id, 64)
                
                # PID
                struct.pack_into("<q", self.registry_mm, entry_base + OFFSET_ENTRY_PID, os.getpid())
                # Heartbeat
                self._update_heartbeat()
                
                # 通知 Scanner 更新 version
                self._inc_registry_version()
                return True
        
        print("[ShmClient] Registry is full!")
        return False

    def _wait_for_scheduler(self):
        """对应 C++ waitForScheduler()"""
        attempts = 0
        while not struct.unpack_from("<?", self.channel_mm, OFFSET_CHANNEL_READY)[0]:
            time.sleep(0.01) # 10ms
            attempts += 1
            if attempts % 100 == 0:
                print("[ShmClient] Waiting for scheduler...")

    # ======================= SPSC 队列实现 =======================

    def _spsc_try_push(self, data: bytes) -> bool:
        # Request Queue 在 Channel 的头部
        base = OFFSET_CHANNEL_REQ
        
        tail_offset = base + OFFSET_Q_TAIL
        head_offset = base + OFFSET_Q_HEAD
        buf_base = base + OFFSET_Q_BUFFER

        tail = struct.unpack_from("<Q", self.channel_mm, tail_offset)[0]
        head = struct.unpack_from("<Q", self.channel_mm, head_offset)[0] # Acquire semantics
        
        next_tail = (tail + 1) % SPSC_QUEUE_SIZE
        if next_tail == head:
            return False # Full

        # 写入数据
        copy_len = min(len(data), SPSC_MSG_SIZE - 1)
        target_offset = buf_base + (tail * SPSC_MSG_SIZE)
        
        self.channel_mm[target_offset : target_offset + copy_len] = data[:copy_len]
        self.channel_mm[target_offset + copy_len] = 0 # Null terminator

        # 更新 Tail (Release semantics)
        struct.pack_into("<Q", self.channel_mm, tail_offset, next_tail)
        return True

    def _spsc_try_pop(self) -> str:
        # Response Queue 在 Request Queue 之后
        base = OFFSET_CHANNEL_RESP
        
        tail_offset = base + OFFSET_Q_TAIL
        head_offset = base + OFFSET_Q_HEAD
        buf_base = base + OFFSET_Q_BUFFER

        head = struct.unpack_from("<Q", self.channel_mm, head_offset)[0]
        tail = struct.unpack_from("<Q", self.channel_mm, tail_offset)[0] # Acquire semantics
        
        if head == tail:
            return None # Empty

        # 读取数据
        source_offset = buf_base + (head * SPSC_MSG_SIZE)
        # 读取直到 \0
        raw_bytes = self.channel_mm[source_offset : source_offset + SPSC_MSG_SIZE]
        null_pos = raw_bytes.find(b'\x00')
        if null_pos != -1:
            data_str = raw_bytes[:null_pos].decode('utf-8')
        else:
            data_str = raw_bytes.decode('utf-8')

        # 更新 Head (Release semantics)
        struct.pack_into("<Q", self.channel_mm, head_offset, (head + 1) % SPSC_QUEUE_SIZE)
        return data_str

    def send_blocking(self, msg: str) -> bool:
        attempts = 0
        data_bytes = msg.encode('utf-8')
        while not self._spsc_try_push(data_bytes):
            if not self.is_connected():
                return False
            attempts += 1
            if attempts > 10000000: return False # 简单超时
            # Busy wait yield
            # Python 没有 _mm_pause，sleep(0) 会让出时间片
            pass 
        return True

    def recv_blocking(self) -> str:
        while True:
            msg = self._spsc_try_pop()
            if msg is not None:
                return msg
            if not self.is_connected():
                return ""
            # Busy wait
            pass

    # ======================= 辅助函数 =======================
    
    def _write_str(self, offset, text, max_len):
        b = text.encode('utf-8')[:max_len-1]
        self.registry_mm[offset : offset + len(b)] = b
        self.registry_mm[offset + len(b)] = 0

    def _inc_registry_version(self):
        v = struct.unpack_from("<I", self.registry_mm, OFFSET_REG_VERSION)[0]
        struct.pack_into("<I", self.registry_mm, OFFSET_REG_VERSION, v + 1)

    def _update_heartbeat(self):
        if self.registry_mm and self.my_slot >= 0:
            ts = int(time.time() * 1000)
            hb_offset = OFFSET_REG_ENTRIES + (self.my_slot * ENTRY_SIZE) + OFFSET_ENTRY_HB
            struct.pack_into("<Q", self.registry_mm, hb_offset, ts)