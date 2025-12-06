import mmap
import os
import struct
import threading
import abc
import sys
import time
import ctypes
import os
from typing import Optional

# --- Configuration ---
SHM_NAME_SGLANG = "/kernel_scheduler_sglang"
CLIENT_ID = "sglang"
UNIQUE_ID = os.getenv("UNIQUE_ID", "")

# SPSC 队列配置（必须与 C++ 端一致）
SPSC_QUEUE_SIZE = 1024        # 队列可存储的消息数量
SPSC_MSG_SIZE = 256           # 每条消息的最大字节数
CACHE_LINE_SIZE = 64          # CPU 缓存行大小

def create_request_message(req_id: str, kernel_type: str) -> bytes:
    """
    构建符合协议的消息: {kernel_type}|{req_id}|{client_id}|{unique_id}
    """
    return f"{kernel_type}|{req_id}|{CLIENT_ID}|{UNIQUE_ID}\n".encode('utf-8')


# --- 共享内存 SPSC 队列实现 ---

class SPSCQueue:
    """
    单生产者单消费者无锁队列的 Python 实现（高性能版本）。
    与 C++ 端的 SPSCQueue 结构布局一致。
    """
    
    # 结构体布局偏移量
    HEAD_OFFSET = 0                                           # head: uint64_t, 对齐到 64 字节
    TAIL_OFFSET = CACHE_LINE_SIZE                             # tail: uint64_t, 对齐到 64 字节  
    BUFFER_OFFSET = 2 * CACHE_LINE_SIZE                       # buffer 开始位置, 对齐到 64 字节
    
    def __init__(self, shm_view: memoryview, offset: int):
        """
        初始化 SPSC 队列视图。
        """
        self.shm = shm_view
        self.base_offset = offset
        # 预计算常用偏移量
        self._head_abs = self.base_offset + self.HEAD_OFFSET
        self._tail_abs = self.base_offset + self.TAIL_OFFSET
        self._buffer_abs = self.base_offset + self.BUFFER_OFFSET
        
    def _read_u64(self, abs_offset: int) -> int:
        """读取 64 位无符号整数（使用绝对偏移）"""
        return struct.unpack_from('<Q', self.shm, abs_offset)[0]
    
    def _write_u64(self, abs_offset: int, value: int):
        """写入 64 位无符号整数（使用绝对偏移）"""
        struct.pack_into('<Q', self.shm, abs_offset, value)
    
    def try_push(self, data: bytes) -> bool:
        """尝试写入消息，返回 True 表示成功，False 表示队列已满"""
        current_tail = self._read_u64(self._tail_abs)
        next_tail = (current_tail + 1) % SPSC_QUEUE_SIZE
        
        # 检查队列是否已满
        if next_tail == self._read_u64(self._head_abs):
            return False
        
        # 写入数据
        copy_len = min(len(data), SPSC_MSG_SIZE - 1)
        msg_offset = self._buffer_abs + current_tail * SPSC_MSG_SIZE
        
        # 使用切片操作写入
        self.shm[msg_offset:msg_offset + copy_len] = data[:copy_len]
        # 确保字符串结尾
        if copy_len < SPSC_MSG_SIZE:
            self.shm[msg_offset + copy_len] = 0
        
        # 发布写入
        self._write_u64(self._tail_abs, next_tail)
        return True
    
    def try_pop(self) -> Optional[bytes]:
        """尝试读取消息，返回消息内容或 None（队列为空）"""
        current_head = self._read_u64(self._head_abs)
        
        # 检查队列是否为空
        if current_head == self._read_u64(self._tail_abs):
            return None
        
        # 读取数据
        msg_offset = self._buffer_abs + current_head * SPSC_MSG_SIZE
        
        # 优化：典型消息长度约 30-50 字节，先读取 64 字节
        first_chunk = bytes(self.shm[msg_offset:msg_offset + 64])
        null_pos = first_chunk.find(b'\x00')
        
        if null_pos >= 0:
            # 找到了结尾，消息在前 64 字节内
            data = first_chunk[:null_pos]
        else:
            # 消息较长，读取剩余部分
            rest = bytes(self.shm[msg_offset + 64:msg_offset + SPSC_MSG_SIZE])
            null_pos = rest.find(b'\x00')
            if null_pos >= 0:
                data = first_chunk + rest[:null_pos]
            else:
                data = first_chunk + rest
        
        # 提交读取
        self._write_u64(self._head_abs, (current_head + 1) % SPSC_QUEUE_SIZE)
        return data
    
    def push_blocking(self, data: bytes, timeout_ms: int = -1) -> bool:
        """阻塞式写入 - 高性能版本：纯忙等待（高频场景）"""
        # 对于高频场景，直接忙等待比 sleep 更快
        # 因为 Python 的 time.sleep() 最小精度约 1ms
        if timeout_ms < 0:
            # 无超时，纯忙等待
            while not self.try_push(data):
                pass
            return True
        else:
            # 有超时
            start = time.perf_counter()
            deadline = start + timeout_ms / 1000.0
            while True:
                if self.try_push(data):
                    return True
                if time.perf_counter() >= deadline:
                    return False
    
    def pop_blocking(self, timeout_ms: int = -1) -> Optional[bytes]:
        """阻塞式读取 - 高性能版本：纯忙等待（高频场景）"""
        if timeout_ms < 0:
            # 无超时，纯忙等待
            while True:
                result = self.try_pop()
                if result is not None:
                    return result
        else:
            # 有超时
            start = time.perf_counter()
            deadline = start + timeout_ms / 1000.0
            while True:
                result = self.try_pop()
                if result is not None:
                    return result
                if time.perf_counter() >= deadline:
                    return None


class ClientChannel:
    """
    客户端通道 - 包含双向 SPSC 队列。
    与 C++ 端的 ClientChannel 结构布局一致。
    """
    
    # 计算各组件大小
    SPSC_QUEUE_STRUCT_SIZE = 2 * CACHE_LINE_SIZE + SPSC_QUEUE_SIZE * SPSC_MSG_SIZE
    
    # 偏移量
    REQUEST_QUEUE_OFFSET = 0
    RESPONSE_QUEUE_OFFSET = SPSC_QUEUE_STRUCT_SIZE
    CLIENT_CONNECTED_OFFSET = 2 * SPSC_QUEUE_STRUCT_SIZE
    SCHEDULER_READY_OFFSET = CLIENT_CONNECTED_OFFSET + CACHE_LINE_SIZE
    
    # 总大小
    TOTAL_SIZE = SCHEDULER_READY_OFFSET + CACHE_LINE_SIZE
    
    def __init__(self, shm_view: memoryview):
        self.shm = shm_view
        self.request_queue = SPSCQueue(shm_view, self.REQUEST_QUEUE_OFFSET)
        self.response_queue = SPSCQueue(shm_view, self.RESPONSE_QUEUE_OFFSET)
    
    def is_scheduler_ready(self) -> bool:
        """检查调度器是否已准备好"""
        value = struct.unpack_from('<?', self.shm, self.SCHEDULER_READY_OFFSET)[0]
        return value
    
    def set_client_connected(self, connected: bool):
        """设置客户端连接状态"""
        struct.pack_into('<?', self.shm, self.CLIENT_CONNECTED_OFFSET, connected)


# --- Abstract Base Class ---

class Kernel(abc.ABC):
    @abc.abstractmethod
    def execute(self):
        """执行内核的具体逻辑"""
        pass


# --- KernelManager Implementation ---

class KernelManager:
    """
    单例客户端，用于管理与 C++ 调度器的通信。
    使用共享内存 SPSC 队列进行通信。
    设计目标：线程安全、故障恢复、静默运行（仅报错时输出）。
    """
    def __init__(self):
        self.channel: Optional[ClientChannel] = None
        self.shm_fd: Optional[int] = None
        self.shm_mm: Optional[mmap.mmap] = None
        self.request_id_counter = 0
        self.lock = threading.Lock()
        self.connected = False
        
        try:
            self._connect_to_scheduler()
            # 仅在初始化成功时打印一次，后续保持静默
            print(f"[KernelManager] Connected to Scheduler via SHM (UNIQUE_ID: {UNIQUE_ID}).")
        except Exception as e:
            print(f"[KernelManager] Initialization failed: {e}", file=sys.stderr)
            # 不抛出异常，允许降级运行

    def _connect_to_scheduler(self):
        """连接到调度器的共享内存"""
        import posix_ipc
        
        try:
            # 打开共享内存（由调度器创建）
            shm = posix_ipc.SharedMemory(SHM_NAME_SGLANG)
            self.shm_fd = shm.fd
            
            # 映射共享内存
            self.shm_mm = mmap.mmap(shm.fd, ClientChannel.TOTAL_SIZE)
            shm.close_fd()  # mmap 完成后可以关闭 fd
            
            # 创建通道视图
            shm_view = memoryview(self.shm_mm)
            self.channel = ClientChannel(shm_view)
            
            # 等待调度器准备好（最多等待 5 秒）
            wait_count = 0
            max_wait = 50  # 50 * 100ms = 5秒
            while not self.channel.is_scheduler_ready():
                wait_count += 1
                if wait_count > max_wait:
                    raise TimeoutError("等待调度器超时")
                time.sleep(0.1)
            
            # 标记客户端已连接
            self.channel.set_client_connected(True)
            self.connected = True
            
        except posix_ipc.ExistentialError:
            raise ConnectionError(f"共享内存 {SHM_NAME_SGLANG} 不存在，调度器可能未启动")

    def _generate_request_id(self) -> str:
        self.request_id_counter += 1
        return f"req_{self.request_id_counter}"

    def enqueue(self, kernel: Kernel):
        with self.lock:
            if not self.connected or not self.channel:
                # 降级模式：直接执行内核
                try:
                    kernel.execute()
                except Exception as e:
                    print(f"[KernelManager] Execution Error (degraded mode): {e}", file=sys.stderr)
                return

            req_id = self._generate_request_id()
            kernel_type = type(kernel).__name__
            
            try:
                # 1. 发送请求到请求队列
                request_msg = create_request_message(req_id, kernel_type)
                if not self.channel.request_queue.push_blocking(request_msg, 5000):
                    print(f"[KernelManager] Timeout sending request (ID: {req_id})", file=sys.stderr)
                    return

                # 2. 等待响应
                response_bytes = self.channel.response_queue.pop_blocking(10000)
                if response_bytes is None:
                    print(f"[KernelManager] Timeout waiting for response (ID: {req_id})", file=sys.stderr)
                    return

                response_message = response_bytes.decode('utf-8').strip()
                
                # 3. 解析: expected format "req_id|1|reason"
                parts = response_message.split('|')
                if len(parts) != 3 or parts[0] != req_id:
                    print(f"[KernelManager] Protocol Error (ID: {req_id}): Malformed response '{response_message}'", file=sys.stderr)
                    return

                permission_granted = (parts[1] == "1")
                reason = parts[2]

                # 4. 执行
                if permission_granted:
                    try:
                        kernel.execute()
                    except Exception as e:
                        print(f"[KernelManager] Execution Error (ID: {req_id}): {e}", file=sys.stderr)
                else:
                    print(f"[KernelManager] Denied (ID: {req_id}): {reason}", file=sys.stderr)

            except Exception as e:
                print(f"[KernelManager] Unexpected Error (ID: {req_id}): {e}", file=sys.stderr)

    def close(self):
        with self.lock:
            if self.channel:
                self.channel.set_client_connected(False)
            
            if self.shm_mm:
                try:
                    self.shm_mm.close()
                except:
                    pass
                self.shm_mm = None
            
            self.channel = None
            self.connected = False
            print("[KernelManager] Connection closed.")

    def __del__(self):
        self.close()


# --- Singleton Instance ---

try:
    the_kernel_manager = KernelManager()
except Exception:
    the_kernel_manager = None