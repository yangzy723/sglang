import mmap
import os
import struct
import threading
import abc
import sys
import time
import ctypes
from typing import Optional

# --- Configuration ---
SHM_NAME_SGLANG = "/kernel_scheduler_sglang"
CLIENT_ID = "sglang"

# SPSC 队列配置（必须与 C++ 端一致）
SPSC_QUEUE_SIZE = 1024        # 队列可存储的消息数量
SPSC_MSG_SIZE = 256           # 每条消息的最大字节数
CACHE_LINE_SIZE = 64          # CPU 缓存行大小

def create_request_message(req_id: str, kernel_type: str) -> bytes:
    """构建符合协议的消息: {kernel_type}|{req_id}|{client_id}"""
    return f"{kernel_type}|{req_id}|{CLIENT_ID}\n".encode('utf-8')


# --- 共享内存 SPSC 队列实现 ---

class SPSCQueue:
    """
    单生产者单消费者无锁队列的 Python 实现。
    与 C++ 端的 SPSCQueue 结构布局一致。
    """
    
    # 结构体布局偏移量
    HEAD_OFFSET = 0                                           # head: uint64_t, 对齐到 64 字节
    TAIL_OFFSET = CACHE_LINE_SIZE                             # tail: uint64_t, 对齐到 64 字节  
    BUFFER_OFFSET = 2 * CACHE_LINE_SIZE                       # buffer 开始位置, 对齐到 64 字节
    
    def __init__(self, shm_view: memoryview, offset: int):
        """
        初始化 SPSC 队列视图。
        
        参数:
            shm_view: 共享内存的 memoryview
            offset: 此队列在共享内存中的偏移量
        """
        self.shm = shm_view
        self.base_offset = offset
        
    def _read_u64(self, offset: int) -> int:
        """原子读取 64 位无符号整数"""
        return struct.unpack_from('<Q', self.shm, self.base_offset + offset)[0]
    
    def _write_u64(self, offset: int, value: int):
        """原子写入 64 位无符号整数"""
        struct.pack_into('<Q', self.shm, self.base_offset + offset, value)
    
    def try_push(self, data: bytes) -> bool:
        """尝试写入消息，返回 True 表示成功，False 表示队列已满"""
        current_tail = self._read_u64(self.TAIL_OFFSET)
        next_tail = (current_tail + 1) % SPSC_QUEUE_SIZE
        
        # 检查队列是否已满
        if next_tail == self._read_u64(self.HEAD_OFFSET):
            return False
        
        # 写入数据
        copy_len = min(len(data), SPSC_MSG_SIZE - 1)
        msg_offset = self.base_offset + self.BUFFER_OFFSET + current_tail * SPSC_MSG_SIZE
        
        # 使用切片操作，更高效
        self.shm[msg_offset:msg_offset + copy_len] = data[:copy_len]
        # 确保字符串结尾
        if copy_len < SPSC_MSG_SIZE:
            self.shm[msg_offset + copy_len] = 0
        
        # 发布写入
        self._write_u64(self.TAIL_OFFSET, next_tail)
        return True
    
    def try_pop(self) -> Optional[bytes]:
        """尝试读取消息，返回消息内容或 None（队列为空）"""
        current_head = self._read_u64(self.HEAD_OFFSET)
        
        # 检查队列是否为空
        if current_head == self._read_u64(self.TAIL_OFFSET):
            return None
        
        # 读取数据
        msg_offset = self.base_offset + self.BUFFER_OFFSET + current_head * SPSC_MSG_SIZE
        
        # 转换为 bytes 后查找字符串结尾
        msg_bytes = bytes(self.shm[msg_offset:msg_offset + SPSC_MSG_SIZE])
        try:
            end = msg_bytes.index(0)
        except ValueError:
            end = SPSC_MSG_SIZE
        
        data = msg_bytes[:end]
        
        # 提交读取
        self._write_u64(self.HEAD_OFFSET, (current_head + 1) % SPSC_QUEUE_SIZE)
        return data
    
    def push_blocking(self, data: bytes, timeout_ms: int = -1) -> bool:
        """阻塞式写入（带超时）- 优化版本：忙等待+退避"""
        import time as time_module
        spin_count = 0
        start_time = time_module.perf_counter() if timeout_ms >= 0 else None
        
        while True:
            if self.try_push(data):
                return True
            
            if timeout_ms >= 0 and start_time:
                elapsed_ms = (time_module.perf_counter() - start_time) * 1000
                if elapsed_ms >= timeout_ms:
                    return False
            
            # 忙等待策略：前 1000 次快速轮询，然后逐渐增加延迟
            if spin_count < 1000:
                # Python 中没有 pause 指令，但可以尝试快速轮询
                spin_count += 1
            elif spin_count < 10000:
                if spin_count % 100 == 0:
                    time.sleep(0.000001)  # 1 微秒（尽可能短）
                spin_count += 1
            else:
                time.sleep(0.00001)  # 10 微秒
                spin_count = 0
    
    def pop_blocking(self, timeout_ms: int = -1) -> Optional[bytes]:
        """阻塞式读取（带超时）- 优化版本：忙等待+退避"""
        import time as time_module
        spin_count = 0
        start_time = time_module.perf_counter() if timeout_ms >= 0 else None
        
        while True:
            result = self.try_pop()
            if result is not None:
                return result
            
            if timeout_ms >= 0 and start_time:
                elapsed_ms = (time_module.perf_counter() - start_time) * 1000
                if elapsed_ms >= timeout_ms:
                    return None
            
            # 忙等待策略：前 1000 次快速轮询，然后逐渐增加延迟
            if spin_count < 1000:
                spin_count += 1
            elif spin_count < 10000:
                if spin_count % 100 == 0:
                    time.sleep(0.000001)  # 1 微秒（尽可能短）
                spin_count += 1
            else:
                time.sleep(0.00001)  # 10 微秒
                spin_count = 0


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
            print("[KernelManager] Connected to Scheduler via SHM.")
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
        """
        向调度器提交内核。阻塞直到收到响应。
        
        流程:
        1. 发送请求
        2. 等待批准
        3. 如果批准 -> 执行 kernel.execute()
        4. 如果拒绝 -> 打印错误日志
        """
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
