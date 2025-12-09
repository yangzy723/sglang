"""
通道抽象层。
定义 IChannel 接口，支持多种 IPC 实现方式。
"""

import abc
import struct
import time
from typing import Optional

from .config import CACHE_LINE_SIZE, SPSC_MSG_SIZE, SPSC_QUEUE_SIZE


class IChannel(abc.ABC):
    """
    通道抽象接口。
    定义客户端与调度器之间的双向通信接口。
    """

    @abc.abstractmethod
    def send_request(self, data: bytes, timeout_ms: int = -1) -> bool:
        """
        发送请求到调度器。
        
        Args:
            data: 请求数据
            timeout_ms: 超时时间（毫秒），-1 表示无限等待
        
        Returns:
            是否发送成功
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recv_response(self, timeout_ms: int = -1) -> Optional[bytes]:
        """
        接收调度器的响应。
        
        Args:
            timeout_ms: 超时时间（毫秒），-1 表示无限等待
        
        Returns:
            响应数据，超时返回 None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_scheduler_ready(self) -> bool:
        """检查调度器是否就绪"""
        raise NotImplementedError

    @abc.abstractmethod
    def set_client_connected(self, connected: bool):
        """设置客户端连接状态"""
        raise NotImplementedError


# ============================================================================
# 共享内存通道实现
# ============================================================================

class SPSCQueue:
    """
    单生产者单消费者无锁队列。
    基于共享内存的环形缓冲区实现，与 C++ 端布局一致。
    """

    HEAD_OFFSET = 0
    TAIL_OFFSET = CACHE_LINE_SIZE
    BUFFER_OFFSET = 2 * CACHE_LINE_SIZE

    def __init__(self, shm_view: memoryview, offset: int):
        self.shm = shm_view
        self.base_offset = offset
        self._head_abs = self.base_offset + self.HEAD_OFFSET
        self._tail_abs = self.base_offset + self.TAIL_OFFSET
        self._buffer_abs = self.base_offset + self.BUFFER_OFFSET

    def _read_u64(self, abs_offset: int) -> int:
        return struct.unpack_from("<Q", self.shm, abs_offset)[0]

    def _write_u64(self, abs_offset: int, value: int):
        struct.pack_into("<Q", self.shm, abs_offset, value)

    def try_push(self, data: bytes) -> bool:
        current_tail = self._read_u64(self._tail_abs)
        next_tail = (current_tail + 1) % SPSC_QUEUE_SIZE
        if next_tail == self._read_u64(self._head_abs):
            return False

        copy_len = min(len(data), SPSC_MSG_SIZE - 1)
        msg_offset = self._buffer_abs + current_tail * SPSC_MSG_SIZE
        self.shm[msg_offset : msg_offset + copy_len] = data[:copy_len]
        if copy_len < SPSC_MSG_SIZE:
            self.shm[msg_offset + copy_len] = 0

        self._write_u64(self._tail_abs, next_tail)
        return True

    def try_pop(self) -> Optional[bytes]:
        current_head = self._read_u64(self._head_abs)
        if current_head == self._read_u64(self._tail_abs):
            return None

        msg_offset = self._buffer_abs + current_head * SPSC_MSG_SIZE
        first_chunk = bytes(self.shm[msg_offset : msg_offset + 64])
        null_pos = first_chunk.find(b"\x00")
        if null_pos >= 0:
            data = first_chunk[:null_pos]
        else:
            rest = bytes(self.shm[msg_offset + 64 : msg_offset + SPSC_MSG_SIZE])
            null_pos = rest.find(b"\x00")
            data = first_chunk + (rest[:null_pos] if null_pos >= 0 else rest)

        self._write_u64(self._head_abs, (current_head + 1) % SPSC_QUEUE_SIZE)
        return data

    def push_blocking(self, data: bytes, timeout_ms: int = -1) -> bool:
        if timeout_ms < 0:
            while not self.try_push(data):
                pass
            return True

        deadline = time.perf_counter() + timeout_ms / 1000.0
        while True:
            if self.try_push(data):
                return True
            if time.perf_counter() >= deadline:
                return False

    def pop_blocking(self, timeout_ms: int = -1) -> Optional[bytes]:
        if timeout_ms < 0:
            while True:
                result = self.try_pop()
                if result is not None:
                    return result

        deadline = time.perf_counter() + timeout_ms / 1000.0
        while True:
            result = self.try_pop()
            if result is not None:
                return result
            if time.perf_counter() >= deadline:
                return None


class SHMChannel(IChannel):
    """
    基于共享内存的通道实现。
    使用 SPSC 无锁队列进行双向通信。
    """

    SPSC_QUEUE_STRUCT_SIZE = 2 * CACHE_LINE_SIZE + SPSC_QUEUE_SIZE * SPSC_MSG_SIZE

    REQUEST_QUEUE_OFFSET = 0
    RESPONSE_QUEUE_OFFSET = SPSC_QUEUE_STRUCT_SIZE
    CLIENT_CONNECTED_OFFSET = 2 * SPSC_QUEUE_STRUCT_SIZE
    SCHEDULER_READY_OFFSET = CLIENT_CONNECTED_OFFSET + CACHE_LINE_SIZE

    TOTAL_SIZE = SCHEDULER_READY_OFFSET + CACHE_LINE_SIZE

    def __init__(self, shm_view: memoryview):
        self.shm = shm_view
        self._request_queue = SPSCQueue(shm_view, self.REQUEST_QUEUE_OFFSET)
        self._response_queue = SPSCQueue(shm_view, self.RESPONSE_QUEUE_OFFSET)

    def send_request(self, data: bytes, timeout_ms: int = -1) -> bool:
        return self._request_queue.push_blocking(data, timeout_ms)

    def recv_response(self, timeout_ms: int = -1) -> Optional[bytes]:
        return self._response_queue.pop_blocking(timeout_ms)

    def is_scheduler_ready(self) -> bool:
        return struct.unpack_from("<?", self.shm, self.SCHEDULER_READY_OFFSET)[0]

    def set_client_connected(self, connected: bool):
        struct.pack_into("<?", self.shm, self.CLIENT_CONNECTED_OFFSET, connected)


# ============================================================================
# 未来可扩展的其他通道实现
# ============================================================================

# class SocketChannel(IChannel):
#     """
#     基于 Unix Socket 的通道实现（示例，未实现）。
#     """
#     def __init__(self, socket_path: str):
#         self.socket_path = socket_path
#         self.socket = None
#
#     def send_request(self, data: bytes, timeout_ms: int = -1) -> bool:
#         # 通过 socket 发送
#         pass
#
#     def recv_response(self, timeout_ms: int = -1) -> Optional[bytes]:
#         # 通过 socket 接收
#         pass
#
#     def is_scheduler_ready(self) -> bool:
#         # 通过握手协议检查
#         pass
#
#     def set_client_connected(self, connected: bool):
#         # 发送连接/断开消息
#         pass

