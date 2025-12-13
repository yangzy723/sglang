"""
KernelManager - 客户端入口。
负责注册、通道建立、请求/响应及降级执行。
"""

import atexit
import os
import sys
import threading
import time
import weakref
from typing import Optional

from .config import CLIENT_ID, UNIQUE_ID, create_request_message
from .channel import IChannel
from .base import Kernel
from .registry import ClientRegistry
from .transport import IPCTransport, create_default_transport


class KernelManager:
    """
    客户端入口：负责注册、通道建立、请求/响应及降级执行。
    通过依赖注入 IPCTransport 和 IChannel 实现与具体 IPC 方式的完全解耦。
    """

    def __init__(self, transport: Optional[IPCTransport] = None):
        """
        初始化 KernelManager。
        
        Args:
            transport: IPC 传输层实例。如果为 None，则使用默认传输层。
        """
        self.transport = transport or create_default_transport()
        self.channel: Optional[IChannel] = None
        self.registry: Optional[ClientRegistry] = None
        self.registry_slot: int = -1
        self.channel_name: str = self.transport.generate_channel_name()
        self.channel_handle = None
        self.registry_handle = None
        self.request_id_counter = 0
        self.lock = threading.Lock()
        self.connected = False

        try:
            self._connect_to_scheduler()
            # 注册 atexit 清理函数，确保进程退出时清理共享内存
            self._weak_self = weakref.ref(self)
            atexit.register(self._atexit_cleanup)
            print(
                f"[KernelManager] Connected to Scheduler via IPC "
                f"(UNIQUE_ID: {UNIQUE_ID}, Channel: {self.channel_name})."
            )
        except Exception as e:
            print(f"[KernelManager] Initialization failed: {e}", file=sys.stderr)
            # 不抛出异常，允许降级运行

    def _connect_to_scheduler(self):
        """
        连接到调度器。使用注入的传输层进行 IPC 资源管理。
        """
        # 通过传输层连接注册表
        registry_view, self.registry_handle = self.transport.connect_registry()
        self.registry = ClientRegistry(registry_view)

        # 等待调度器就绪
        wait_count = 0
        max_wait = 50  # 5 秒
        while not self.registry.is_scheduler_ready():
            wait_count += 1
            if wait_count > max_wait:
                raise TimeoutError("等待调度器超时，注册表已存在但调度器未就绪")
            time.sleep(0.1)

        # 通过传输层创建客户端通道
        self.channel, self.channel_handle = self.transport.create_channel(
            self.channel_name
        )

        # 注册客户端
        unique_id_str = UNIQUE_ID if UNIQUE_ID else str(os.getpid())
        self.registry_slot = self.registry.register_client(
            self.channel_name, CLIENT_ID, unique_id_str, os.getpid()
        )
        if self.registry_slot < 0:
            raise RuntimeError("注册表已满，无法注册客户端")

        self.channel.set_client_connected(True)

        # 等待调度器服务通道
        wait_count = 0
        max_wait_channel = 100  # 10 秒
        while not self.channel.is_scheduler_ready():
            wait_count += 1
            if wait_count > max_wait_channel:
                self.registry.unregister_client(self.registry_slot)
                raise TimeoutError(f"等待调度器服务通道超时: {self.channel_name}")
            time.sleep(0.1)

        self.connected = True

    def _generate_request_id(self) -> str:
        self.request_id_counter += 1
        return f"req_{self.request_id_counter}"

    def enqueue(self, kernel: Kernel):
        """
        将内核加入执行队列。
        如果未连接到调度器，则直接执行（降级模式）。
        """
        with self.lock:
            if not self.connected or not self.channel:
                try:
                    kernel.execute()
                except Exception as e:
                    print(
                        f"[KernelManager] Execution Error (degraded mode): {e}",
                        file=sys.stderr,
                    )
                return

            req_id = self._generate_request_id()
            kernel_type = type(kernel).__name__

            try:
                # 发送请求
                request_msg = create_request_message(req_id, kernel_type)
                if not self.channel.send_request(request_msg, 5000):
                    print(
                        f"[KernelManager] Timeout sending request (ID: {req_id})",
                        file=sys.stderr,
                    )
                    return

                # 等待响应
                response_bytes = self.channel.recv_response(10000)
                if response_bytes is None:
                    print(
                        f"[KernelManager] Timeout waiting for response (ID: {req_id})",
                        file=sys.stderr,
                    )
                    return

                # 解析响应
                response_message = response_bytes.decode("utf-8").strip()
                parts = response_message.split("|")
                if len(parts) != 3 or parts[0] != req_id:
                    print(
                        (
                            f"[KernelManager] Protocol Error (ID: {req_id}): "
                            f"Malformed response '{response_message}'"
                        ),
                        file=sys.stderr,
                    )
                    return

                permission_granted = parts[1] == "1"
                reason = parts[2]

                if permission_granted:
                    try:
                        kernel.execute()
                    except Exception as e:
                        print(
                            f"[KernelManager] Execution Error (ID: {req_id}): {e}",
                            file=sys.stderr,
                        )
                else:
                    print(f"[KernelManager] Denied (ID: {req_id}): {reason}", file=sys.stderr)

            except Exception as e:
                print(f"[KernelManager] Unexpected Error (ID: {req_id}): {e}", file=sys.stderr)

    def close(self):
        """
        关闭连接并清理资源。
        """
        with self.lock:
            if not self.connected and self.channel is None:
                return  # 已经清理过了

            if self.registry and self.registry_slot >= 0:
                try:
                self.registry.unregister_client(self.registry_slot)
                except Exception:
                    pass
                self.registry_slot = -1

            if self.channel:
                try:
                self.channel.set_client_connected(False)
                except Exception:
                    pass

            # 通过传输层关闭通道
            if self.channel_name and self.transport:
            self.transport.close_channel(self.channel_name, self.channel_handle)
            self.channel_handle = None

            # 通过传输层关闭注册表连接
            if self.transport:
            self.transport.close_registry(self.registry_handle)
            self.registry_handle = None

            self.channel = None
            self.registry = None
            self.connected = False
            print(f"[KernelManager] Connection closed (Channel: {self.channel_name}).")

    def _atexit_cleanup(self):
        """atexit 回调，确保进程退出时清理共享内存"""
        try:
            self.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


# ============================================================================
# 全局单例
# ============================================================================

try:
    the_kernel_manager = KernelManager()
except Exception:
    the_kernel_manager = None
