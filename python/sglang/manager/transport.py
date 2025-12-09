"""
IPC 传输层抽象接口及实现。
负责 IPC 资源的生命周期管理和通道创建。
"""

import abc
import mmap
from typing import Optional, Tuple

from .config import SHM_NAME_REGISTRY, generate_shm_name
from .channel import IChannel, SHMChannel
from .registry import ClientRegistry


class IPCTransport(abc.ABC):
    """
    IPC 传输层抽象基类。
    负责 IPC 资源的创建、连接和销毁，以及通道的创建。
    """

    @abc.abstractmethod
    def connect_registry(self) -> Tuple[memoryview, any]:
        """
        连接到注册表。
        
        Returns:
            Tuple[memoryview, any]: 注册表的 memoryview 和底层资源句柄
        
        Raises:
            ConnectionError: 无法连接到注册表
        """
        raise NotImplementedError

    @abc.abstractmethod
    def create_channel(self, channel_name: str) -> Tuple[IChannel, any]:
        """
        创建客户端通道。
        
        Args:
            channel_name: 通道名称
        
        Returns:
            Tuple[IChannel, any]: 通道实例和底层资源句柄
        
        Raises:
            RuntimeError: 无法创建通道
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close_channel(self, channel_name: str, handle: any):
        """
        关闭并清理通道资源。
        
        Args:
            channel_name: 通道名称
            handle: 底层资源句柄
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close_registry(self, handle: any):
        """
        关闭注册表连接。
        
        Args:
            handle: 底层资源句柄
        """
        raise NotImplementedError

    def generate_channel_name(self) -> str:
        """
        生成唯一的通道名称。
        子类可以覆盖此方法以提供不同的命名策略。
        """
        return generate_shm_name()


class SHMTransport(IPCTransport):
    """
    基于 POSIX 共享内存的传输层实现。
    """

    def __init__(self):
        self._posix_ipc = None

    def _ensure_posix_ipc(self):
        """延迟导入 posix_ipc"""
        if self._posix_ipc is None:
            import posix_ipc
            self._posix_ipc = posix_ipc
        return self._posix_ipc

    def connect_registry(self) -> Tuple[memoryview, mmap.mmap]:
        posix_ipc = self._ensure_posix_ipc()
        
        try:
            registry_shm = posix_ipc.SharedMemory(SHM_NAME_REGISTRY)
            registry_mm = mmap.mmap(registry_shm.fd, ClientRegistry.TOTAL_SIZE)
            registry_shm.close_fd()
            return memoryview(registry_mm), registry_mm
        except posix_ipc.ExistentialError:
            raise ConnectionError(
                f"注册表共享内存 {SHM_NAME_REGISTRY} 不存在，调度器可能未启动"
            )

    def create_channel(self, channel_name: str) -> Tuple[IChannel, mmap.mmap]:
        posix_ipc = self._ensure_posix_ipc()
        size = SHMChannel.TOTAL_SIZE
        
        try:
            channel_shm = posix_ipc.SharedMemory(
                channel_name,
                flags=posix_ipc.O_CREAT | posix_ipc.O_RDWR,
                size=size,
            )
        except posix_ipc.ExistentialError:
            # 如果已存在，先删除再创建
            posix_ipc.unlink_shared_memory(channel_name)
            channel_shm = posix_ipc.SharedMemory(
                channel_name,
                flags=posix_ipc.O_CREAT | posix_ipc.O_RDWR,
                size=size,
            )

        channel_mm = mmap.mmap(channel_shm.fd, size)
        channel_shm.close_fd()
        
        # 初始化为零
        channel_mm.seek(0)
        channel_mm.write(b"\x00" * size)
        
        # 创建通道实例
        channel = SHMChannel(memoryview(channel_mm))
        
        return channel, channel_mm

    def close_channel(self, channel_name: str, handle: Optional[mmap.mmap]):
        if handle:
            try:
                handle.close()
            except Exception:
                pass

        if channel_name:
            try:
                posix_ipc = self._ensure_posix_ipc()
                posix_ipc.unlink_shared_memory(channel_name)
            except Exception:
                pass

    def close_registry(self, handle: Optional[mmap.mmap]):
        if handle:
            try:
                handle.close()
            except Exception:
                pass


# ============================================================================
# 工厂函数
# ============================================================================

def create_default_transport() -> IPCTransport:
    """
    创建默认的传输层实例。
    可以通过环境变量或配置切换不同的传输层实现。
    """
    return SHMTransport()
