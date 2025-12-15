import os
import sys
import atexit

from .shm_client import ShmClient
from .config import CLIENT_ID, UNIQUE_ID

class KernelManager:

    def __init__(self):
        self.req_counter = 0
        self.unique_id = UNIQUE_ID
        
        # unique_id 默认为空，与 C++ get_unique_id() 逻辑一致
        pid_str = str(os.getpid())
        print(f"[KernelManager] Initializing connection for {CLIENT_ID}: {pid_str}...")
        
        self.client = ShmClient(CLIENT_ID, pid_str)
        
        if self.client.connect():
            print("[KernelManager] Successfully connected to Scheduler.")
        else:
            print("[KernelManager] Failed to connect to Scheduler!", file=sys.stderr)
            
        # 注册退出清理
        atexit.register(self._cleanup)

    def _cleanup(self):
        if self.client:
            self.client.disconnect()

    def is_connected(self) -> bool:
        return self.client and self.client.is_connected()

    def generate_request_id(self) -> str:
        self.req_counter += 1
        return f"req_{self.req_counter}"

    def get_kernel_type(self, kernel) -> str:
        return type(kernel).__name__

    def create_request_message(self, req_id: str, kernel) -> str:
        k_type = self.get_kernel_type(kernel)
        # 格式: KernelType|ReqId|ClientType[|UniqueId]\n
        msg = f"{k_type}|{req_id}|{CLIENT_ID}"
        if self.unique_id:
            msg += f"|{self.unique_id}"
        msg += "\n"
        return msg

    def enqueue(self, kernel):
        # 如果未连接，降级执行
        if not self.is_connected():
            print("[KernelManager] Not connected. Executing directly.", file=sys.stderr)
            kernel.execute()
            return

        req_id = self.generate_request_id()
        req_msg = self.create_request_message(req_id, kernel)

        # 发送请求
        if not self.client.send_blocking(req_msg):
            print("[KernelManager] Send timeout. Skipping.", file=sys.stderr)
            return

        # 等待响应
        resp_msg = self.client.recv_blocking()
        if not resp_msg:
            print("[KernelManager] Recv timeout.", file=sys.stderr)
            return

        # 解析响应: ReqId|1/0|Msg
        resp_msg = resp_msg.strip()
        parts = resp_msg.split('|')
        
        if len(parts) >= 2 and parts[0] == req_id:
            allowed = (parts[1] == "1")
            if allowed:
                kernel.execute()
            else:
                reason = parts[2] if len(parts) > 2 else "Unknown"
                print(f"[KernelManager] Kernel denied: {reason}", file=sys.stderr)
        else:
            print(f"[KernelManager] Invalid response: {resp_msg}", file=sys.stderr)

try:
    the_kernel_manager = KernelManager()
except Exception:
    the_kernel_manager = None
