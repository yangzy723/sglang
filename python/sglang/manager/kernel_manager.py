import socket
import threading
import abc
import sys
from typing import Optional

# --- Configuration ---
SCHEDULER_PORT = 9999
LOCALHOST = "127.0.0.1"
CLIENT_ID = "sglang"

def create_request_message(req_id: str, kernel_type: str) -> bytes:
    """构建符合协议的消息: {kernel_type}|{req_id}|{client_id}"""
    return f"{kernel_type}|{req_id}|{CLIENT_ID}\n".encode('utf-8')

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
    设计目标：线程安全、故障恢复、静默运行（仅报错时输出）。
    """
    
    def __init__(self):
        self.sock: Optional[socket.socket] = None
        self.rfile = None
        self.request_id_counter = 0
        self.lock = threading.Lock()
        
        try:
            self._connect_to_scheduler()
            # 仅在初始化成功时打印一次，后续保持静默
            print("[KernelManager] Connected to Scheduler.")
        except Exception as e:
            print(f"[KernelManager] Initialization failed: {e}", file=sys.stderr)
            raise

    def _connect_to_scheduler(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0) # 设置超时防止死锁
        self.sock.connect((LOCALHOST, SCHEDULER_PORT))
        self.rfile = self.sock.makefile('rb')

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
            if not self.sock or not self.rfile:
                print("[KernelManager] Error: Not connected.", file=sys.stderr)
                return

            req_id = self._generate_request_id()
            kernel_type = type(kernel).__name__
            
            try:
                # 1. 发送
                self.sock.sendall(create_request_message(req_id, kernel_type))

                # 2. 接收响应
                response_bytes = self.rfile.readline()
                if not response_bytes:
                    raise ConnectionResetError("Empty response from scheduler")

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

            except (socket.error, ConnectionResetError) as e:
                print(f"[KernelManager] Connection Error (ID: {req_id}): {e}", file=sys.stderr)
                self.close()
            except Exception as e:
                print(f"[KernelManager] Unexpected Error (ID: {req_id}): {e}", file=sys.stderr)

    def close(self):
        with self.lock:
            if self.rfile:
                try: self.rfile.close()
                except: pass
                self.rfile = None
            if self.sock:
                try: self.sock.close()
                except: pass
                self.sock = None
                print("[KernelManager] Connection closed.")

    def __del__(self):
        self.close()

# --- Singleton Instance ---

try:
    the_kernel_manager = KernelManager()
except Exception:
    the_kernel_manager = None
