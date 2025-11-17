import socket
import threading
import abc
import sys
from typing import Optional

# --- 来自 IPCProtocol.h 的定义 ---

SCHEDULER_PORT = 9999
LOCALHOST = "127.0.0.1"

def create_request_message(req_id: str, kernel_type: str) -> bytes:
    """
    对应于 IPCProtocol.h 中的 createRequestMessage
    返回字节流以便通过套接字发送。
    """
    return f"{req_id}|{kernel_type}\n".encode('utf-8')

# --- 内核的抽象基类 ---

class Kernel(abc.ABC):
    """
    对应于 C++ 中的 'Kernel' 基类。
    任何被 enqueued 的对象都必须继承这个类并实现 execute()。
    """
    @abc.abstractmethod
    def execute(self):
        """
        内核将执行的实际工作。
        """
        pass

# --- KernelManager 实现 ---

class KernelManager:
    """
    KernelManager 的 Python 实现，作为一个单例客户端
    连接到 C++ 调度器。

    这个类旨在被实例化一次（作为模块单例），
    并在应用程序的整个生命周期内保持连接。
    """
    
    def __init__(self):
        self.sock: Optional[socket.socket] = None
        self.rfile = None  # 用于带缓冲的套接字读取
        self.request_id_counter = 0
        
        # C++ 版本使用 std::atomic 来增加计数器，这表明
        # enqueue 可能是从多个线程调用的。
        # 这个锁确保了整个 (send + receive) 操作是原子的，
        # 防止来自不同线程的响应被混淆。
        self.lock = threading.Lock()
        
        try:
            self._connect_to_scheduler()
            print("[KernelManager] 已初始化并连接到调度器。")
        except Exception as e:
            print(f"[KernelManager] 初始化失败: {e}", file=sys.stderr)
            # 在 C++ 版本中，这里会 exit(EXIT_FAILURE)
            # 在 Python 中，我们重新引发异常以指示严重故障
            raise

    def _connect_to_scheduler(self):
        """
        私有辅助方法，在构造时调用。
        """
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((LOCALHOST, SCHEDULER_PORT))
            # 为 readline() 创建一个文件对象，它会缓冲
            # 并正确处理以 '\n' 结尾的消息。
            self.rfile = self.sock.makefile('rb')
            print("[KernelManager] 成功连接到调度器。")
        except socket.error as e:
            print(f"[KernelManager] 连接失败！调度器进程是否已启动？ 错误: {e}", file=sys.stderr)
            if self.sock:
                self.sock.close()
            raise
        except Exception as e:
            print(f"[KernelManager] Socket 创建失败: {e}", file=sys.stderr)
            raise

    def _generate_request_id(self) -> str:
        """
        生成一个唯一的请求 ID。
        注意：这假定它总是在 `self.lock` 持有期间被调用。
        """
        self.request_id_counter += 1
        return f"req_{self.request_id_counter}"

    def enqueue(self, kernel: Kernel):
        """
        将内核提交给 Scheduler 审批。
        
        此函数将通过持久连接发送请求并阻塞，
        直到收到响应。
        
        参数:
            kernel (Kernel): 一个实现了 execute() 方法的 Kernel 对象。
        """
        
        # 锁定整个事务以确保线程安全
        with self.lock:
            if not self.sock or not self.rfile:
                print("[KernelManager] 错误：未连接。无法对内核进行排队。", file=sys.stderr)
                return

            # 1. 生成 ID 和类型
            req_id = self._generate_request_id()
            
            # Python 的 type(obj).__name__ 相当于 C++
            # 中复杂的 getClassName (demangle)
            kernel_type = type(kernel).__name__
            
            # 2. 创建并发送请求
            request_message = create_request_message(req_id, kernel_type)
            # print(f"[KernelManager] 准备提交 (ID: {req_id}): {kernel_type}")

            try:
                # print(f"[KernelManager] (ID: {req_id}) 发送请求...")
                self.sock.sendall(request_message)

                # 3. 阻塞并等待响应
                # print(f"[KernelManager] (ID: {req_id}) 正在等待调度器批准...")
                response_bytes = self.rfile.readline()

                if not response_bytes:
                    print(f"[KernelManager] (ID: {req_id}) 从服务器读取响应失败。连接已断开。", file=sys.stderr)
                    return

                # 解码并去除末尾的 \n 或 \r\n
                response_message = response_bytes.decode('utf-8').strip()

                # 4. 解析响应
                parts = response_message.split('|')
                if len(parts) != 3 or parts[0] != req_id:
                    print(f"[KernelManager] (ID: {req_id}) 收到格式错误的响应: {response_message}", file=sys.stderr)
                    return

                # 5. 根据权限执行
                permission_granted = (parts[1] == "1")
                reason = parts[2]

                if permission_granted:
                    # print(f"[KernelManager] (ID: {req_id}) 批准！正在执行内核...")
                    try:
                        kernel.execute()
                        # print(f"[KernelManager] (ID: {req_id}) 内核执行完毕。")
                    except Exception as e:
                        print(f"[KernelManager] (ID: {req_id}) 执行时异常: {e}", file=sys.stderr)
                else:
                    print(f"[KernelManager] (ID: {req_id}) 内核被拒绝！原因: {reason}", file=sys.stderr)

            except socket.error as e:
                print(f"[KernelManager] (ID: {req_id}) 套接字错误: {e}。连接可能已断开。", file=sys.stderr)
                # 在发生严重套接字错误时，可能需要关闭并标记为未连接
                self.close()
            except Exception as e:
                print(f"[KernelManager] (ID: {req_id}) 发生意外错误: {e}", file=sys.stderr)

    def close(self):
        """
        显式关闭与调度器的连接。
        """
        with self.lock:
            if self.rfile:
                self.rfile.close()
                self.rfile = None
            if self.sock:
                self.sock.close()
                self.sock = None
                print("[KernelManager] 已关闭与调度器的连接。")

    def __del__(self):
        """
        在对象销毁时尝试关闭连接。
        """
        self.close()

# --- 单例实例 ---

# 创建模块级的单例实例。
# 在 C++ 中，你会使用 KernelManager::getInstance()。
# 在 Python 中，你只需从这个模块导入 `kernel_manager` 实例。
try:
    the_kernel_manager = KernelManager()
except Exception:
    print("[KernelManager] 无法创建 KernelManager 单例。调度器可能未运行。", file=sys.stderr)
    the_kernel_manager = None # 将其设置为 None，以便导入它的代码可以检查