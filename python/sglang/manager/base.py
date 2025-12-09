"""
Kernel 抽象基类。
所有需要被调度执行的内核都应继承此类。
"""

import abc


class Kernel(abc.ABC):
    """
    内核抽象基类。
    子类需要实现 execute() 方法来定义具体的执行逻辑。
    """

    @abc.abstractmethod
    def execute(self):
        """执行内核的具体逻辑"""
        raise NotImplementedError

