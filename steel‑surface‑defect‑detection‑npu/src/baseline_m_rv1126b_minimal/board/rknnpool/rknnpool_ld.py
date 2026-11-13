"""
RKNN 多实例 NPU 推理线程池。

通过维护多个 RKNNLite 实例 + ThreadPoolExecutor 实现异步流水线推理。
每个实例独立加载模型，轮转分配帧以实现负载均衡。

原始来源: rknnpool_ld.py (项目根目录)
"""

from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

from rknnlite.api import RKNNLite


def init_rknn(model_path: str, instance_id: int = 0) -> RKNNLite:
    """
    初始化单个 RKNNLite 实例。

    Args:
        model_path:   RKNN 模型文件路径
        instance_id:  实例编号（仅用于日志）

    Returns:
        已加载模型并初始化运行时的 RKNNLite 实例
    """
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"RKNN 模型加载失败: {model_path} (ret={ret})")
    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"RKNN 运行时初始化失败 (ret={ret})")
    print(f"  [RKNN-{instance_id}] 已加载: {model_path}")
    return rknn


def init_rknn_pool(model_path: str, tpes: int = 3) -> list:
    """
    批量初始化多个 RKNNLite 实例。

    Args:
        model_path: RKNN 模型文件路径
        tpes:       实例数量

    Returns:
        RKNNLite 实例列表
    """
    return [init_rknn(model_path, i) for i in range(tpes)]


class rknnPoolExecutor:
    """
    RKNN 多线程推理池执行器。

    核心设计:
      - 维护 TPEs 个 RKNNLite 实例
      - 使用 ThreadPoolExecutor 并发推理
      - 轮转分配 (round-robin) 实现负载均衡
      - put/get 接口实现生产者-消费者异步流水线
    """

    def __init__(self, rknnModel: str, TPEs: int, func):
        """
        Args:
            rknnModel: RKNN 模型文件路径
            TPEs:      线程池大小（并发推理实例数）
            func:       后处理函数 func(rknn_instance, image) -> annotated_image
        """
        self.TPEs = TPEs
        self.queue = Queue()
        self.rknnPool = init_rknn_pool(rknnModel, TPEs)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0

    def put(self, frame):
        """
        提交一帧到推理队列（非阻塞）。

        通过轮转计数器将帧分配给不同 RKNN 实例。
        """
        self.queue.put(
            self.pool.submit(self.func, self.rknnPool[self.num % self.TPEs], frame)
        )
        self.num += 1

    def get(self):
        """
        获取一个推理完成的结果（阻塞直到有结果）。

        Returns:
            (annotated_frame, success_flag)
        """
        if self.queue.empty():
            return None, False
        temp = [self.queue.get()]
        for frame in as_completed(temp):
            return frame.result(), True
        return None, False

    def get_num(self) -> int:
        """返回已提交帧总数。"""
        return self.num

    def release(self):
        """释放所有 NPU 资源。"""
        self.pool.shutdown()
        for rknn in self.rknnPool:
            rknn.release()
