# 文件名: SPARCS/graph/node.py
# 描述: Node 类的同步实现版本

import shortuuid
import traceback
from typing import List, Any, Optional, Dict
from abc import ABC, abstractmethod
import warnings
import copy
import time # <--- NEW: 导入 time 模块用于模拟

class Node(ABC):
    """
    表示图计算框架中的一个处理单元 (同步版本)。

    此类封装了图中节点的功能，管理与其他节点的连接，处理输入输出，
    并执行指定的操作。它支持空间和时间上的信息传递。
    """

    def __init__(self,
                 id: Optional[str] = None,
                 agent_name: str = "",
                 domain: str = "",
                 llm_name: str = ""):
        # ... (构造函数 __init__ 的所有代码保持不变) ...
        self.id: str = id if id is not None else shortuuid.ShortUUID().random(length=4)
        self.agent_name: str = agent_name
        self.domain: str = domain
        self.llm_name: str = llm_name
        self.role: str = ""

        self.spatial_predecessors: List['Node'] = []
        self.spatial_successors: List['Node'] = []
        self.temporal_predecessors: List['Node'] = []
        self.temporal_successors: List['Node'] = []

        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.raw_inputs: List[Any] = []

        self.memory: Any = None
        self.last_memory: Dict[str, List[Any]] = {'inputs': [], 'outputs': [], 'raw_inputs': []}

        self.reset()

    def reset(self):
        # ... (reset 方法代码保持不变) ...
        self.inputs = []
        self.outputs = []
        self.raw_inputs = []
        if isinstance(self.memory, dict):
            self.memory = {}
        elif isinstance(self.memory, list):
            self.memory = []
        else:
            self.memory = None

    @property
    def node_name(self):
        return self.__class__.__name__

    # --- 连接管理、内存更新、信息获取等辅助方法 (所有这些都保持不变) ---
    def add_predecessor(self, operation: 'Node', st='spatial'):
        # ... (代码不变) ...
        target_predecessors = self.spatial_predecessors if st == 'spatial' else self.temporal_predecessors
        target_successors = operation.spatial_successors if st == 'spatial' else operation.temporal_successors
        if operation not in target_predecessors:
            target_predecessors.append(operation)
            if self not in target_successors:
                target_successors.append(self)

    def add_successor(self, operation: 'Node', st='spatial'):
        # ... (代码不变) ...
        target_successors = self.spatial_successors if st == 'spatial' else self.temporal_successors
        target_predecessors = operation.spatial_predecessors if st == 'spatial' else operation.temporal_predecessors
        if operation not in target_successors:
            target_successors.append(operation)
            if self not in target_predecessors:
                target_predecessors.append(self)

    def remove_predecessor(self, operation: 'Node', st='spatial'):
        # ... (代码不变) ...
        target_predecessors = self.spatial_predecessors if st == 'spatial' else self.temporal_predecessors
        target_successors = operation.spatial_successors if st == 'spatial' else operation.temporal_successors
        if operation in target_predecessors:
            target_predecessors.remove(operation)
            if self in target_successors:
                target_successors.remove(self)

    def remove_successor(self, operation: 'Node', st='spatial'):
        # ... (代码不变) ...
        target_successors = self.spatial_successors if st == 'spatial' else self.temporal_successors
        target_predecessors = operation.spatial_predecessors if st == 'spatial' else operation.temporal_predecessors
        if operation in target_successors:
            target_successors.remove(operation)
            if self in target_predecessors:
                target_predecessors.remove(self)

    def clear_connections(self):
        # ... (代码不变) ...
        for pred in list(self.spatial_predecessors): self.remove_predecessor(pred, 'spatial')
        for succ in list(self.spatial_successors): self.remove_successor(succ, 'spatial')
        for pred in list(self.temporal_predecessors): self.remove_predecessor(pred, 'temporal')
        for succ in list(self.temporal_successors): self.remove_successor(succ, 'temporal')
        self.spatial_predecessors.clear()
        self.spatial_successors.clear()
        self.temporal_predecessors.clear()
        self.temporal_successors.clear()

    def update_memory(self):
        # ... (代码不变) ...
        self.last_memory['inputs'] = copy.deepcopy(self.inputs)
        self.last_memory['outputs'] = copy.deepcopy(self.outputs)
        self.last_memory['raw_inputs'] = copy.deepcopy(self.raw_inputs)

    def get_spatial_info(self) -> Dict[str, Dict]:
        # ... (代码不变) ...
        spatial_info = {}
        for predecessor in set(self.spatial_predecessors):
            out = predecessor.outputs
            value = out[-1] if isinstance(out, list) and out else out
            if value is not None:
                spatial_info[predecessor.id] = {"role": getattr(predecessor, 'role', 'Unknown'), "output": copy.deepcopy(value)}
        return spatial_info

    def get_temporal_info(self) -> Dict[str, Dict]:
        # ... (代码不变) ...
        temporal_info = {}
        for predecessor in set(self.temporal_predecessors):
            out = predecessor.last_memory.get('outputs', [])
            value = out[-1] if isinstance(out, list) and out else out
            if value is not None:
                temporal_info[predecessor.id] = {"role": getattr(predecessor, 'role', 'Unknown'), "output": copy.deepcopy(value)}
        return temporal_info

    # <--- CHANGED: 主同步执行方法 ---
    def execute(self, input_dict: Dict, **kwargs):
        """
        同步执行节点的主要入口点。
        """
        print(f"[节点 {self.id} | {self.__class__.__name__}] 进入 execute (同步)", flush=True)
        self.outputs = []
        self.raw_inputs = [copy.deepcopy(input_dict)]

        spatial_info = self.get_spatial_info()
        temporal_info = self.get_temporal_info()

        try:
            print(f"[节点 {self.id}] 调用 _process_inputs...", flush=True)
            processed_input = self._process_inputs(self.raw_inputs, spatial_info, temporal_info, **kwargs)
            self.inputs = processed_input
            
            # 打印 Prompt (与异步版本逻辑相同)
            if isinstance(processed_input, (list, tuple)) and len(processed_input) == 2:
                system_prompt, user_prompt = processed_input
                print(f"\n==========🧠 节点 {self.id} | {self.role or self.agent_name} 的 PROMPT 内容 ==========")
                print(f">>> SYSTEM PROMPT:\n{system_prompt.strip()[:1000]}")
                print(f"\n>>> USER PROMPT:\n{user_prompt.strip()[:3000]}")
                print("==========================================================\n")

        except Exception as e:
            print(f"错误: 节点 {self.id} 在 _process_inputs 时出错: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            self.outputs = ["<Input Processing Error>"]
            return self.outputs

        try:
            # 调用具体的同步执行逻辑
            result = self._execute(self.inputs, spatial_info, temporal_info, **kwargs)
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
            print(f"[节点 {self.id}] 同步执行结果: {result}", flush=True)

        except Exception as e:
            print(f"错误: 节点 {self.id} 在 _execute 时出错: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            self.outputs = ["<Execution Error>"]

        return self.outputs

    # <--- CHANGED: _execute 现在调用新的抽象方法 _run_llm_sync ---
    def _execute(self, processed_inputs: List[Any], spatial_info: Dict[str, Dict],
                 temporal_info: Dict[str, Dict], **kwargs) -> List[Any]:
        """
        具体的同步执行逻辑。它调用一个必须由子类实现的同步 LLM 方法。
        """
        # 子类必须实现 _run_llm_sync
        return self._run_llm_sync(processed_inputs, spatial_info, temporal_info, **kwargs)

    # <--- NEW: 新的同步抽象方法，强制子类实现 ---
    @abstractmethod
    def _run_llm_sync(self, processed_inputs: List[Any], spatial_info: Dict[str, Dict],
                      temporal_info: Dict[str, Dict], **kwargs) -> List[Any]:
        """
        子类必须实现此方法以执行同步的 LLM API 调用。
        """
        raise NotImplementedError

    # === 抽象方法 (_process_inputs 保持不变) ===
    @abstractmethod
    def _process_inputs(self, raw_inputs: List[Any], spatial_info: Dict[str, Dict],
                        temporal_info: Dict[str, Dict], **kwargs) -> List[Any]:
        """
        处理输入数据的抽象方法。必须由子类实现。
        """
        raise NotImplementedError

    # === 异步方法 (保留但标记为在同步模式下不使用) ===
    async def async_execute(self, input_dict: Dict, **kwargs):
        """
        异步执行入口点。在同步工作流中不被调用。
        """
        # ... (此方法的代码保持不变，但你可以添加一个警告) ...
        warnings.warn("async_execute 被调用，但程序可能在同步模式下运行。")
        # ... (原来的 async_execute 代码) ...
        print(f"[节点 {self.id} | {self.__class__.__name__}] 进入 async_execute", flush=True)
        self.outputs = []
        self.raw_inputs = [copy.deepcopy(input_dict)]
        spatial_info = self.get_spatial_info()
        temporal_info = self.get_temporal_info()
        try:
            processed_input = self._process_inputs(self.raw_inputs, spatial_info, temporal_info, **kwargs)
            self.inputs = processed_input
        except Exception as e:
            self.outputs = ["<Input Processing Error>"]
            return self.outputs
        try:
            result = await self._async_execute(self.inputs, spatial_info, temporal_info, **kwargs)
            if result is None: result = ["<None Result>"]
            elif not isinstance(result, list): result = [result]
            self.outputs.extend(result)
        except Exception as e:
            self.outputs = ["<Execution Error>"]
        return self.outputs

    async def _async_execute(self, processed_inputs: List[Any], spatial_info: Dict[str, Dict],
                             temporal_info: Dict[str, Dict], **kwargs):
        """
        异步 LLM 调用。在同步工作流中不被调用。
        子类可以保留此实现以支持双模式。
        """
        # 默认情况下，可以尝试调用同步版本并发出警告
        warnings.warn("_async_execute 未被子类实现，将回退到同步执行。")
        # 为了避免阻塞事件循环，这里不直接调用同步方法，而是返回错误
        return ["<_async_execute not implemented>"]
