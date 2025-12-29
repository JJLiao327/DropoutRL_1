# -*- coding: utf-8 -*-
import shortuuid
import traceback
from typing import List, Any, Optional, Dict
from abc import ABC, abstractmethod
import warnings
import asyncio
import copy  # 导入 copy 模块


class Node(ABC):
    """
    表示图计算框架中的一个处理单元 (适配 RL 环境)。

    此类封装了图中节点的功能，管理与其他节点的连接，处理输入输出，
    并执行指定的操作。它支持空间和时间上的信息传递。
    添加了 reset() 方法和 memory 属性以更好地支持作为 RL 环境的一部分。
    """

    def __init__(self,
                 id: Optional[str] = None,
                 agent_name: str = "",
                 domain: str = "",
                 llm_name: str = ""):
        """
        初始化一个新的 Node 实例。
        """
        self.id: str = id if id is not None else shortuuid.ShortUUID().random(length=4)
        self.agent_name: str = agent_name
        self.domain: str = domain
        self.llm_name: str = llm_name
        self.role: str = ""  # 应在子类中设置

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
        """
        重置节点状态，为新的任务执行做准备。
        不清除连接关系，仅清空状态和记忆。
        """
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

    # 连接管理
    def add_predecessor(self, operation: 'Node', st='spatial'):
        target_predecessors = self.spatial_predecessors if st == 'spatial' else self.temporal_predecessors
        target_successors = operation.spatial_successors if st == 'spatial' else operation.temporal_successors
        if operation not in target_predecessors:
            target_predecessors.append(operation)
            if self not in target_successors:
                target_successors.append(self)

    def add_successor(self, operation: 'Node', st='spatial'):
        target_successors = self.spatial_successors if st == 'spatial' else self.temporal_successors
        target_predecessors = operation.spatial_predecessors if st == 'spatial' else operation.temporal_predecessors
        if operation not in target_successors:
            target_successors.append(operation)
            if self not in target_predecessors:
                target_predecessors.append(self)

    def remove_predecessor(self, operation: 'Node', st='spatial'):
        target_predecessors = self.spatial_predecessors if st == 'spatial' else self.temporal_predecessors
        target_successors = operation.spatial_successors if st == 'spatial' else operation.temporal_successors
        if operation in target_predecessors:
            target_predecessors.remove(operation)
            if self in target_successors:
                target_successors.remove(self)

    def remove_successor(self, operation: 'Node', st='spatial'):
        target_successors = self.spatial_successors if st == 'spatial' else self.temporal_successors
        target_predecessors = operation.spatial_predecessors if st == 'spatial' else operation.temporal_predecessors
        if operation in target_successors:
            target_successors.remove(operation)
            if self in target_predecessors:
                target_predecessors.remove(self)

    def clear_connections(self):
        for pred in list(self.spatial_predecessors):
            self.remove_predecessor(pred, 'spatial')
        for succ in list(self.spatial_successors):
            self.remove_successor(succ, 'spatial')
        for pred in list(self.temporal_predecessors):
            self.remove_predecessor(pred, 'temporal')
        for succ in list(self.temporal_successors):
            self.remove_successor(succ, 'temporal')
        self.spatial_predecessors.clear()
        self.spatial_successors.clear()
        self.temporal_predecessors.clear()
        self.temporal_successors.clear()

    def update_memory(self):
        self.last_memory['inputs'] = copy.deepcopy(self.inputs)
        self.last_memory['outputs'] = copy.deepcopy(self.outputs)
        self.last_memory['raw_inputs'] = copy.deepcopy(self.raw_inputs)

    def get_spatial_info(self) -> Dict[str, Dict]:
        spatial_info = {}
        for predecessor in set(self.spatial_predecessors):
            out = predecessor.outputs
            value = out[-1] if isinstance(out, list) and out else out
            if value is not None:
                spatial_info[predecessor.id] = {
                    "role": getattr(predecessor, 'role', 'Unknown'),
                    "output": copy.deepcopy(value)
                }
        return spatial_info

    def get_temporal_info(self) -> Dict[str, Dict]:
        temporal_info = {}
        for predecessor in set(self.temporal_predecessors):
            out = predecessor.last_memory.get('outputs', [])
            value = out[-1] if isinstance(out, list) and out else out
            if value is not None:
                temporal_info[predecessor.id] = {
                    "role": getattr(predecessor, 'role', 'Unknown'),
                    "output": copy.deepcopy(value)
                }
        return temporal_info

    def execute(self, input_dict: Dict, **kwargs):
        warnings.warn("同步 execute 方法不推荐在异步环境中使用，请使用 async_execute。", DeprecationWarning)
        self.outputs = []
        self.raw_inputs = [copy.deepcopy(input_dict)]

        spatial_info = self.get_spatial_info()
        temporal_info = self.get_temporal_info()

        try:
            processed_input = self._process_inputs(self.raw_inputs, spatial_info, temporal_info, **kwargs)
            self.inputs = processed_input
        except NotImplementedError:
            print(f"警告: 节点 {self.id} 的 _process_inputs 方法未实现。")
            self.inputs = self.raw_inputs
        except Exception as e:
            print(f"错误: 节点 {self.id} 在 _process_inputs 时出错: {e}")
            self.outputs = ["<Input Processing Error>"]
            return self.outputs

        try:
            result = self._execute(self.inputs, spatial_info, temporal_info, **kwargs)
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
        except NotImplementedError:
            print(f"错误: 节点 {self.id} 的 _execute 方法未实现。")
            self.outputs = ["<Not Implemented>"]
        except Exception as e:
            print(f"错误: 节点 {self.id} 在 _execute 时出错: {e}")
            self.outputs = ["<Execution Error>"]

        return self.outputs

    async def async_execute(self, input_dict: Dict, **kwargs):
        print(f"[节点 {self.id} | {self.__class__.__name__}] 进入 async_execute", flush=True)
        self.outputs = []
        self.raw_inputs = [copy.deepcopy(input_dict)]

        spatial_info = self.get_spatial_info()
        temporal_info = self.get_temporal_info()

        try:
            print(f"[节点 {self.id} | {self.__class__.__name__}] 调用 _process_inputs 处理输入...", flush=True)
            processed_input = self._process_inputs(self.raw_inputs, spatial_info, temporal_info, **kwargs)

            if isinstance(processed_input, (list, tuple)) and len(processed_input) == 2:
                system_prompt, user_prompt = processed_input
                print(f"\n==========🧠 节点 {self.id} | {self.role or self.agent_name} 的 PROMPT 内容 ==========")
                print(">>> SYSTEM PROMPT:")
                print(system_prompt.strip()[:1000])
                print("\n>>> USER PROMPT:")
                print(user_prompt.strip()[:3000])
                print("==========================================================\n")
            else:
                print(f"[节点 {self.id}] ⚠️ processed_input 结构不是二元组，无法提取 prompt")

            print(f"[节点 {self.id}] 🔗 空间前驱信息:")
            for aid, info in spatial_info.items():
                print(f"  来自 {aid} ({info['role']}): {str(info['output'])[:200]}")

            print(f"[节点 {self.id}] ⏳ 时间前驱信息:")
            for aid, info in temporal_info.items():
                print(f"  来自 {aid} ({info['role']}): {str(info['output'])[:200]}")

            self.inputs = processed_input
            print(f"[节点 {self.id} | {self.__class__.__name__}] _process_inputs 完成。", flush=True)

        except NotImplementedError:
            print(f"[节点 {self.id} | {self.__class__.__name__}] _process_inputs 未实现。", flush=True)
            self.inputs = self.raw_inputs
        except Exception as e:
            print(f"[节点 {self.id} | {self.__class__.__name__}] _process_inputs 错误: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            self.outputs = ["<Input Processing Error>"]
            return self.outputs

        try:
            print(f"[节点 {self.id}] await _async_execute...", flush=True)
            result = await self._async_execute(self.inputs, spatial_info, temporal_info, **kwargs)

            if result is None:
                print(f"[节点 {self.id}] _async_execute 返回 None。", flush=True)
                result = ["<None Result>"]
            elif not isinstance(result, list):
                result = [result]

            self.outputs.extend(result)
            print(f"[节点 {self.id}] async_execute 执行结果: {result}", flush=True)

        except NotImplementedError:
            print(f"[节点 {self.id}] _async_execute 未实现。", flush=True)
            self.outputs = ["<Not Implemented>"]
        except Exception as e:
            print(f"[节点 {self.id}] _async_execute 异常: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            self.outputs = ["<Execution Error>"]

        return self.outputs

    # === 抽象方法 ===
    @abstractmethod
    def _execute(self, processed_inputs: List[Any], spatial_info: Dict[str, Dict],
                 temporal_info: Dict[str, Dict], **kwargs):
        raise NotImplementedError

    @abstractmethod
    async def _async_execute(self, processed_inputs: List[Any], spatial_info: Dict[str, Dict],
                             temporal_info: Dict[str, Dict], **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _process_inputs(self, raw_inputs: List[Any], spatial_info: Dict[str, Dict],
                        temporal_info: Dict[str, Dict], **kwargs) -> List[Any]:
        raise NotImplementedError
