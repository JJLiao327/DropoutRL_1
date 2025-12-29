# 文件名: SPARCS/graph/graph.py
# 描述: Graph 类的同步实现版本

import shortuuid
from typing import Any, List, Optional, Dict, Union, Tuple
from abc import ABC
import numpy as np
import torch
import time  # <--- CHANGED: Import time for synchronous retries
import traceback
import random
from collections import deque # <--- NEW: Import deque for efficient queue operations

from SPARCS.graph.node import Node
from SPARCS.agents.agent_registry import AgentRegistry

class Graph(ABC):
    """
    一个用于管理和执行节点网络的框架（同步版本）。
    节点连接（空间和时间）通过 set_runtime_masks 方法由外部源提供。
    """

    @staticmethod
    def _ensure_zero_indegree_in_mask(mask: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        确保给定的空间掩码产生的图中至少有一个节点的入度为0。
        直接在提供的掩码上操作。
        """
        if num_nodes == 0:
            return mask
        in_degrees = mask.sum(dim=0)
        if (in_degrees > 0).all():
            node_to_clear_indegree_idx = random.choice(range(num_nodes))
            mask[:, node_to_clear_indegree_idx] = 0
            print(f"INFO: 运行时空间掩码已调整 - 节点索引 {node_to_clear_indegree_idx} 的入度被强制为0。", flush=True)
        return mask

    def __init__(self,
                 domain: str,
                 llm_name: Optional[str],
                 agent_names: List[str],
                 decision_method: str,
                 mode: str = "FullConnected",
                 fixed_spatial_masks: Optional[List[List[int]]] = None,
                 diff: bool = False,
                 rounds: int = 1,
                 fixed_temporal_masks: Optional[List[List[int]]] = None,
                 node_kwargs: Optional[List[Dict]] = None,
                 ):
        # ... (构造函数 __init__ 的所有代码保持不变) ...
        num_agents = len(agent_names)
        self.mode = mode

        if fixed_spatial_masks is None:
            _mask_data = []
            if num_agents > 0:
                if mode == "FullConnected":
                    _mask_data = [[1 if i != j else 0 for j in range(num_agents)] for i in range(num_agents)]
                elif mode == "Chain":
                    _mask_data = [[0] * num_agents for _ in range(num_agents)]
                    if num_agents > 1:
                        for i in range(num_agents - 1):
                            _mask_data[i][i+1] = 1
                elif mode == "Star":
                    _mask_data = [[0] * num_agents for _ in range(num_agents)]
                    if num_agents > 0:
                        for j in range(1, num_agents):
                            _mask_data[0][j] = 1
                elif mode == "Layered":
                    _mask_data = [[0] * num_agents for _ in range(num_agents)]
                    if num_agents > 1:
                        num_layer1 = (num_agents + 1) // 2
                        for i in range(num_layer1):
                            for j in range(num_layer1, num_agents):
                                _mask_data[i][j] = 1
                elif mode == "DirectAnswer" or mode == "Debate":
                    _mask_data = [[0] * num_agents for _ in range(num_agents)]
                else:
                    print(f"警告: 未知的 mode '{mode}', 将使用 FullConnected 作为 fixed_spatial_masks。", flush=True)
                    _mask_data = [[1 if i != j else 0 for j in range(num_agents)] for i in range(num_agents)]
            
            if not _mask_data and num_agents > 0:
                 _mask_data = [[0] * num_agents for _ in range(num_agents)]
            
            self.fixed_spatial_masks = torch.tensor(_mask_data, dtype=torch.float32) if num_agents > 0 else torch.empty((0,0), dtype=torch.float32)
        else:
            self.fixed_spatial_masks = torch.tensor(fixed_spatial_masks, dtype=torch.float32)

        if num_agents > 0:
            assert self.fixed_spatial_masks.shape == (num_agents, num_agents), \
                f"fixed_spatial_masks 形状必须是 ({num_agents}, {num_agents}), 得到 {self.fixed_spatial_masks.shape}"
        elif not (self.fixed_spatial_masks.shape == (0,0) or self.fixed_spatial_masks.numel() == 0) :
             raise ValueError(f"fixed_spatial_masks 形状应为 (0,0) 当 num_agents=0, 得到 {self.fixed_spatial_masks.shape}")

        if fixed_temporal_masks is None:
            self.fixed_temporal_masks = torch.tensor([[1 for _ in range(num_agents)] for _ in range(num_agents)], dtype=torch.float32) if num_agents > 0 else torch.empty((0,0), dtype=torch.float32)
        else:
            self.fixed_temporal_masks = torch.tensor(fixed_temporal_masks, dtype=torch.float32)

        if num_agents > 0:
            assert self.fixed_temporal_masks.shape == (num_agents, num_agents), \
                f"fixed_temporal_masks 形状必须是 ({num_agents}, {num_agents}), 得到 {self.fixed_temporal_masks.shape}"
        elif not (self.fixed_temporal_masks.shape == (0,0) or self.fixed_temporal_masks.numel() == 0):
            raise ValueError(f"fixed_temporal_masks 形状应为 (0,0) 当 num_agents=0, 得到 {self.fixed_temporal_masks.shape}")

        self.id: str = shortuuid.ShortUUID().random(length=4)
        self.domain: str = domain
        self.llm_name: Optional[str] = llm_name
        self.agent_names: List[str] = agent_names

        try:
            self.decision_node: Node = AgentRegistry.get(decision_method, **{"domain": self.domain, "llm_name": self.llm_name})
        except Exception as e:
            print(f"错误: 初始化 Decision Node '{decision_method}' 失败. LLM: {self.llm_name}. 错误: {e}", flush=True)
            raise

        self.nodes: Dict[str, Node] = {}
        self.potential_spatial_edges: List[Tuple[str, str]] = []
        self.potential_temporal_edges: List[Tuple[str, str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        self.diff: bool = diff
        self.rounds: int = rounds
        self.runtime_spatial_masks: Union[torch.Tensor, List[torch.Tensor], None] = None
        self.runtime_temporal_masks: Union[torch.Tensor, List[torch.Tensor], None] = None

        self.init_nodes()
        self.init_potential_edges()

        print(f"Graph {self.id} 初始化完成 (模式: {self.mode})，包含 {self.num_nodes} 个 Agent。", flush=True)
        print(f"  Fixed Spatial Mask (初始/默认):\n{self.fixed_spatial_masks}", flush=True)
        print(f"  潜在空间边数量: {len(self.potential_spatial_edges)}", flush=True)
        print(f"  潜在时间边数量: {len(self.potential_temporal_edges)}", flush=True)
        print(f"  使用 Diff 模式 (每轮不同 Mask): {self.diff}", flush=True)
        print(f"  交互轮数: {self.rounds}", flush=True)

    # --- Properties (保持不变) ---
    @property
    def spatial_adj_matrix(self) -> np.ndarray:
        # ... (代码不变) ...
        node_list = list(self.nodes.keys())
        node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}
        matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i, node1_id in enumerate(node_list):
            node1 = self.nodes[node1_id]
            for successor in node1.spatial_successors:
                if successor.id in node_to_idx:
                    j = node_to_idx[successor.id]
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self) -> np.ndarray:
        # ... (代码不变) ...
        node_list = list(self.nodes.keys())
        node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}
        matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i, node1_id in enumerate(node_list):
            node1 = self.nodes[node1_id]
            for successor in node1.temporal_successors:
                 if successor.id in node_to_idx:
                    j = node_to_idx[successor.id]
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self) -> int:
        # ... (代码不变) ...
        num_edges = 0
        for node in self.nodes.values():
            num_edges += sum(1 for succ in node.spatial_successors if succ.id in self.nodes)
            num_edges += sum(1 for succ in node.temporal_successors if succ.id in self.nodes)
        return num_edges

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    # --- Helper, Init, Connection, Cycle Check Methods (所有这些都保持不变) ---
    def find_node(self, node_id: str) -> Node:
        # ... (代码不变) ...
        if node_id in self.nodes:
            return self.nodes[node_id]
        if hasattr(self, 'decision_node') and self.decision_node.id == node_id:
            return self.decision_node
        raise KeyError(f"节点未找到: {node_id}")

    def add_node(self, node: Node) -> Node:
        # ... (代码不变) ...
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes or (hasattr(self, 'decision_node') and node_id == self.decision_node.id):
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node
        
    def init_nodes(self):
        # ... (代码不变) ...
        self.nodes.clear()
        for agent_name, kwargs in zip(self.agent_names, self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                full_kwargs = {"domain": self.domain, "llm_name": self.llm_name, **kwargs}
                try:
                    agent_instance = AgentRegistry.get(agent_name, **full_kwargs)
                    self.add_node(agent_instance)
                except Exception as e:
                    print(f"错误：初始化 Agent '{agent_name}' 失败，参数: {full_kwargs}。错误信息: {e}", flush=True)
                    raise
            else:
                print(f"警告：Agent 类型 '{agent_name}' 未在 AgentRegistry 中注册。", flush=True)

    def init_potential_edges(self):
        # ... (代码不变) ...
        self.potential_spatial_edges.clear()
        self.potential_temporal_edges.clear()
        node_ids = list(self.nodes.keys())
        for node1_id in node_ids:
            for node2_id in node_ids:
                self.potential_spatial_edges.append((node1_id, node2_id))
                self.potential_temporal_edges.append((node1_id, node2_id))

    def clear_spatial_connection(self):
        # ... (代码不变) ...
        for node in self.nodes.values():
            node.spatial_predecessors.clear()
            node.spatial_successors.clear()
        if hasattr(self, 'decision_node'):
            self.decision_node.spatial_predecessors.clear()
            self.decision_node.spatial_successors.clear()

    def clear_temporal_connection(self):
        # ... (代码不变) ...
        for node in self.nodes.values():
            node.temporal_predecessors.clear()
            node.temporal_successors.clear()

    def connect_decision_node(self):
        # ... (代码不变) ...
        if not hasattr(self, 'decision_node'):
            print("警告: 尝试连接决策节点，但 decision_node 未初始化。", flush=True)
            return
        self.decision_node.spatial_predecessors.clear()
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node, st='spatial')

    def check_cycle(self, start_node: Node, visited: set, current_path: set) -> bool:
        # ... (代码不变) ...
        visited.add(start_node.id)
        current_path.add(start_node.id)
        for successor in start_node.spatial_successors:
            if successor.id not in visited:
                if self.check_cycle(successor, visited, current_path):
                    return True
            elif successor.id in current_path:
                return True
        current_path.remove(start_node.id)
        return False

    def has_spatial_cycle(self) -> bool:
        # ... (代码不变) ...
        visited = set()
        current_path = set()
        for node_id in self.nodes:
            if node_id not in visited:
                if self.check_cycle(self.nodes[node_id], visited, current_path):
                    return True
        return False

    # --- Mask and Connection Methods (所有这些都保持不变) ---
    def set_runtime_masks(self, spatial_masks, temporal_masks):
        # ... (代码不变) ...
        N = self.num_nodes
        if N == 0:
            self.runtime_spatial_masks = torch.empty((0,0), dtype=torch.float32) if not self.diff else []
            self.runtime_temporal_masks = torch.empty((0,0), dtype=torch.float32) if not self.diff else []
            return
        if self.diff:
            assert isinstance(spatial_masks, list) and len(spatial_masks) == self.rounds
            processed_masks_list = []
            for i, mask_tensor in enumerate(spatial_masks):
                cloned_mask = mask_tensor.clone()
                processed_mask = Graph._ensure_zero_indegree_in_mask(cloned_mask, N)
                processed_masks_list.append(processed_mask)
            self.runtime_spatial_masks = processed_masks_list
        else:
            cloned_mask = spatial_masks.clone()
            processed_mask = Graph._ensure_zero_indegree_in_mask(cloned_mask, N)
            self.runtime_spatial_masks = processed_mask
        
        expected_temporal_len = self.rounds - 1 if self.rounds > 0 else 0
        if expected_temporal_len < 0: expected_temporal_len = 0
        
        if self.diff:
            if expected_temporal_len > 0:
                assert isinstance(temporal_masks, list) and len(temporal_masks) == expected_temporal_len
                self.runtime_temporal_masks = [mask.clone() for mask in temporal_masks]
            else:
                self.runtime_temporal_masks = []
        else:
            self.runtime_temporal_masks = temporal_masks.clone()

    def construct_spatial_connection(self, round_idx: int = 0):
        # ... (代码不变) ...
        self.clear_spatial_connection()
        current_mask: Optional[torch.Tensor] = None

        if self.runtime_spatial_masks is None:
            print("警告: runtime_spatial_masks 未设置，使用固定的初始空间掩码。", flush=True)
            current_mask = self.fixed_spatial_masks
        elif self.diff:
            if isinstance(self.runtime_spatial_masks, list) and round_idx < len(self.runtime_spatial_masks):
                 current_mask = self.runtime_spatial_masks[round_idx]
            else:
                current_mask = self.fixed_spatial_masks
        else:
            current_mask = self.runtime_spatial_masks

        if current_mask is None or self.num_nodes == 0: return

        node_list = list(self.nodes.keys())
        for i, out_node_id in enumerate(node_list):
            for j, in_node_id in enumerate(node_list):
                if current_mask[i, j] == 1:
                    out_node = self.find_node(out_node_id)
                    in_node = self.find_node(in_node_id)
                    if out_node != in_node:
                        out_node.add_successor(in_node, 'spatial')

    def construct_temporal_connection(self, round_idx: int = 0):
        # ... (代码不变) ...
        self.clear_temporal_connection()
        if round_idx == 0: return

        mask_idx = round_idx - 1
        current_mask: Optional[torch.Tensor] = None

        if self.runtime_temporal_masks is None:
            current_mask = self.fixed_temporal_masks
        elif self.diff:
            if isinstance(self.runtime_temporal_masks, list) and mask_idx < len(self.runtime_temporal_masks):
                 current_mask = self.runtime_temporal_masks[mask_idx]
            else:
                current_mask = self.fixed_temporal_masks
        else:
            current_mask = self.runtime_temporal_masks
        
        if current_mask is None or self.num_nodes == 0: return

        node_list = list(self.nodes.keys())
        for i, out_node_id in enumerate(node_list):
            for j, in_node_id in enumerate(node_list):
                if current_mask[i, j] == 1:
                    out_node = self.find_node(out_node_id)
                    in_node = self.find_node(in_node_id)
                    out_node.add_successor(in_node, 'temporal')


    # <--- CHANGED: arun() is now run() and is fully synchronous ---
    def run(self, input_dict: Dict[str, str],
            num_rounds: Optional[int] = None,
            max_tries: int = 3,
            max_time: int = 6000,
            ) -> Tuple[List[Any], List[Dict[str, List[str]]]]:
        """
        同步执行图。
        """
        if self.runtime_spatial_masks is None or self.runtime_temporal_masks is None:
             print("警告: 运行时 Masks 未通过 set_runtime_masks() 设置，将使用固定的初始 Masks。", flush=True)
             self.set_runtime_masks(self.fixed_spatial_masks, self.fixed_temporal_masks)

        current_rounds = num_rounds if num_rounds is not None else self.rounds
        all_answers: List[Dict[str, List[str]]] = []

        for node in self.nodes.values():
            node.reset()
        if hasattr(self, 'decision_node'):
            self.decision_node.reset()

        for round_idx in range(current_rounds):
            print(f"[图 {self.id}] --- 开始执行第 {round_idx + 1}/{current_rounds} 轮 ---", flush=True)
            
            print(f"[图 {self.id} | 轮次 {round_idx+1}] 构建空间和时间连接...", flush=True)
            self.construct_spatial_connection(round_idx)
            self.construct_temporal_connection(round_idx)

            print(f"[图 {self.id} | 轮次 {round_idx+1}] 开始拓扑排序执行...", flush=True)
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            # Use deque for efficient pop from the left
            queue = deque([node_id for node_id, deg in in_degree.items() if deg == 0])
            
            print(f"[图 {self.id} | 轮次 {round_idx+1}] 计算得到的初始入度: {in_degree}", flush=True)
            print(f"[图 {self.id} | 轮次 {round_idx+1}] 初始零入度队列: {list(queue)}", flush=True)
            
            completed_node_ids_this_round = set()

            while queue:
                node_id = queue.popleft()
                node = self.nodes[node_id]
                
                print(f"[图 {self.id} | 轮次 {round_idx+1}] 执行节点 {node_id}", flush=True)
                try:
                    # This is now a blocking call
                    self.execute_node_with_retry(node, input_dict, max_tries, max_time)
                    completed_node_ids_this_round.add(node_id)
                except Exception as e:
                    print(f"错误: [图 {self.id} | 轮次 {round_idx+1}] 节点 {node_id} 执行最终失败: {e}", flush=True)
                    if node_id in self.nodes:
                        self.nodes[node_id].outputs = [f"<Execution Error: {type(e).__name__}>"]

                # Update successors' in-degrees
                for successor in node.spatial_successors:
                    if successor.id in self.nodes and successor.id in in_degree:
                        in_degree[successor.id] -= 1
                        print(f"  - 后继节点 {successor.id} 新入度: {in_degree[successor.id]}", flush=True)
                        if in_degree[successor.id] == 0:
                            print(f"  - 后继节点 {successor.id} 入度为 0，加入队列。", flush=True)
                            queue.append(successor.id)

            # --- Round finished, collect answers and update memory ---
            all_graph_node_ids = set(self.nodes.keys())
            if completed_node_ids_this_round != all_graph_node_ids:
                 uncompleted_nodes = all_graph_node_ids - completed_node_ids_this_round
                 print(f"警告: 轮次 {round_idx+1} 结束时，并非所有节点都成功执行 ({len(completed_node_ids_this_round)}/{self.num_nodes})。", flush=True)
                 print(f"  未完成/处理的节点: {list(uncompleted_nodes)}", flush=True)

            round_answers = {f"{node.role}_{node_id}": node.outputs[:] if node.outputs else ["<Not Executed or No Output>"] for node_id, node in self.nodes.items()}
            all_answers.append(round_answers)

            print(f"[图 {self.id} | 轮次 {round_idx+1}] 更新所有节点 Memory...", flush=True)
            self.update_memory()
            print(f"[图 {self.id}] --- 第 {round_idx + 1} 轮结束 ---", flush=True)

        # --- All rounds finished, execute decision node ---
        final_answers = ["<Decision Node Not Executed or No Output>"]
        if hasattr(self, 'decision_node') and self.decision_node:
            try:
                print(f"[图 {self.id}] --- 开始执行决策节点 {self.decision_node.id} ---", flush=True)
                self.connect_decision_node()
                self.execute_node_with_retry(self.decision_node, input_dict, max_tries, max_time)
                final_answers = self.decision_node.outputs[:] or ["<No Answer from Decision Node>"]
                print(f"[图 {self.id}] --- 决策节点 {self.decision_node.id} 执行完毕, 输出: {final_answers} ---", flush=True)
            except Exception as e:
                print(f"错误: 决策节点 {self.decision_node.id} 执行失败: {e}", flush=True)
                final_answers = [f"<Decision Error: {type(e).__name__}>"]
        else:
             print("警告: 未配置决策节点或决策节点为 None。", flush=True)
             # Fallback logic remains the same
             if self.nodes:
                 last_node_key = list(self.nodes.keys())[-1]
                 final_answers = self.nodes[last_node_key].outputs[:] if self.nodes[last_node_key].outputs else ["<No Decision Node, Last Agent No Output>"]
             else:
                 final_answers = ["<No Decision Node and No Agents>"]

        print(f"[图 {self.id}] run 方法执行完毕，返回最终答案和所有轮次答案。", flush=True)
        return final_answers, all_answers

    # <--- CHANGED: execute_node_with_retry is now synchronous ---
    def execute_node_with_retry(self, node: Node, input_dict: Dict, max_tries: int, timeout: int):
        """
        同步执行一个节点，并带有重试逻辑。
        注意: 'timeout' 在同步模式下难以精确实现，这里它将不被使用。
        你需要确保你的底层同步API调用有自己的超时设置。
        """
        tries = 0
        last_exception = None
        while tries < max_tries:
            tries += 1
            try:
                print(f"[图 {self.id} | 节点 {node.id}] 第 {tries}/{max_tries} 次尝试执行...", flush=True)
                # 假设你的 Node 类有一个同步的 execute 方法
                # 如果没有，你需要创建它，例如: def execute(self, input_dict): ...
                node.execute(input_dict) 
                print(f"[图 {self.id} | 节点 {node.id}] 第 {tries} 次尝试成功。", flush=True)
                return
            except Exception as e:
                last_exception = e
                print(f"警告: [图 {self.id} | 节点 {node.id}] 第 {tries}/{max_tries} 次尝试执行出错: {e}", flush=True)
            
            if tries < max_tries:
                 time.sleep(0.1 * tries) # 同步的延迟

        print(f"错误: [图 {self.id} | 节点 {node.id}] 在 {max_tries} 次尝试后执行失败。最后错误: {last_exception}", flush=True)
        raise last_exception if last_exception else RuntimeError(f"节点 {node.id} 执行失败且无明确异常")

    # --- Other methods (update_memory, get_agent_states, etc.) remain unchanged ---
    def update_memory(self):
        for node in self.nodes.values():
            node.update_memory()

    def get_agent_states(self) -> Dict[str, Dict[str, Any]]:
        states = {}
        for node_id, node in self.nodes.items():
            states[node_id] = {
                "outputs": node.outputs[:],
                "memory": copy.deepcopy(node.memory) if hasattr(node, 'memory') and node.memory is not None else None,
                "role": node.role if hasattr(node, 'role') else 'Unknown'
            }
        return states

    def get_current_masks(self) -> Tuple[Union[torch.Tensor, List[torch.Tensor], None], Union[torch.Tensor, List[torch.Tensor], None]]:
        return self.runtime_spatial_masks, self.runtime_temporal_masks

    def get_intermediate_results(self, all_answers_from_arun: List[Dict[str, List[str]]]) -> Any:
        return all_answers_from_arun

