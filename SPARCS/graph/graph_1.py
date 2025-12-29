# -*- coding: utf-8 -*-
import shortuuid
from typing import Any, List, Optional, Dict, Union, Tuple
from abc import ABC
import numpy as np
import torch
import asyncio
import copy
import traceback
import random

from SPARCS.graph.node import Node  # Assuming Node class exists
from SPARCS.agents.agent_registry import AgentRegistry # Assuming AgentRegistry exists

class Graph(ABC):
    """
    A framework for managing and executing a network of nodes (adapted for RL control).
    Nodes connections (spatial and temporal) are provided by an external source
    (e.g., RL Agent) via set_runtime_masks method.
    """

    @staticmethod
    def _ensure_zero_indegree_in_mask(mask: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Ensures that the given spatial mask results in a graph
        where at least one node has an in-degree of 0.
        Operates on the provided mask (which should be a clone if original needs preservation).
        """
        if num_nodes == 0:
            return mask

        # Calculate in-degrees: mask[sender, receiver]=1 means sender->receiver.
        # In-degree of node j is sum of column j: mask[:, j].
        in_degrees = mask.sum(dim=0)

        if (in_degrees > 0).all(): # If all nodes have in-degree > 0
            node_to_clear_indegree_idx = random.choice(range(num_nodes))
            mask[:, node_to_clear_indegree_idx] = 0 # Set all incoming edges to this node to 0
            print(f"INFO: Runtime spatial mask adjusted - Node index {node_to_clear_indegree_idx} in-degree forced to 0.", flush=True)
            # For verification:
            # new_in_degrees = mask.sum(dim=0)
            # print(f"  Adjusted in-degrees: {new_in_degrees}", flush=True)
        return mask

    def __init__(self,
                 domain: str,
                 llm_name: Optional[str],
                 agent_names: List[str],
                 decision_method: str,
                 mode: str = "FullConnected", # ADDED mode parameter
                 fixed_spatial_masks: Optional[List[List[int]]] = None, # Kept for explicit override
                 diff: bool = False,
                 rounds: int = 1,
                 fixed_temporal_masks: Optional[List[List[int]]] = None,
                 node_kwargs: Optional[List[Dict]] = None,
                 ):
        """
        Initializes the Graph object.
        """
        num_agents = len(agent_names)
        self.mode = mode # Store the mode

        # --- Initialize fixed_spatial_masks based on mode or provided argument ---
        if fixed_spatial_masks is None:
            _mask_data = []
            if num_agents > 0:
                if mode == "FullConnected":
                    _mask_data = [[1 if i != j else 0 for j in range(num_agents)] for i in range(num_agents)]
                elif mode == "Chain":
                    _mask_data = [[0] * num_agents for _ in range(num_agents)]
                    if num_agents > 1:
                        for i in range(num_agents - 1):
                            _mask_data[i][i+1] = 1 # Node i sends to Node i+1
                elif mode == "Star": # Node 0 is center, sends to others
                    _mask_data = [[0] * num_agents for _ in range(num_agents)]
                    if num_agents > 0: # Ensure Node 0 exists
                        for j in range(1, num_agents): # Node 0 sends to Node j (1 to N-1)
                            _mask_data[0][j] = 1
                elif mode == "Layered": # Simple 2-layer: first ceil(N/2) send to rest
                    _mask_data = [[0] * num_agents for _ in range(num_agents)]
                    if num_agents > 1:
                        num_layer1 = (num_agents + 1) // 2
                        for i in range(num_layer1):
                            for j in range(num_layer1, num_agents):
                                _mask_data[i][j] = 1
                elif mode == "DirectAnswer" or mode == "Debate": # No spatial connections
                    _mask_data = [[0] * num_agents for _ in range(num_agents)]
                else:
                    print(f"警告: 未知的 mode '{mode}', 将使用 FullConnected 作为 fixed_spatial_masks。", flush=True)
                    _mask_data = [[1 if i != j else 0 for j in range(num_agents)] for i in range(num_agents)]
            # Ensure _mask_data is correctly sized even for num_agents=0 or 1 after logic
            if not _mask_data and num_agents > 0:
                 _mask_data = [[0] * num_agents for _ in range(num_agents)] # Default to no connections if logic failed for N>0
            
            self.fixed_spatial_masks = torch.tensor(_mask_data, dtype=torch.float32) if num_agents > 0 else torch.empty((0,0), dtype=torch.float32)

        else: # User provided an explicit fixed_spatial_masks
            self.fixed_spatial_masks = torch.tensor(fixed_spatial_masks, dtype=torch.float32)

        if num_agents > 0:
            assert self.fixed_spatial_masks.shape == (num_agents, num_agents), \
                f"fixed_spatial_masks 形状必须是 ({num_agents}, {num_agents}), 得到 {self.fixed_spatial_masks.shape}"
        elif not (self.fixed_spatial_masks.shape == (0,0) or self.fixed_spatial_masks.numel() == 0) : # For num_agents = 0
             raise ValueError(f"fixed_spatial_masks 形状应为 (0,0) 当 num_agents=0, 得到 {self.fixed_spatial_masks.shape}")


        # --- Initialize fixed_temporal_masks ---
        if fixed_temporal_masks is None:
            self.fixed_temporal_masks = torch.tensor([[1 for _ in range(num_agents)] for _ in range(num_agents)], dtype=torch.float32) if num_agents > 0 else torch.empty((0,0), dtype=torch.float32)
        else:
            self.fixed_temporal_masks = torch.tensor(fixed_temporal_masks, dtype=torch.float32)

        if num_agents > 0:
            assert self.fixed_temporal_masks.shape == (num_agents, num_agents), \
                f"fixed_temporal_masks 形状必须是 ({num_agents}, {num_agents}), 得到 {self.fixed_temporal_masks.shape}"
        elif not (self.fixed_temporal_masks.shape == (0,0) or self.fixed_temporal_masks.numel() == 0): # For num_agents = 0
            raise ValueError(f"fixed_temporal_masks 形状应为 (0,0) 当 num_agents=0, 得到 {self.fixed_temporal_masks.shape}")


        self.id: str = shortuuid.ShortUUID().random(length=4)
        self.domain: str = domain
        self.llm_name: Optional[str] = llm_name # Retain Optional
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
        num_edges = 0
        for node in self.nodes.values():
            num_edges += sum(1 for succ in node.spatial_successors if succ.id in self.nodes)
            num_edges += sum(1 for succ in node.temporal_successors if succ.id in self.nodes)
        return num_edges

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    # --- Helper Methods (保持不变) ---
    def find_node(self, node_id: str) -> Node:
        if node_id in self.nodes:
            return self.nodes[node_id]
        if hasattr(self, 'decision_node') and self.decision_node.id == node_id:
            return self.decision_node
        raise KeyError(f"节点未找到: {node_id} 在 {[node.id for node in self.nodes.values()] + ([self.decision_node.id] if hasattr(self, 'decision_node') else [])} 中")

    def add_node(self, node: Node) -> Node:
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes or (hasattr(self, 'decision_node') and node_id == self.decision_node.id):
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node

    def init_nodes(self):
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
        self.potential_spatial_edges.clear()
        self.potential_temporal_edges.clear()
        node_ids = list(self.nodes.keys())
        for node1_id in node_ids:
            for node2_id in node_ids:
                self.potential_spatial_edges.append((node1_id, node2_id))
                self.potential_temporal_edges.append((node1_id, node2_id))

    def clear_spatial_connection(self):
        for node in self.nodes.values():
            node.spatial_predecessors.clear()
            node.spatial_successors.clear()
        if hasattr(self, 'decision_node'):
            self.decision_node.spatial_predecessors.clear()
            self.decision_node.spatial_successors.clear()

    def clear_temporal_connection(self):
        for node in self.nodes.values():
            node.temporal_predecessors.clear()
            node.temporal_successors.clear()

    def connect_decision_node(self):
        if not hasattr(self, 'decision_node'):
            print("警告: 尝试连接决策节点，但 decision_node 未初始化。", flush=True)
            return
        self.decision_node.spatial_predecessors.clear() # Clear old ones first
        for node_id in self.nodes.keys(): # Agent nodes connect to decision node
            self.nodes[node_id].add_successor(self.decision_node, st='spatial')

    # --- check_cycle 和 has_spatial_cycle (保持不变) ---
    def check_cycle(self, start_node: Node, visited: set, current_path: set) -> bool:
        visited.add(start_node.id)
        current_path.add(start_node.id)
        for successor in start_node.spatial_successors:
            if successor.id not in visited:
                if self.check_cycle(successor, visited, current_path):
                    return True
            elif successor.id in current_path: # Cycle detected
                # print(f"  Cycle detected: path {current_path}, successor {successor.id} already in path", flush=True)
                return True
        current_path.remove(start_node.id)
        return False

    def has_spatial_cycle(self) -> bool:
        visited = set()
        current_path = set()
        for node_id in self.nodes:
            if node_id not in visited:
                # print(f"Checking cycle starting from {node_id}", flush=True)
                if self.check_cycle(self.nodes[node_id], visited, current_path):
                    # print(f"Spatial cycle detected involving node {node_id} or its successors.", flush=True)
                    return True
        # print("No spatial cycles detected.", flush=True)
        return False

    # --- RL 接口方法 (MODIFIED set_runtime_masks) ---
    def set_runtime_masks(self,
                          spatial_masks: Union[torch.Tensor, List[torch.Tensor]],
                          temporal_masks: Union[torch.Tensor, List[torch.Tensor]]):
        N = self.num_nodes
        if N == 0: # Handle case with no agent nodes
            self.runtime_spatial_masks = torch.empty((0,0), dtype=torch.float32) if not self.diff else []
            self.runtime_temporal_masks = torch.empty((0,0), dtype=torch.float32) if not self.diff else []
            return

        # --- Spatial Masks ---
        if self.diff:
            assert isinstance(spatial_masks, list) and len(spatial_masks) == self.rounds, \
                f"Diff 模式下 spatial_masks 必须是长度为 {self.rounds} 的列表, 得到 {len(spatial_masks) if isinstance(spatial_masks, list) else '非列表'}"
            processed_masks_list = []
            for i, mask_tensor in enumerate(spatial_masks):
                assert isinstance(mask_tensor, torch.Tensor) and mask_tensor.shape == (N, N), \
                    f"Diff 模式下 spatial_masks[{i}] 形状必须是 ({N}, {N}), 得到 {mask_tensor.shape}"
                assert ((mask_tensor == 0) | (mask_tensor == 1)).all(), f"spatial_masks[{i}] 必须是 0/1 掩码"
                cloned_mask = mask_tensor.clone()
                processed_mask = Graph._ensure_zero_indegree_in_mask(cloned_mask, N) # Apply fix
                processed_masks_list.append(processed_mask)
            self.runtime_spatial_masks = processed_masks_list
        else: # Not diff
            assert isinstance(spatial_masks, torch.Tensor) and spatial_masks.shape == (N, N), \
                f"非 Diff 模式下 spatial_masks 形状必须是 ({N}, {N}), 得到 {spatial_masks.shape}"
            assert ((spatial_masks == 0) | (spatial_masks == 1)).all(), "spatial_masks 必须是 0/1 掩码"
            cloned_mask = spatial_masks.clone()
            processed_mask = Graph._ensure_zero_indegree_in_mask(cloned_mask, N) # Apply fix
            self.runtime_spatial_masks = processed_mask

        # --- Temporal Masks ---
        expected_temporal_len = self.rounds - 1 if self.rounds > 0 else 0
        if expected_temporal_len < 0: expected_temporal_len = 0

        if self.diff:
            if expected_temporal_len > 0:
                assert isinstance(temporal_masks, list) and len(temporal_masks) == expected_temporal_len, \
                    f"Diff 模式下 temporal_masks 必须是长度为 {expected_temporal_len} 的列表, 得到 {len(temporal_masks) if isinstance(temporal_masks, list) else '非列表'}"
                self.runtime_temporal_masks = [mask.clone() for mask in temporal_masks] # Assuming temporal masks are valid
                for i, mask_t in enumerate(self.runtime_temporal_masks):
                     assert isinstance(mask_t, torch.Tensor) and mask_t.shape == (N, N), \
                        f"Diff 模式下 temporal_masks[{i}] 形状必须是 ({N}, {N})"
                     assert ((mask_t == 0) | (mask_t == 1)).all(), f"temporal_masks[{i}] 必须是 0/1 掩码"
            else: # No temporal masks expected (e.g., rounds=1)
                assert isinstance(temporal_masks, list) and len(temporal_masks) == 0, \
                    f"Diff 模式下，当 rounds <= 1 (exp_len={expected_temporal_len}) 时，temporal_masks 应为空列表, 得到 {temporal_masks}"
                self.runtime_temporal_masks = []
        else: # Not diff
            assert isinstance(temporal_masks, torch.Tensor) and temporal_masks.shape == (N, N), \
                f"非 Diff 模式下 temporal_masks 形状必须是 ({N}, {N}), 得到 {temporal_masks.shape}"
            assert ((temporal_masks == 0) | (temporal_masks == 1)).all(), "temporal_masks 必须是 0/1 掩码"
            self.runtime_temporal_masks = temporal_masks.clone()


    # --- 连接构建方法 (保持不变) ---
    def construct_spatial_connection(self, round_idx: int = 0):
        self.clear_spatial_connection()
        current_mask: Optional[torch.Tensor] = None

        if self.runtime_spatial_masks is None:
            print("警告: runtime_spatial_masks 未设置，使用固定的初始空间掩码。", flush=True)
            current_mask = self.fixed_spatial_masks # This is already a tensor
        elif self.diff:
            if isinstance(self.runtime_spatial_masks, list) and round_idx < len(self.runtime_spatial_masks):
                 current_mask = self.runtime_spatial_masks[round_idx]
            else:
                print(f"警告: Diff 模式下无法获取轮次 {round_idx} 的空间掩码 (列表长度 {len(self.runtime_spatial_masks) if isinstance(self.runtime_spatial_masks, list) else 'N/A'})，使用固定掩码。", flush=True)
                current_mask = self.fixed_spatial_masks
        else: # Not diff, runtime_spatial_masks should be a Tensor
            current_mask = self.runtime_spatial_masks

        if current_mask is None or not isinstance(current_mask, torch.Tensor) or current_mask.numel() == 0 and self.num_nodes > 0:
            print(f"错误/警告: construct_spatial_connection 中 current_mask 无效或为空 (对于 {self.num_nodes} 节点)。将不构建连接。", flush=True)
            # Fallback to fixed_spatial_masks if current_mask became None unexpectedly and nodes exist
            if self.num_nodes > 0 and (current_mask is None or current_mask.numel() == 0):
                print(f"  回退到 fixed_spatial_masks。", flush=True)
                current_mask = self.fixed_spatial_masks
            elif self.num_nodes == 0: # No nodes, no connections
                return


        assert current_mask is not None and isinstance(current_mask, torch.Tensor), \
            f"current_mask 必须是 Tensor, 得到 {type(current_mask)}"
        
        # Ensure mask shape matches num_nodes, unless num_nodes is 0
        if self.num_nodes > 0:
            assert current_mask.shape == (self.num_nodes, self.num_nodes), \
                f"current_mask 形状 ({current_mask.shape}) 与节点数 ({self.num_nodes}) 不匹配"

        node_list = list(self.nodes.keys())
        for i, out_node_id in enumerate(node_list):
            for j, in_node_id in enumerate(node_list):
                if current_mask[i, j] == 1: # out_node_id (i) sends to in_node_id (j)
                    out_node = self.find_node(out_node_id)
                    in_node = self.find_node(in_node_id)
                    if out_node != in_node: # Avoid self-loops from mask if any
                        out_node.add_successor(in_node, 'spatial')


    def construct_temporal_connection(self, round_idx: int = 0):
        self.clear_temporal_connection()
        if round_idx == 0: # No temporal connections before the first round's state exists
            return

        mask_idx = round_idx - 1 # Temporal mask connects previous round state to current round
        current_mask: Optional[torch.Tensor] = None

        if self.runtime_temporal_masks is None:
            print("警告: runtime_temporal_masks 未设置，使用固定的初始时间掩码。", flush=True)
            current_mask = self.fixed_temporal_masks
        elif self.diff:
            if isinstance(self.runtime_temporal_masks, list) and mask_idx < len(self.runtime_temporal_masks):
                 current_mask = self.runtime_temporal_masks[mask_idx]
            else:
                print(f"警告: Diff 模式下无法获取轮次 {round_idx} (mask index {mask_idx}) 的时间掩码 (列表长度 {len(self.runtime_temporal_masks) if isinstance(self.runtime_temporal_masks, list) else 'N/A'})，使用固定掩码。", flush=True)
                current_mask = self.fixed_temporal_masks
        else: # Not diff
            current_mask = self.runtime_temporal_masks
        
        if current_mask is None or not isinstance(current_mask, torch.Tensor) or current_mask.numel() == 0 and self.num_nodes > 0:
            print(f"错误/警告: construct_temporal_connection 中 current_mask 无效或为空 (对于 {self.num_nodes} 节点)。将不构建连接。", flush=True)
            if self.num_nodes > 0 and (current_mask is None or current_mask.numel() == 0):
                print(f"  回退到 fixed_temporal_masks。", flush=True)
                current_mask = self.fixed_temporal_masks
            elif self.num_nodes == 0:
                return

        assert current_mask is not None and isinstance(current_mask, torch.Tensor), \
            f"current_mask 必须是 Tensor, 得到 {type(current_mask)}"
        if self.num_nodes > 0:
             assert current_mask.shape == (self.num_nodes, self.num_nodes), \
                f"current_mask 形状 ({current_mask.shape}) 与节点数 ({self.num_nodes}) 不匹配"

        node_list = list(self.nodes.keys())
        for i, out_node_id in enumerate(node_list):
            for j, in_node_id in enumerate(node_list):
                if current_mask[i, j] == 1:
                    out_node = self.find_node(out_node_id)
                    in_node = self.find_node(in_node_id)
                    # Temporal connections can be self-loops (node's own memory from prev round)
                    out_node.add_successor(in_node, 'temporal')


    # --- Asynchronous Execution Method (arun - with your extensive logging) ---
    async def arun(self, input_dict: Dict[str, str],
                   num_rounds: Optional[int] = None,
                   max_tries: int = 3,
                   max_time: int = 6000, # Assuming this is in seconds as per your comment in execute_node
                   ) -> Tuple[List[Any], List[Dict[str, List[str]]]]:
        if self.runtime_spatial_masks is None or self.runtime_temporal_masks is None:
             print("警告: 运行时 Masks 未通过 set_runtime_masks() 设置，将使用固定的初始 Masks。", flush=True)
             # This call will now pass fixed_spatial_masks (based on mode)
             # through _ensure_zero_indegree_in_mask logic.
             self.set_runtime_masks(self.fixed_spatial_masks, self.fixed_temporal_masks)

        current_rounds = num_rounds if num_rounds is not None else self.rounds
        all_answers: List[Dict[str, List[str]]] = []

        for node in self.nodes.values():
            node.reset()
        if hasattr(self, 'decision_node'):
            self.decision_node.reset()

        for round_idx in range(current_rounds):
            print(f"[图 {self.id}] --- 开始执行第 {round_idx + 1}/{current_rounds} 轮 ---", flush=True)
            round_answers = {}
            completed_node_ids_this_round = set()

            print(f"[图 {self.id} | 轮次 {round_idx+1}] 构建空间连接...", flush=True)
            self.construct_spatial_connection(round_idx)
            # Optional: Check for cycles after construction if PPO can create them
            # if self.has_spatial_cycle():
            #     print(f"警告: [图 {self.id} | 轮次 {round_idx+1}] 检测到空间环! 执行可能无法按预期进行。", flush=True)
            #     # Decide on behavior: stop, try to break, or proceed with caution
            #     # For now, PPO should learn acyclic graphs or _ensure_zero_indegree helps DAGs.

            print(f"[图 {self.id} | 轮次 {round_idx+1}] 构建时间连接...", flush=True)
            self.construct_temporal_connection(round_idx)

            print(f"[图 {self.id} | 轮次 {round_idx+1}] 开始拓扑排序执行...", flush=True)
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]
            print(f"[图 {self.id} | 轮次 {round_idx+1}] 计算得到的初始入度: {in_degree}", flush=True)
            print(f"[图 {self.id} | 轮次 {round_idx+1}] 初始零入度队列: {zero_in_degree_queue}", flush=True)
            
            execution_tasks: Dict[str, asyncio.Task] = {}
            # processed_nodes_count = 0 # Renamed to completed_node_ids_this_round

            print(f"[图 {self.id} | 轮次 {round_idx+1}] 进入拓扑排序循环 (初始队列大小: {len(zero_in_degree_queue)}, 初始任务数: {len(execution_tasks)})", flush=True)

            while zero_in_degree_queue or execution_tasks:
                print(f"[图 {self.id} | 轮次 {round_idx+1} | 循环开始] 队列: {zero_in_degree_queue}, 运行中: {list(execution_tasks.keys())}", flush=True)

                runnable_nodes_from_queue = []
                while zero_in_degree_queue: # Process all current zero-in-degree nodes
                    node_id = zero_in_degree_queue.pop(0)
                    if node_id not in execution_tasks: # Not already running or submitted
                         runnable_nodes_from_queue.append(node_id)
                
                if runnable_nodes_from_queue:
                    print(f"[图 {self.id} | 轮次 {round_idx+1}] 准备运行的节点: {runnable_nodes_from_queue}", flush=True)
                for node_id in runnable_nodes_from_queue:
                    print(f"[图 {self.id} | 轮次 {round_idx+1}] 为节点 {node_id} 创建执行任务", flush=True)
                    node = self.nodes[node_id]
                    task = asyncio.create_task(self.execute_node_with_retry(node, input_dict, max_tries, max_time))
                    execution_tasks[node_id] = task
                
                if not execution_tasks: # All tasks submitted from queue are done, and queue is empty
                    print(f"[图 {self.id} | 轮次 {round_idx+1}] 无运行中任务且队列已空，拓扑排序段结束。", flush=True)
                    break # Exit while loop

                print(f"[图 {self.id} | 轮次 {round_idx+1}] 等待任务完成... 当前运行中: {list(execution_tasks.keys())}", flush=True)
                done, pending = await asyncio.wait(list(execution_tasks.values()), return_when=asyncio.FIRST_COMPLETED)
                
                newly_completed_ids_processed_this_iter = []
                for task in done:
                    completed_node_id = None
                    for nid, t_val in execution_tasks.items():
                        if t_val == task:
                            completed_node_id = nid
                            break
                    
                    if completed_node_id:
                        print(f"[图 {self.id} | 轮次 {round_idx+1}] 节点 {completed_node_id} 的任务完成", flush=True)
                        completed_node_ids_this_round.add(completed_node_id)
                        try:
                            await task # Retrieve result or raise exception
                        except Exception as e:
                            print(f"错误: [图 {self.id} | 轮次 {round_idx+1}] 节点 {completed_node_id} 执行最终失败: {e}", flush=True)
                            # traceback.print_exc() # Optionally print full traceback for node error
                            if completed_node_id in self.nodes: # Store error state
                                self.nodes[completed_node_id].outputs = [f"<Execution Error: {type(e).__name__}>"]

                        completed_node_obj = self.nodes.get(completed_node_id)
                        if completed_node_obj:
                            print(f"[图 {self.id} | 轮次 {round_idx+1}] 更新节点 {completed_node_id} 的后继节点入度...", flush=True)
                            for successor in completed_node_obj.spatial_successors:
                                if successor.id in self.nodes and successor.id in in_degree:
                                    in_degree[successor.id] -= 1
                                    print(f"  - 后继节点 {successor.id} 新入度: {in_degree[successor.id]}", flush=True)
                                    if in_degree[successor.id] == 0:
                                        print(f"  - 后继节点 {successor.id} 入度为 0，加入队列。", flush=True)
                                        if successor.id not in zero_in_degree_queue and successor.id not in execution_tasks: # Avoid re-adding if already processed or running
                                            zero_in_degree_queue.append(successor.id)
                                elif successor.id not in self.nodes:
                                     print(f"  - 警告: 后继节点 {successor.id} 不在图中 (可能是决策节点)。", flush=True)
                                elif successor.id not in in_degree:
                                     print(f"  - 警告: 后继节点 {successor.id} 不在入度字典中。", flush=True)
                        
                        newly_completed_ids_processed_this_iter.append(completed_node_id)

                for node_id in newly_completed_ids_processed_this_iter:
                    if node_id in execution_tasks:
                        # print(f"[图 {self.id} | 轮次 {round_idx+1}] 从 execution_tasks 中移除已完成的任务 {node_id}", flush=True)
                        del execution_tasks[node_id]
                print(f"[图 {self.id} | 轮次 {round_idx+1} | 循环结束] 队列: {zero_in_degree_queue}, 运行中: {list(execution_tasks.keys())}", flush=True)
            # --- while loop for topological sort ends ---

            print(f"[图 {self.id} | 轮次 {round_idx+1}] 拓扑排序循环结束。", flush=True)
            all_graph_node_ids = set(self.nodes.keys())
            if completed_node_ids_this_round != all_graph_node_ids:
                 uncompleted_nodes = all_graph_node_ids - completed_node_ids_this_round
                 print(f"警告: 轮次 {round_idx+1} 结束时，并非所有节点都成功执行或处理 ({len(completed_node_ids_this_round)}/{self.num_nodes})。", flush=True)
                 print(f"  未完成/处理的节点: {list(uncompleted_nodes)}", flush=True)
                 # Log in-degrees of uncompleted nodes
                 for un_node_id in uncompleted_nodes:
                     print(f"    - {un_node_id} 当前计算入度: {in_degree.get(un_node_id, 'N/A')}", flush=True)


            for node_id, node in self.nodes.items():
                round_answers[f"{node.role}_{node_id}"] = node.outputs[:] if node.outputs else ["<Not Executed or No Output>"]
            all_answers.append(round_answers)

            print(f"[图 {self.id} | 轮次 {round_idx+1}] 更新所有节点 Memory...", flush=True)
            self.update_memory()
            print(f"[图 {self.id}] --- 第 {round_idx + 1} 轮结束 ---", flush=True)
        # --- for round_idx loop ends ---

        final_answers = ["<Decision Node Not Executed or No Output>"]
        if hasattr(self, 'decision_node') and self.decision_node:
            try:
                print(f"[图 {self.id}] --- 开始执行决策节点 {self.decision_node.id} ---", flush=True)
                self.connect_decision_node() # Connect all agent nodes to decision node
                await self.execute_node_with_retry(self.decision_node, input_dict, max_tries, max_time)
                final_answers = self.decision_node.outputs[:]
                if not final_answers: # Ensure there's always a list, even if empty from node
                    final_answers = ["<No Answer from Decision Node>"]
                print(f"[图 {self.id}] --- 决策节点 {self.decision_node.id} 执行完毕, 输出: {final_answers} ---", flush=True)
            except Exception as e:
                print(f"错误: 决策节点 {self.decision_node.id} 执行失败: {e}", flush=True)
                print(f"决策节点错误堆栈:\n{traceback.format_exc()}", flush=True)
                final_answers = [f"<Decision Error: {type(e).__name__}>"]
        else:
             print("警告: 未配置决策节点或决策节点为 None。", flush=True)
             # Fallback if no decision node: try to use last agent's output or a generic message
             if self.nodes:
                 # This is arbitrary; consider a more defined fallback if needed
                 last_node_key = list(self.nodes.keys())[-1]
                 final_answers = self.nodes[last_node_key].outputs[:] if self.nodes[last_node_key].outputs else ["<No Decision Node, Last Agent No Output>"]
             else:
                 final_answers = ["<No Decision Node and No Agents>"]


        print(f"[图 {self.id}] arun 方法执行完毕，返回最终答案和所有轮次答案。", flush=True)
        return final_answers, all_answers


    async def execute_node_with_retry(self, node: Node, input_dict: Dict, max_tries: int, timeout: int):
        tries = 0
        last_exception = None
        # print(f"[图 {self.id} | 节点 {node.id}] 进入 execute_node_with_retry", flush=True)
        while tries < max_tries:
            tries += 1
            try:
                # print(f"[图 {self.id} | 节点 {node.id}] 第 {tries}/{max_tries} 次尝试执行 (超时: {timeout}s)...", flush=True)
                await asyncio.wait_for(node.async_execute(input_dict), timeout=timeout)
                # print(f"[图 {self.id} | 节点 {node.id}] 第 {tries} 次尝试成功。", flush=True)
                return
            except asyncio.TimeoutError:
                last_exception = asyncio.TimeoutError(f"节点 {node.id} 执行超时 ({timeout}s) 在尝试 {tries}/{max_tries}")
                print(f"警告: [图 {self.id} | 节点 {node.id}] 第 {tries}/{max_tries} 次尝试超时。", flush=True)
            except Exception as e:
                last_exception = e
                print(f"警告: [图 {self.id} | 节点 {node.id}] 第 {tries}/{max_tries} 次尝试执行出错: {e}", flush=True)
                # print(f"  节点 {node.id} 错误堆栈 (尝试 {tries}):\n{traceback.format_exc()}", flush=True) # Can be very verbose
            
            if tries < max_tries: # Optional: delay before retrying
                 await asyncio.sleep(0.1 * tries) # Small increasing delay

        print(f"错误: [图 {self.id} | 节点 {node.id}] 在 {max_tries} 次尝试后执行失败。最后错误: {last_exception}", flush=True)
        raise last_exception if last_exception else RuntimeError(f"节点 {node.id} 执行失败且无明确异常 после {max_tries} попыток")


    def update_memory(self):
        for node in self.nodes.values():
            node.update_memory()

    # --- RL 状态获取方法 (保持不变) ---
    def get_agent_states(self) -> Dict[str, Dict[str, Any]]:
        states = {}
        for node_id, node in self.nodes.items():
            states[node_id] = {
                "outputs": node.outputs[:], # Get a copy
                "memory": copy.deepcopy(node.memory) if hasattr(node, 'memory') and node.memory is not None else None,
                "role": node.role if hasattr(node, 'role') else 'Unknown'
            }
        return states

    def get_current_masks(self) -> Tuple[Union[torch.Tensor, List[torch.Tensor], None], Union[torch.Tensor, List[torch.Tensor], None]]:
        return self.runtime_spatial_masks, self.runtime_temporal_masks

    def get_intermediate_results(self, all_answers_from_arun: List[Dict[str, List[str]]]) -> Any:
        # This method seems to just return its input; its purpose might be for specific post-processing.
        return all_answers_from_arun

