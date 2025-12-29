"""
utils_rl.py
Utils for reinforcement learning in multi-agent communication environments.

适用于强化学习通信结构优化项目
Author: ChatGPT + Tianzhe
Updated: Apr-2025
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import networkx as nx


# =============================
# 🎯 通信图奖励结构
# =============================

REWARD_WEIGHTS = {
    "task_perf": 1.0,            # 任务性能（准确性、得分等）
    "comm_cost": -0.05,          # 通信开销（节点数、边数、token数）
    "graph_diversity": 0.1,      # 图结构多样性鼓励项
    "dropout_efficiency": 0.1,   # Dropout是否精准（踢掉无用Agent）
}


@dataclass
class StepResult:
    reward: float
    info: Dict

    def __iter__(self):
        return iter((self.reward, self.info))


def compute_task_performance(task_metric: float) -> float:
    """任务完成度得分，标准化到[0, 1]"""
    return np.clip(task_metric, 0.0, 1.0)


def compute_comm_cost(graph: nx.Graph, max_nodes: int, max_edges: int) -> float:
    """通信图的成本（节点 + 边）归一化"""
    node_penalty = len(graph.nodes) / max_nodes
    edge_penalty = len(graph.edges) / max_edges
    return node_penalty + edge_penalty


def compute_graph_diversity(graph: nx.Graph) -> float:
    """图结构多样性度量，可替换为信息熵 / Degree variance"""
    degrees = [d for _, d in graph.degree()]
    if len(degrees) <= 1:
        return 0.0
    return np.std(degrees) / (np.mean(degrees) + 1e-5)


def compute_dropout_efficiency(dropout_mask: List[int], useless_agent_ids: List[int]) -> float:
    """Dropout是否精准：是否踢掉了真正没用的Agent"""
    correct_drops = sum([1 for i in useless_agent_ids if dropout_mask[i] == 0])
    return correct_drops / (len(useless_agent_ids) + 1e-5)


# =============================
# 🧠 总奖励函数
# =============================

def compute_total_reward(task_metric: float,
                         graph: nx.Graph,
                         dropout_mask: List[int],
                         useless_agent_ids: List[int],
                         max_nodes: int,
                         max_edges: int) -> StepResult:
    """
    综合奖励计算函数（用于PPO环境中）
    """

    task_score = compute_task_performance(task_metric)
    comm_cost = compute_comm_cost(graph, max_nodes, max_edges)
    diversity = compute_graph_diversity(graph)
    dropout_eff = compute_dropout_efficiency(dropout_mask, useless_agent_ids)

    total_reward = (
        REWARD_WEIGHTS["task_perf"] * task_score +
        REWARD_WEIGHTS["comm_cost"] * comm_cost +
        REWARD_WEIGHTS["graph_diversity"] * diversity +
        REWARD_WEIGHTS["dropout_efficiency"] * dropout_eff
    )

    return StepResult(total_reward, {
        "task_score": task_score,
        "comm_cost": comm_cost,
        "diversity": diversity,
        "dropout_eff": dropout_eff
    })


# =============================
# 📦 PPO Buffer Utility
# =============================

@dataclass
class Transition:
    state: Dict
    action: Dict
    reward: float
    next_state: Dict
    done: bool
    log_prob: Optional[float] = None
    value: Optional[float] = None
    advantage: Optional[float] = None
    return_: Optional[float] = None


class PPOBuffer:
    """用于收集交互轨迹"""
    def __init__(self):
        self.buffer = []

    def store(self, transition: Transition):
        self.buffer.append(transition)

    def clear(self):
        self.buffer = []

    def get(self) -> List[Transition]:
        return self.buffer

    def compute_advantages(self, gamma: float = 0.99, lam: float = 0.95):
        """GAE Advantage计算"""
        rewards = [t.reward for t in self.buffer]
        values = [t.value for t in self.buffer]
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * (values[t+1] if t + 1 < len(values) else 0) - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        for i, adv in enumerate(advantages):
            self.buffer[i].advantage = adv
            self.buffer[i].return_ = self.buffer[i].advantage + self.buffer[i].value


# =============================
# 🔧 通信结构分析工具
# =============================

def build_graph_from_mask(agent_mask: List[int], edge_mask: List[Tuple[int, int]]) -> nx.Graph:
    """根据动作输出构建子图"""
    G = nx.Graph()
    active_nodes = [i for i, keep in enumerate(agent_mask) if keep == 1]
    G.add_nodes_from(active_nodes)
    for (i, j) in edge_mask:
        if i in G.nodes and j in G.nodes:
            G.add_edge(i, j)
    return G


def summarize_graph_stats(graph: nx.Graph) -> Dict:
    """输出图的结构性统计指标"""
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "avg_degree": np.mean([d for _, d in graph.degree()]) if graph.number_of_nodes() > 0 else 0,
        "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0
    }


# =============================
# ✅ Debug 测试用例
# =============================

if __name__ == "__main__":
    import random

    # 随机生成一个通信子图
    agent_mask = [1, 0, 1, 1, 0]  # 总共5个Agent，只有0/2/3保留
    edge_mask = [(0, 2), (2, 3), (0, 3), (1, 2)]  # 只有其中几个有效

    G = build_graph_from_mask(agent_mask, edge_mask)

    # 模拟情况
    task_metric = 0.85  # 假设系统正确完成任务
    useless_agents = [1, 4]  # 实际无用的Agent
    max_nodes, max_edges = 5, 10

    result = compute_total_reward(task_metric, G, agent_mask, useless_agents, max_nodes, max_edges)

    print(f"✅ Total Reward: {result.reward:.4f}")
    print("📊 Breakdown:", result.info)
    print("📈 Graph Stats:", summarize_graph_stats(G))
