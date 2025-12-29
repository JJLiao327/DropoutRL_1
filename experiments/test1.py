import sys
import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
import torch
import numpy as np
import copy
import random

# 路径设置（同训练脚本）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
rl_root = os.path.join(project_root, 'rl')
sys.path.append(project_root)
sys.path.append(rl_root)

from typing import List, Union, Literal, Dict, Any, Tuple
from SPARCS.utils.const import AgentPrune_ROOT
from SPARCS.graph.graph import Graph
from SPARCS.tools.reader.readers import JSONLReader
from SPARCS.utils.globals import Time, Cost, PromptTokens, CompletionTokens
from datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict
from datasets.aqua_dataset import aqua_data_process, aqua_get_predict
from SPARCS.agents.agent_registry import AgentRegistry
from rl.trainer import PPOTrainer
from rl.configs import load_rl_config

# 其它辅助函数(get_kwargs, get_rl_state, calculate_comm_cost, calculate_graph_diversity)
# 直接复制训练脚本里的实现即可

def parse_args():
    parser = argparse.ArgumentParser(description="RL Agent 推理/测试")
    parser.add_argument("--test_dataset_json", type=str, required=True, help="测试数据集路径")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的 RL 模型参数路径（.pt）")
    parser.add_argument("--rl_config", type=str, required=True, help="RL 配置文件路径")
    parser.add_argument("--domain", type=str, default="gsm8k")
    parser.add_argument("--llm_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'])
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4])
    parser.add_argument('--decision_method', type=str, default='FinalRefer')
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--mode', type=str, default='FullConnected')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument("--result_file", type=str, default=None)
    args = parser.parse_args()
    return args

async def main():
    args = parse_args()
    # 数据集加载
    test_dataset = JSONLReader.parse_file(args.test_dataset_json)
    if args.domain == "gsm8k":
        test_dataset = gsm_data_process(test_dataset)
    elif args.domain == "aqua":
        test_dataset = aqua_data_process(test_dataset)

    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    num_agents = len(agent_names)
    initial_kwargs = get_kwargs(args.mode, num_agents)
    fixed_temporal_masks = torch.tensor(initial_kwargs["fixed_temporal_masks"], dtype=torch.float32)
    base_graph = Graph(domain=args.domain,
                       llm_name=args.llm_name,
                       agent_names=agent_names,
                       decision_method=args.decision_method,
                       rounds=args.num_rounds,
                       fixed_temporal_masks=fixed_temporal_masks,
                       node_kwargs=initial_kwargs["node_kwargs"],
                       fixed_spatial_masks=initial_kwargs["fixed_spatial_masks"]
                      )
    # RL Agent 加载
    rl_config = load_rl_config(args.rl_config)
    state_dim = num_agents + num_agents * num_agents
    action_dim = num_agents * num_agents
    rl_trainer = PPOTrainer(state_dim=state_dim, action_dim=action_dim, config=rl_config)
    rl_trainer.load(args.model_path)
    print(f"加载 RL 模型: {args.model_path}")

    agent_ema_perf = np.full(num_agents, 0.5, dtype=np.float32)
    initial_mask_np = np.array(initial_kwargs["fixed_spatial_masks"], dtype=np.float32)
    np.fill_diagonal(initial_mask_np, 0)
    last_spatial_mask = initial_mask_np

    total_solved = 0
    total_executed = 0
    num_test_batches = (len(test_dataset) + args.batch_size - 1) // args.batch_size
    data_to_save = []

    for i_batch in range(num_test_batches):
        current_batch = test_dataset[i_batch * args.batch_size : (i_batch + 1) * args.batch_size]
        batch_tasks_info = []
        for i_record, record in enumerate(current_batch):
            task = record["task"]
            input_dict = {"task": task}
            # RL 决策
            history_for_state = {
                'agent_ema_perf': agent_ema_perf,
                'last_spatial_mask': last_spatial_mask
            }
            state = get_rl_state(history_for_state, num_agents)
            action_mask_flat, _, _ = rl_trainer.get_action_and_value(state)
            action_mask_flat = action_mask_flat.float()
            current_spatial_mask_tensor = action_mask_flat.reshape((num_agents, num_agents))
            realized_graph = copy.deepcopy(base_graph)
            realized_graph.set_runtime_masks(spatial_masks=current_spatial_mask_tensor,
                                             temporal_masks=fixed_temporal_masks)
            task_future = asyncio.create_task(realized_graph.arun(input_dict, args.num_rounds))
            batch_tasks_info.append({
                "future": task_future,
                "record": record,
                "spatial_mask_tensor": current_spatial_mask_tensor
            })

        results = await asyncio.gather(*(info["future"] for info in batch_tasks_info))
        for i, (raw_result, task_info) in enumerate(zip(results, batch_tasks_info)):
            record = task_info["record"]
            task = record["task"]
            true_answer = record["answer"]
            spatial_mask_tensor = task_info["spatial_mask_tensor"]
            if isinstance(raw_result, tuple) and len(raw_result) == 2:
                final_answer_list, _ = raw_result
            elif isinstance(raw_result, list):
                final_answer_list = raw_result
            else:
                final_answer_list = ["<Error>"]
            final_answer_str = final_answer_list[0] if final_answer_list else "<No Answer>"
            if args.domain == "gsm8k":
                predict_answer = gsm_get_predict(final_answer_str)
                try: is_solved = float(predict_answer) == float(true_answer)
                except (ValueError, TypeError): is_solved = False
            elif args.domain == "aqua":
                predict_answer = aqua_get_predict(final_answer_str)
                is_solved = predict_answer == true_answer
            else:
                predict_answer = final_answer_str
                is_solved = predict_answer == true_answer
            total_solved += float(is_solved)
            total_executed += 1
            updated_item = {
                "Batch": i_batch + 1, "Record_Index": i,
                "Question": task, "True Answer": true_answer,
                "Response": final_answer_str, "Predicted Answer": predict_answer,
                "Solved": bool(is_solved),
            }
            data_to_save.append(updated_item)
            # 更新 last_spatial_mask
            last_spatial_mask = spatial_mask_tensor.cpu().numpy()

        print(f"\rBatch {i_batch+1}/{num_test_batches}, Acc: {total_solved/total_executed:.4f}   ", end="", flush=True)

    print(f"\n测试集准确率: {total_solved/total_executed:.4f} ({total_executed} samples)")
    # 保存结果
    result_file = args.result_file or (Path(args.model_path).with_suffix('.test.json'))
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)
    print(f"测试结果已保存至: {result_file}")

# get_kwargs、get_rl_state、calculate_comm_cost、calculate_graph_diversity 直接复制训练脚本里的实现










def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star']]
               ,N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None

    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
        if N == 0 or layer_num <= 0: return adj_matrix
        base_size = N // layer_num
        remainder = N % layer_num
        start_idx = 0
        node_layers = [-1] * N
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layer_nodes = list(range(start_idx, start_idx + size))
            for node_idx in layer_nodes:
                if node_idx < N:
                    node_layers[node_idx] = i
            start_idx += size
        for i in range(N):
            current_layer = node_layers[i]
            if current_layer == -1 or current_layer == layer_num - 1:
                continue
            for j in range(N):
                if node_layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix

    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        if n <= 1: return matrix
        center_node = 0
        for i in range(1, n):
            matrix[center_node][i] = 1
            matrix[i][center_node] = 1
        return matrix

    if N <= 0:
        return {"initial_spatial_probability": 0.0, "fixed_spatial_masks": [],
                "initial_temporal_probability": 0.0, "fixed_temporal_masks": [],
                "node_kwargs": []}

    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        fixed_temporal_masks = [[0 for _ in range(N)] for _ in range(N)]
        if N==1:
            fixed_spatial_masks = [[0]]
            fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Math Solver'}] * N
    elif mode=='FullConnected':
        fixed_spatial_masks = [[1 if i != j else 0 for j in range(N)] for i in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random':
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for j in range(N)] for i in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain':
        fixed_spatial_masks = [[1 if j == i + 1 else 0 for j in range(N)] for i in range(N)]
        fixed_temporal_masks = [[0 for _ in range(N)] for _ in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]

    if node_kwargs is None:
        node_kwargs = [{}] * N
    elif len(node_kwargs) != N:
         print(f"警告: 模式 '{mode}' 的 node_kwargs 长度与 Agent 数量 {N} 不匹配，使用默认值。")
         node_kwargs = [{}] * N

    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}



def get_rl_state(history: Dict, num_agents: int) -> np.ndarray:
    state_features = []
    agent_ema_perf = history.get('agent_ema_perf')
    if agent_ema_perf is None or len(agent_ema_perf) != num_agents:
        agent_ema_perf = np.full(num_agents, 0.5, dtype=np.float32)
    state_features.append(agent_ema_perf.astype(np.float32))
    last_mask = history.get('last_spatial_mask')
    if last_mask is None or last_mask.shape != (num_agents, num_agents):
        last_mask = np.ones((num_agents, num_agents), dtype=np.float32)
        np.fill_diagonal(last_mask, 0)
    state_features.append(last_mask.flatten().astype(np.float32))
    final_state_vector = np.concatenate(state_features)
    expected_dim = num_agents + num_agents * num_agents
    if final_state_vector.shape[0] != expected_dim:
        raise ValueError(f"状态维度计算错误！预期 {expected_dim}, 得到 {final_state_vector.shape[0]}")
    return final_state_vector

def calculate_reward(task_perf: float, comm_cost: float, graph_diversity: float, args: argparse.Namespace) -> float:
    reward = args.reward_alpha * task_perf - args.reward_beta * comm_cost + args.reward_gamma * graph_diversity
    return reward

def calculate_comm_cost(spatial_masks: torch.Tensor, temporal_masks: torch.Tensor) -> float:
    num_agents = spatial_masks.shape[0]
    if num_agents <= 1:
        return 0.0
    active_spatial_edges = spatial_masks.sum()
    total_possible_spatial_edges = float(num_agents * num_agents)
    if total_possible_spatial_edges == 0:
        return 0.0
    cost = active_spatial_edges / total_possible_spatial_edges
    return cost.item()

def calculate_graph_diversity(masks: torch.Tensor) -> float:
    if masks is None or masks.numel() == 0:
        return 0.0
    p = masks.mean().item()
    if p <= 0 or p >= 1:
        return 0.0
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return entropy



















if __name__ == '__main__':
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
