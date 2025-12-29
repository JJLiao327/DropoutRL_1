# -*- coding: utf-8 -*-
import sys
import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
import torch
import torch.nn.functional as F
import copy # <--- 确保导入 copy
from typing import List, Union, Literal, Dict, Any, Tuple
import numpy as np # 引入 numpy
import random

# --- 路径设置 (根据你的项目结构调整) ---
# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加项目根目录 (假设脚本在 experiments/ 目录下)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
rl_root = os.path.join(project_root, 'rl') # RL 文件夹路径
sys.path.append(project_root)
sys.path.append(rl_root) # 将 RL 目录添加到 sys.path

from SPARCS.utils.const import AgentPrune_ROOT
# 假设 Graph 类已修改，可以接收 masks 并提供状态信息
from SPARCS.graph.graph import Graph
from SPARCS.tools.reader.readers import JSONLReader # JSONReader 未使用，移除
from SPARCS.utils.globals import Time
from SPARCS.utils.globals import Cost, PromptTokens, CompletionTokens
from datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict
from datasets.aqua_dataset import aqua_data_process, aqua_get_predict
from SPARCS.agents.agent_registry import AgentRegistry

# --- RL 相关导入 ---
from rl.trainer import PPOTrainer # <--- 修改：导入正确的训练器
from rl.configs import load_rl_config

# --- Helper Functions (保持不变) ---
def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w', encoding='utf-8') as file:
            json.dump([], file)
    with open(result_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    start_idx = i_batch * batch_size
    end_idx = start_idx + batch_size
    if start_idx >= len(data_list):
        return None
    return data_list[start_idx:end_idx]

# --- Argument Parsing (保持不变) ---
def parse_args():
    parser = argparse.ArgumentParser(description="RL-based Agent Connection Learning on gsm8k")
    # --- 数据集和结果文件 ---
    parser.add_argument("--dataset_json", type=str, default="datasets/gsm8k/gsm8k.jsonl", help="测试数据集路径")
    parser.add_argument("--train_dataset_json", type=str, default="datasets/gsm8k/train.jsonl", help="训练数据集路径")
    parser.add_argument("--result_file", type=str, default=None, help="结果保存文件路径 (可选, 默认自动生成)")
    parser.add_argument("--domain", type=str, default="gsm8k", help="任务领域 (数据集名称), 默认 'gsm8k'")

    # --- LLM 和 Agent 配置 ---
    parser.add_argument("--llm_name", type=str, default="gpt-3.5-turbo", help="使用的大语言模型名称")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'], help='Agent类型列表')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4], help='每种Agent类型的数量')
    parser.add_argument('--decision_method', type=str, default='FinalRefer', help='最终决策Agent的方法')
    parser.add_argument('--num_rounds', type=int, default=1, help="每个查询的Agent交互轮数")

    # --- 原始模式参数 (保留用于初始化或比较) ---
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain','Debate','Layered','Star'],
                        help="用于获取初始固定Mask的模式 (如果RL不学习初始结构)")

    # --- RL 相关参数 ---
    parser.add_argument('--use_rl', action='store_true', help='启用RL学习连接关系')
    parser.add_argument('--rl_config', type=str, default='RL/configs/default_ppo.yaml', help='RL配置文件路径') # 示例
    parser.add_argument('--rl_train_iterations', type=int, default=100, help='RL训练的总迭代次数 (或 episodes)')
    parser.add_argument('--rl_update_steps', type=int, default=2048, help='收集多少步经验后更新一次RL策略') # 例如，PPO常用2048步
    parser.add_argument('--rl_ema_alpha', type=float, default=0.1, help='Agent历史表现EMA更新的alpha值') # 新增EMA alpha参数

    parser.add_argument('--reward_alpha', type=float, default=2.5, help='任务性能奖励权重')
    parser.add_argument('--reward_beta', type=float, default=0.5, help='通信成本惩罚权重')
    parser.add_argument('--reward_gamma', type=float, default=0.05, help='图多样性奖励权重')
    # --- 新增：无效图惩罚参数 ---
    parser.add_argument('--invalid_graph_penalty', type=float, default=-1.0, help='当生成的图无效（无起始节点）时施加的惩罚')

    # --- 训练和批处理 ---
    parser.add_argument('--batch_size', type=int, default=4, help="批处理大小 (注意：RL更新基于步数，batch_size影响数据加载)")

    args = parser.parse_args()
    result_path = Path(project_root) / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("Agent名称列表长度必须与Agent数量列表长度一致。")

    # --- 自动生成结果文件名 ---
    if args.result_file is None:
        current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        Time.instance().value = current_time
        result_dir = result_path / args.domain # 使用拼接路径
        result_dir.mkdir(parents=True, exist_ok=True)
        rl_tag = "_RL" if args.use_rl else ""
        llm_name_sanitized = args.llm_name.split('/')[-1].replace('-', '_')
        args.result_file = result_dir / f"{args.domain}_{llm_name_sanitized}{rl_tag}_{current_time}.json"
    else:
        args.result_file = Path(args.result_file)

    print(f"结果将保存在: {args.result_file}")
    return args

# --- RL 相关辅助函数 ---
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

# --- Main Execution Logic ---
async def main():
    args = parse_args()

    # --- 数据加载 ---
    try:
        train_dataset = JSONLReader.parse_file(args.train_dataset_json)
        if args.domain == "gsm8k":
            train_dataset = gsm_data_process(train_dataset)
        elif args.domain == "aqua":
            train_dataset = aqua_data_process(train_dataset)
        else:
            print(f"警告: 未知的 domain '{args.domain}' 用于训练集, 使用原始数据。")
    except FileNotFoundError as e:
        print(f"错误: 数据文件未找到 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 加载或处理数据时出错 - {e}")
        sys.exit(1)

    # --- Agent 和 Graph 初始化 ---
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    num_agents = len(agent_names)
    print(f"总 Agent 数量: {num_agents}")
    print(f"Agent 名称列表: {agent_names}")

    initial_kwargs = get_kwargs(args.mode, num_agents)
    fixed_temporal_masks = torch.tensor(initial_kwargs["fixed_temporal_masks"], dtype=torch.float32) if initial_kwargs["fixed_temporal_masks"] else torch.ones((num_agents, num_agents), dtype=torch.float32) # 默认全连接

    # **修改部分：添加 fixed_spatial_masks 以作为 PPO 学习的起点**
    base_graph = Graph(domain=args.domain,
                       llm_name=args.llm_name,
                       agent_names=agent_names,
                       decision_method=args.decision_method,
                       rounds=args.num_rounds,
                       fixed_temporal_masks=fixed_temporal_masks,
                       node_kwargs=initial_kwargs["node_kwargs"],
                       fixed_spatial_masks=initial_kwargs["fixed_spatial_masks"]  # 新增参数
                      )

    # --- RL Agent 初始化 ---
    rl_trainer = None
    if args.use_rl:
        print("启用强化学习模式...")
        try:
            rl_config = load_rl_config(args.rl_config)
            print(f"加载 RL 配置: {args.rl_config}")
        except FileNotFoundError:
            print(f"错误: RL 配置文件 {args.rl_config} 未找到！")
            sys.exit(1)
        except Exception as e:
            print(f"错误: 加载 RL 配置时出错 - {e}")
            sys.exit(1)

        state_dim = num_agents + num_agents * num_agents
        print(f"计算得到 RL 状态维度 state_dim = {state_dim}")
        action_dim = num_agents * num_agents # N*N

        try:
            rl_trainer = PPOTrainer(state_dim=state_dim,
                                    action_dim=action_dim,
                                    config=rl_config)
            print("PPOTrainer 初始化成功。")
        except Exception as e:
            print(f"错误: 初始化 PPOTrainer 失败 - {e}")
            sys.exit(1)

    # --- 训练循环 ---
    total_solved, total_executed = (0, 0)
    total_steps = 0 # 记录 RL 交互的总步数
    num_train_batches = (len(train_dataset) + args.batch_size - 1) // args.batch_size
    num_iterations = args.rl_train_iterations if args.use_rl else 1

    # --- 初始化 RL 状态维护变量 ---
    agent_ema_perf = np.full(num_agents, 0.5, dtype=np.float32)
    initial_mask_np = np.array(initial_kwargs["fixed_spatial_masks"], dtype=np.float32) if initial_kwargs["fixed_spatial_masks"] else np.ones((num_agents, num_agents), dtype=np.float32)
    np.fill_diagonal(initial_mask_np, 0) # 确保无自环
    last_spatial_mask = initial_mask_np # N x N NumPy 数组

    for i_iter in range(num_iterations):
        print(f"\n{'='*30} 迭代 {i_iter + 1}/{num_iterations} {'='*30}")
        iteration_start_ts = time.time()
        iter_solved, iter_executed = 0, 0
        Cost.instance().reset()
        PromptTokens.instance().reset()
        CompletionTokens.instance().reset()

        for i_batch in range(num_train_batches):
            batch_start_ts = time.time()
            current_batch = dataloader(train_dataset, args.batch_size, i_batch)
            if current_batch is None:
                print("训练数据加载完毕。")
                break

            batch_tasks_info = []

            for i_record, record in enumerate(current_batch):
                task = record["task"]
                input_dict = {"task": task}

                # --- RL 决策步骤 ---
                current_spatial_mask_tensor = None # PyTorch Tensor
                log_prob = None
                value = None
                state = None # NumPy Array

                if args.use_rl and rl_trainer:
                    history_for_state = {
                        'agent_ema_perf': agent_ema_perf,
                        'last_spatial_mask': last_spatial_mask
                    }
                    state = get_rl_state(history_for_state, num_agents)
                    action_mask_flat, log_prob, value = rl_trainer.get_action_and_value(state)
                    action_mask_flat = action_mask_flat.float()
                    try:
                        current_spatial_mask_tensor = action_mask_flat.reshape((num_agents, num_agents))
                    except RuntimeError as e:
                         print(f"错误: 无法将动作 reshape 为 ({num_agents}, {num_agents}) - {e}")
                         current_spatial_mask_tensor = torch.from_numpy(last_spatial_mask).float()
                         log_prob = torch.tensor(0.0) # 使用默认值，避免后续计算出错
                         value = torch.tensor(0.0)  # 使用默认值
                else:
                    current_spatial_mask_tensor = torch.from_numpy(last_spatial_mask).float()

                # --- 执行 Agent 交互 ---
                realized_graph = copy.deepcopy(base_graph)
                realized_graph.set_runtime_masks(spatial_masks=current_spatial_mask_tensor,
                                                 temporal_masks=fixed_temporal_masks)

                task_future = asyncio.create_task(realized_graph.arun(input_dict, args.num_rounds))

                # 存储任务信息
                batch_tasks_info.append({
                    "future": task_future,
                    "record": record,
                    "state": state,
                    "action": action_mask_flat if state is not None else None,
                    "log_prob": log_prob,
                    "value": value,
                    "spatial_mask_tensor": current_spatial_mask_tensor,
                    "temporal_mask": fixed_temporal_masks
                })

            # --- 收集结果并计算奖励 & 存储经验 & 更新状态维护变量 ---
            results = await asyncio.gather(*(info["future"] for info in batch_tasks_info))
            data_to_save = load_result(args.result_file)

            for i, (raw_result, task_info) in enumerate(zip(results, batch_tasks_info)):
                record = task_info["record"]
                task = record["task"]
                true_answer = record["answer"]
                spatial_mask_tensor = task_info["spatial_mask_tensor"]
                temporal_mask = task_info["temporal_mask"]

                # 解析结果
                if isinstance(raw_result, tuple) and len(raw_result) == 2:
                     final_answer_list, _ = raw_result
                elif isinstance(raw_result, list):
                    final_answer_list = raw_result
                else:
                    print(f"警告: 任务 {i} 返回了意外的格式: {type(raw_result)}")
                    final_answer_list = ["<Error>"]
                final_answer_str = final_answer_list[0] if final_answer_list else "<No Answer>"

                # 评估任务性能
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

                task_perf = float(is_solved)
                iter_solved += task_perf
                iter_executed += 1
                total_solved += task_perf
                total_executed += 1
                current_accuracy = iter_solved / iter_executed if iter_executed > 0 else 0
                overall_accuracy = total_solved / total_executed if total_executed > 0 else 0

                # --- 计算 RL 奖励 & 存储经验 (如果使用RL) ---
                reward = 0.0 # 初始化奖励
                if args.use_rl and rl_trainer and task_info["state"] is not None:
                    # ================== 新增：图有效性检查 ==================
                    is_graph_valid = True # 默认为有效
                    if num_agents > 0: # 只有当有节点时才需要检查
                        temp_graph_for_check = copy.deepcopy(base_graph)
                        try:
                            temp_graph_for_check.set_runtime_masks(spatial_masks=spatial_mask_tensor, temporal_masks=fixed_temporal_masks)
                            temp_graph_for_check.construct_spatial_connection(round_idx=0)
                            in_degree_check = {node_id: len(node.spatial_predecessors) for node_id, node in temp_graph_for_check.nodes.items()}
                            zero_in_degree_nodes_exist = any(deg == 0 for deg in in_degree_check.values())

                            if not zero_in_degree_nodes_exist:
                                is_graph_valid = False
                                print(f"[奖励计算 | Iter {i_iter+1} Batch {i_batch+1} Rec {i}] 警告: 生成的图无效 (无起始节点)，将应用惩罚。", flush=True)
                        except Exception as e:
                            print(f"[奖励计算 | Iter {i_iter+1} Batch {i_batch+1} Rec {i}] 检查图有效性时出错: {e}", flush=True)
                            is_graph_valid = False
                        finally:
                            del temp_graph_for_check
                    # ======================================================

                    comm_cost = calculate_comm_cost(spatial_mask_tensor, temporal_mask)
                    graph_diversity = calculate_graph_diversity(spatial_mask_tensor)
                    reward = calculate_reward(task_perf, comm_cost, graph_diversity, args)

                    if not is_graph_valid:
                        invalid_graph_penalty = args.invalid_graph_penalty
                        print(f"[奖励计算 | Iter {i_iter+1} Batch {i_batch+1} Rec {i}] 应用无效图惩罚: {invalid_graph_penalty:.2f}", flush=True)
                        reward += invalid_graph_penalty

                    done = True
                    total_steps += 1

                    rl_trainer.store_experience(
                        state=task_info["state"],
                        action=task_info["action"],
                        log_prob=task_info["log_prob"],
                        reward=reward,
                        value=task_info["value"],
                        done=done
                    )

                    if total_steps > 0 and total_steps % args.rl_update_steps == 0:
                        print(f"\n--- 达到 {total_steps} 步，执行 RL Policy Update ---", flush=True)
                        update_start_ts = time.time()
                        rl_trainer.update()
                        try:
                           model_save_path = Path(args.result_file).with_suffix('.pt')
                           rl_trainer.save(str(model_save_path))
                           print(f"RL 模型已保存至: {model_save_path}")
                        except Exception as e:
                                             print(f"保存 RL 模型失败: {e}")
                        print(f"[奖励计算 | Iter {i_iter+1} Batch {i_batch+1} Rec {i}] 奖励: {reward:.2f}, 通信成本: {comm_cost:.2f}, 图多样性: {graph_diversity:.2f}", flush=True)
                        print(f"RL 更新耗时: {time.time() - update_start_ts:.3f}s", flush=True)

                # --- 更新 Agent 的 EMA 历史表现 ---
                if args.use_rl:
                    alpha = args.rl_ema_alpha
                    current_mask_np = spatial_mask_tensor.cpu().numpy()
                    for agent_idx in range(num_agents):
                        is_involved = (current_mask_np[agent_idx, :].sum() + current_mask_np[:, agent_idx].sum()) > 0
                        if is_involved:
                            agent_ema_perf[agent_idx] = alpha * task_perf + (1 - alpha) * agent_ema_perf[agent_idx]

                # --- 更新上一轮 Mask (为下一次迭代准备) ---
                last_spatial_mask = spatial_mask_tensor.cpu().numpy()

                # --- 记录结果 (同之前逻辑) ---
                updated_item = {
                    "Iteration": i_iter + 1, "Batch": i_batch + 1, "Record_Index": i,
                    "Question": task, "True Answer": true_answer,
                    "Response": final_answer_str, "Predicted Answer": predict_answer,
                    "Solved": bool(is_solved),
                    "Reward (RL)": f"{reward:.4f}{' (Invalid Graph Penalty Applied)' if args.use_rl and not is_graph_valid else ''}" if args.use_rl and task_info['state'] is not None else "N/A",
                    "Iter Accuracy": f"{current_accuracy:.4f}",
                    "Overall Accuracy": f"{overall_accuracy:.4f}",
                }
                data_to_save.append(updated_item)

            # --- 保存当前批次结果 ---
            try:
                with open(args.result_file, 'w', encoding='utf-8') as file:
                    json.dump(data_to_save, file, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"错误: 保存结果文件失败 - {e}")

            # 实时打印
            if args.use_rl:
                print(f"\rIter {i_iter+1}, Batch {i_batch+1}/{num_train_batches}, Steps: {total_steps}, Overall Acc: {overall_accuracy:.4f}   ", end="", flush=True)

        # --- 迭代结束 ---
        iter_accuracy = iter_solved / iter_executed if iter_executed > 0 else 0
        print(f"\n{'='*30} 迭代 {i_iter + 1} 结束 {'='*30}")
        print(f"迭代耗时: {time.time() - iteration_start_ts:.3f}s")
        print(f"迭代准确率: {iter_accuracy:.4f}")
        print(f"累计总准确率: {overall_accuracy:.4f}")
        print(f"累计成本: {Cost.instance().value:.2f}")
        print(f"累计 Prompt Tokens: {PromptTokens.instance().value}")
        print(f"累计 Completion Tokens: {CompletionTokens.instance().value}")
        if args.use_rl:
             print(f"总 RL 步数: {total_steps}")
             print("最终 Agent EMA 表现分数:")
             for idx, score in enumerate(agent_ema_perf):
                 print(f"  Agent {idx}: {score:.4f}")

    # --- 所有迭代结束后 ---
    print("\n训练/评估完成。")
    print(f"最终总准确率 ({total_executed} 个样本): {overall_accuracy:.4f}")
    print(f"结果已保存至: {args.result_file}")
    if args.use_rl and rl_trainer:
        model_save_path = Path(args.result_file).with_suffix('.pt')
    try:
        rl_trainer.save(str(model_save_path))
        print(f"RL 模型已保存至: {model_save_path}")
    except Exception as e:
        print(f"保存 RL 模型失败: {e}")
        
# --- get_kwargs (保持不变) ---
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

# --- Entry Point ---
if __name__ == '__main__':
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n检测到中断信号，程序退出。")
    except Exception as e:
        print(f"\n程序运行时发生未捕获的异常: {e}")
        import traceback
        traceback.print_exc()
