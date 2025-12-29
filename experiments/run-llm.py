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
import copy
from typing import List, Union, Literal, Dict, Any, Tuple
import numpy as np
import random

# <--- NEW: 导入 transformers 库 ---
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("警告: transformers 库未安装。如果需要使用 --structure_generator 'llm' 模式，请运行 'pip install transformers torch'.")
    AutoModelForCausalLM, AutoTokenizer = None, None
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

# --- Argument Parsing ---
def parse_args(): # <--- MODIFIED: 修改参数解析
    parser = argparse.ArgumentParser(description="Agent Connection Learning on gsm8k")
    # --- 数据集和结果文件 ---
    parser.add_argument("--dataset_json", type=str, default="datasets/gsm8k/gsm8k.jsonl", help="测试数据集路径")
    parser.add_argument("--train_dataset_json", type=str, default="datasets/gsm8k/train.jsonl", help="训练数据集路径")
    parser.add_argument("--result_file", type=str, default=None, help="结果保存文件路径 (可选, 默认自动生成)")
    parser.add_argument("--domain", type=str, default="gsm8k", help="任务领域 (数据集名称), 默认 'gsm8k'")

    # --- LLM 和 Agent 配置 ---
    parser.add_argument("--llm_name", type=str, default="gpt-3.5-turbo", help="执行任务的大语言模型名称")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'], help='Agent类型列表')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4], help='每种Agent类型的数量')
    parser.add_argument('--decision_method', type=str, default='FinalRefer', help='最终决策Agent的方法')
    parser.add_argument('--num_rounds', type=int, default=1, help="每个查询的Agent交互轮数")

    # --- 结构生成器配置 ---
    parser.add_argument('--structure_generator', type=str, default='fixed',
                        choices=['fixed', 'rl', 'llm'],
                        help="选择通信结构的生成方式: 'fixed' (使用 --fixed_mode), 'rl' (强化学习), 'llm' (大模型生成)")
    parser.add_argument('--fixed_mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain','Debate','Layered','Star'],
                        help="当 --structure_generator 为 'fixed' 时，使用的固定拓扑结构")

    # --- LLM 结构生成器特定参数 ---
    parser.add_argument('--llm_generator_path', type=str, default=None,
                        help="用于生成结构的本地LLM模型路径 (当 --structure_generator='llm' 时必须)")

    # --- RL 相关参数 ---
    parser.add_argument('--rl_config', type=str, default='RL/configs/default_ppo.yaml', help='RL配置文件路径')
    parser.add_argument('--rl_train_iterations', type=int, default=100, help='RL训练的总迭代次数 (或 episodes)')
    parser.add_argument('--rl_update_steps', type=int, default=2048, help='收集多少步经验后更新一次RL策略')
    parser.add_argument('--rl_ema_alpha', type=float, default=0.1, help='Agent历史表现EMA更新的alpha值')

    # --- 奖励函数参数 ---
    parser.add_argument('--reward_alpha', type=float, default=2.5, help='任务性能奖励权重')
    parser.add_argument('--reward_beta', type=float, default=0.5, help='通信成本惩罚权重')
    parser.add_argument('--reward_gamma', type=float, default=0.05, help='图多样性奖励权重')
    parser.add_argument('--invalid_graph_penalty', type=float, default=-1.0, help='当生成的图无效时施加的惩罚')

    # --- 训练和批处理 ---
    parser.add_argument('--batch_size', type=int, default=4, help="批处理大小")

    args = parser.parse_args()
    result_path = Path(project_root) / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("Agent名称列表长度必须与Agent数量列表长度一致。")
    if args.structure_generator == 'llm' and not args.llm_generator_path:
        parser.error("--structure_generator='llm' 需要提供 --llm_generator_path 参数。")

    # --- 自动生成结果文件名 ---
    if args.result_file is None:
        current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        Time.instance().value = current_time
        result_dir = result_path / args.domain
        result_dir.mkdir(parents=True, exist_ok=True)
        generator_tag = f"_{args.structure_generator.upper()}"
        llm_name_sanitized = args.llm_name.split('/')[-1].replace('-', '_')
        args.result_file = result_dir / f"{args.domain}_{llm_name_sanitized}{generator_tag}_{current_time}.json"
    else:
        args.result_file = Path(args.result_file)

    print(f"结果将保存在: {args.result_file}")
    return args

# --- RL 相关辅助函数 (保持不变) ---
def get_rl_state(history: Dict, num_agents: int) -> np.ndarray:
    # ... (代码无变化) ...
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
    # ... (代码无变化) ...
    reward = args.reward_alpha * task_perf - args.reward_beta * comm_cost + args.reward_gamma * graph_diversity
    return reward

def calculate_comm_cost(spatial_masks: torch.Tensor, temporal_masks: torch.Tensor) -> float:
    # ... (代码无变化) ...
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
    # ... (代码无变化) ...
    if masks is None or masks.numel() == 0:
        return 0.0
    p = masks.mean().item()
    if p <= 0 or p >= 1:
        return 0.0
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return entropy

# <--- MODIFIED: LLM 结构生成器函数 (已重构) ---
def generate_structure_with_llm(
    task: str,
    agent_list: List[str], # 传入完整的 agent 列表
    model,
    tokenizer
) -> Union[List[List[int]], None]:
    """
    使用大模型根据任务动态生成通信矩阵。
    此版本能处理LLM返回agent子集的情况。
    """
    num_agents = len(agent_list)
    agent_descriptions = "\n".join([f"- Agent {i}: {name}" for i, name in enumerate(agent_list)])
    prompt =  f"""
我正在解决一个任务，需要为一组智能体设计通信结构。请根据任务描述，生成一个合理的通信方案。

任务描述: "{task}"

可用智能体列表 (共 {num_agents} 个):
{agent_descriptions}

请按以下步骤操作：
1.  **选择智能体**: 从上面的列表中选择解决此任务所需的一个智能体子集。
2.  **设计通信**: 为你选择的智能体子集设计一个通信矩阵。矩阵 `M[i][j] = 1` 表示你选择的第 `i` 个智能体可以将信息发送给第 `j` 个。对角线必须为0。

⚠️ **约束要求**：

- 通信矩阵必须构成一个 **前向无环图（DAG）**，禁止出现闭环或自我依赖。
- 结构中必须明确 **信息传递的终点 Agent**，因为最终输出将直接来自这个 Agent。
- 请避免将信息传递给 `"FinalRefer"`、`"Reflector"`、`"Checker"` 这类评估角色，它们不会生成有效的最终输出。
- 最终一个被执行的 Agent（无出边者）必须是一个能独立给出结果的 Agent（例如 `"MathSolver"` 或 `"Mathematical Analyst"`）

目标是设计一个清晰、高效的信息流结构，逐步汇聚推理，并由某个核心 Agent 给出最终答案。

请直接输出 JSON 格式，不要包含任何解释或思考过程。JSON 格式如下:
{{
  "reasoning": "简要说明你选择这些智能体和这种结构的理由。",
  "used_agent_indices": [/* 一个包含你选择的智能体原始索引的列表，例如 [0, 2, 3] */],
  "communication_matrix": [
    /* 一个 k*k 的矩阵, k 是 used_agent_indices 的长度。矩阵顺序必须与索引列表的顺序对应 */
    [0, 1, ...],
    [0, 0, ...],
    ...
  ]
}}
"""

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            print(f"警告: LLM 输出中未找到有效的 JSON 对象。输出: {content}")
            return None

        json_str = content[json_start:json_end]
        data = json.loads(json_str)

        subset_matrix = data.get("communication_matrix")
        used_indices = data.get("used_agent_indices")

        # --- 验证和重建矩阵 ---
        if not isinstance(used_indices, list) or not all(isinstance(i, int) for i in used_indices):
            print(f"警告: LLM 返回的 'used_agent_indices' 格式无效。")
            return None
        if not isinstance(subset_matrix, list):
            print(f"警告: LLM 返回的 'communication_matrix' 格式无效。")
            return None
        if len(subset_matrix) != len(used_indices):
            print(f"警告: LLM 返回的矩阵大小 ({len(subset_matrix)}) 与索引列表长度 ({len(used_indices)}) 不匹配。")
            return None
        for i, row in enumerate(subset_matrix):
            if not isinstance(row, list) or len(row) != len(used_indices):
                print(f"警告: LLM 返回的矩阵不是方阵或尺寸不匹配 (第 {i} 行)。")
                return None

        # --- 成功，开始重建完整 n*n 矩阵 ---
        full_matrix = [[0] * num_agents for _ in range(num_agents)]
        for i, source_row_idx in enumerate(used_indices):
            for j, target_col_idx in enumerate(used_indices):
                if i == j: continue # 确保对角线为0
                # 将 subset_matrix[i][j] 的值赋给 full_matrix 的正确位置
                if source_row_idx < num_agents and target_col_idx < num_agents:
                    full_matrix[source_row_idx][target_col_idx] = subset_matrix[i][j]
                else:
                    print(f"警告: LLM 返回了无效的索引 {source_row_idx} 或 {target_col_idx} (总智能体数: {num_agents})。")


        print(f"LLM 生成结构成功。理由: {data.get('reasoning', 'N/A')}")
        return full_matrix

    except json.JSONDecodeError as e:
        print(f"错误: 解析 LLM 输出的 JSON 失败: {e}\n原始输出: {content}")
        return None
    except Exception as e:
        print(f"错误: 调用 LLM 生成结构时发生未知错误: {e}")
        import traceback
        traceback.print_exc()
        return None


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
    except Exception as e:
        print(f"错误: 加载或处理数据时出错 - {e}")
        sys.exit(1)

    # --- Agent 和 Graph 初始化 ---
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    num_agents = len(agent_names)
    print(f"总 Agent 数量: {num_agents}")
    print(f"Agent 名称列表: {agent_names}")

    # <--- MODIFIED: 使用 fixed_mode ---
    initial_kwargs = get_kwargs(args.fixed_mode, num_agents)
    fixed_temporal_masks = torch.tensor(initial_kwargs["fixed_temporal_masks"], dtype=torch.float32) if initial_kwargs["fixed_temporal_masks"] else torch.ones((num_agents, num_agents), dtype=torch.float32)

    base_graph = Graph(domain=args.domain,
                       llm_name=args.llm_name,
                       agent_names=agent_names,
                       decision_method=args.decision_method,
                       rounds=args.num_rounds,
                       fixed_temporal_masks=fixed_temporal_masks,
                       node_kwargs=initial_kwargs["node_kwargs"],
                       fixed_spatial_masks=initial_kwargs["fixed_spatial_masks"]
                      )

    # --- 初始化结构生成器 ---
    rl_trainer = None
    llm_generator_model, llm_generator_tokenizer = None, None

    # <--- MODIFIED: 根据模式初始化对应的生成器 ---
    if args.structure_generator == 'rl':
        print("启用强化学习 (RL) 结构生成器...")
        try:
            rl_config = load_rl_config(args.rl_config)
            state_dim = num_agents + num_agents * num_agents
            action_dim = num_agents * num_agents
            rl_trainer = PPOTrainer(state_dim=state_dim, action_dim=action_dim, config=rl_config)
            print("PPOTrainer 初始化成功。")
        except Exception as e:
            print(f"错误: 初始化 PPOTrainer 失败 - {e}")
            sys.exit(1)
    elif args.structure_generator == 'llm':
        print(f"启用大模型 (LLM) 结构生成器，加载模型: {args.llm_generator_path}")
        if AutoModelForCausalLM is None:
             print("错误: 'transformers' 库未加载，无法使用 LLM 生成器。")
             sys.exit(1)
        try:
            llm_generator_tokenizer = AutoTokenizer.from_pretrained(args.llm_generator_path, trust_remote_code=True)
            llm_generator_model = AutoModelForCausalLM.from_pretrained(
                args.llm_generator_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print("LLM 结构生成器加载成功。")
        except Exception as e:
            print(f"错误: 加载 LLM 结构生成器模型失败 - {e}")
            sys.exit(1)
    else: # 'fixed'
        print(f"启用固定结构生成器，模式: {args.fixed_mode}")


    # --- 训练循环 ---
    total_solved, total_executed = (0, 0)
    total_steps = 0
    num_train_batches = (len(train_dataset) + args.batch_size - 1) // args.batch_size
    # <--- MODIFIED: 迭代次数现在对所有模式都有效 ---
    num_iterations = args.rl_train_iterations if args.structure_generator == 'rl' else 1

    # --- 初始化状态维护变量 ---
    agent_ema_perf = np.full(num_agents, 0.5, dtype=np.float32)
    initial_mask_np = np.array(initial_kwargs["fixed_spatial_masks"], dtype=np.float32) if initial_kwargs["fixed_spatial_masks"] else np.ones((num_agents, num_agents), dtype=np.float32)
    np.fill_diagonal(initial_mask_np, 0)
    last_spatial_mask = initial_mask_np # 作为所有模式的初始/回退 Mask

    for i_iter in range(num_iterations):
        print(f"\n{'='*30} 迭代 {i_iter + 1}/{num_iterations} {'='*30}")
        iteration_start_ts = time.time()
        iter_solved, iter_executed = 0, 0
        # ... (重置 Cost 等) ...

        for i_batch in range(num_train_batches):
            # ... (dataloader) ...
            current_batch = dataloader(train_dataset, args.batch_size, i_batch)
            if current_batch is None: break

            batch_tasks_info = []

            for i_record, record in enumerate(current_batch):
                task = record["task"]
                input_dict = {"task": task}

                # --- 结构生成决策步骤 ---
                current_spatial_mask_tensor = None
                log_prob, value, state, action_mask_flat = None, None, None, None

                # <--- MODIFIED: 核心逻辑，根据模式选择如何生成 Mask ---
                if args.structure_generator == 'rl':
                    history_for_state = {'agent_ema_perf': agent_ema_perf, 'last_spatial_mask': last_spatial_mask}
                    state = get_rl_state(history_for_state, num_agents)
                    action_mask_flat, log_prob, value = rl_trainer.get_action_and_value(state)
                    current_spatial_mask_tensor = action_mask_flat.float().reshape((num_agents, num_agents))

                elif args.structure_generator == 'llm':
                    print(f"\n[Iter {i_iter+1} Batch {i_batch+1} Rec {i_record+1}] 使用 LLM 生成结构...")
                    llm_generated_matrix = generate_structure_with_llm(task, agent_names, llm_generator_model, llm_generator_tokenizer)
                    if llm_generated_matrix is not None:
                        current_spatial_mask_tensor = torch.tensor(llm_generated_matrix, dtype=torch.float32)
                        # 更新 last_spatial_mask 以便下次失败时回退
                        last_spatial_mask = current_spatial_mask_tensor.cpu().numpy()
                    else:
                        print("LLM 生成失败，回退到上一个有效或默认的结构。")
                        current_spatial_mask_tensor = torch.from_numpy(last_spatial_mask).float()

                else: # 'fixed' mode
                    current_spatial_mask_tensor = torch.from_numpy(last_spatial_mask).float()

                # --- 执行 Agent 交互 ---
                realized_graph = copy.deepcopy(base_graph)
                realized_graph.set_runtime_masks(spatial_masks=current_spatial_mask_tensor,
                                                 temporal_masks=fixed_temporal_masks)
                task_future = asyncio.create_task(realized_graph.arun(input_dict, args.num_rounds))

                # 存储任务信息
                batch_tasks_info.append({
                    "future": task_future, "record": record,
                    "state": state, "action": action_mask_flat, "log_prob": log_prob, "value": value,
                    "spatial_mask_tensor": current_spatial_mask_tensor, "temporal_mask": fixed_temporal_masks
                })

            # --- 收集结果、计算奖励、存储经验 ---
            results = await asyncio.gather(*(info["future"] for info in batch_tasks_info))
            data_to_save = load_result(args.result_file)

            for i, (raw_result, task_info) in enumerate(zip(results, batch_tasks_info)):
                # ... (评估任务性能的代码，与原来基本相同) ...
                record = task_info["record"]
                task = record["task"]
                true_answer = record["answer"]
                spatial_mask_tensor = task_info["spatial_mask_tensor"]
                temporal_mask = task_info["temporal_mask"]
                if isinstance(raw_result, tuple) and len(raw_result) == 2: final_answer_list, _ = raw_result
                elif isinstance(raw_result, list): final_answer_list = raw_result
                else: final_answer_list = ["<Error>"]
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
                task_perf = float(is_solved)
                iter_solved += task_perf
                iter_executed += 1
                total_solved += task_perf
                total_executed += 1
                current_accuracy = iter_solved / iter_executed if iter_executed > 0 else 0
                overall_accuracy = total_solved / total_executed if total_executed > 0 else 0

                # --- 计算 RL 奖励 & 存储经验 (仅在 RL 模式下) ---
                reward = 0.0
                is_graph_valid = True # 默认为有效
                # <--- MODIFIED: 仅在 RL 模式下计算奖励和更新 ---
                if args.structure_generator == 'rl' and rl_trainer and task_info["state"] is not None:
                    # 图有效性检查 (代码不变)
                    if num_agents > 0:
                        temp_graph_for_check = copy.deepcopy(base_graph)
                        try:
                            temp_graph_for_check.set_runtime_masks(spatial_masks=spatial_mask_tensor, temporal_masks=fixed_temporal_masks)
                            temp_graph_for_check.construct_spatial_connection(round_idx=0)
                            in_degree_check = {node_id: len(node.spatial_predecessors) for node_id, node in temp_graph_for_check.nodes.items()}
                            if not any(deg == 0 for deg in in_degree_check.values()):
                                is_graph_valid = False
                        finally:
                            del temp_graph_for_check

                    comm_cost = calculate_comm_cost(spatial_mask_tensor, temporal_mask)
                    graph_diversity = calculate_graph_diversity(spatial_mask_tensor)
                    reward = calculate_reward(task_perf, comm_cost, graph_diversity, args)

                    if not is_graph_valid:
                        reward += args.invalid_graph_penalty

                    done = True
                    total_steps += 1
                    rl_trainer.store_experience(
                        state=task_info["state"], action=task_info["action"], log_prob=task_info["log_prob"],
                        reward=reward, value=task_info["value"], done=done
                    )

                    if total_steps > 0 and total_steps % args.rl_update_steps == 0:
                        print(f"\n--- 达到 {total_steps} 步，执行 RL Policy Update ---", flush=True)
                        rl_trainer.update()
                        try:
                           model_save_path = Path(args.result_file).with_suffix('.pt')
                           rl_trainer.save(str(model_save_path))
                           print(f"RL 模型已保存至: {model_save_path}")
                        except Exception as e:
                           print(f"保存 RL 模型失败: {e}")

                # --- 更新 Agent 的 EMA 历史表现 (仅在 RL 模式下) ---
                if args.structure_generator == 'rl':
                    alpha = args.rl_ema_alpha
                    current_mask_np = spatial_mask_tensor.cpu().numpy()
                    for agent_idx in range(num_agents):
                        is_involved = (current_mask_np[agent_idx, :].sum() + current_mask_np[:, agent_idx].sum()) > 0
                        if is_involved:
                            agent_ema_perf[agent_idx] = alpha * task_perf + (1 - alpha) * agent_ema_perf[agent_idx]

                # --- 更新上一轮 Mask (仅在 RL 模式下, LLM模式在成功生成时已更新) ---
                if args.structure_generator == 'rl':
                    last_spatial_mask = spatial_mask_tensor.cpu().numpy()

                # --- 记录结果 ---
                updated_item = {
                    "Iteration": i_iter + 1, "Batch": i_batch + 1, "Record_Index": i,
                    "Structure_Generator": args.structure_generator, # <--- NEW: 记录生成器类型
                    "Question": task, "True Answer": true_answer,
                    "Response": final_answer_str, "Predicted Answer": predict_answer,
                    "Solved": bool(is_solved),
                    "Reward (RL)": f"{reward:.4f}{' (Invalid Graph Penalty)' if not is_graph_valid else ''}" if args.structure_generator == 'rl' else "N/A",
                    "Iter Accuracy": f"{current_accuracy:.4f}",
                    "Overall Accuracy": f"{overall_accuracy:.4f}",
                }
                data_to_save.append(updated_item)

            # --- 保存当前批次结果 ---
            with open(args.result_file, 'w', encoding='utf-8') as file:
                json.dump(data_to_save, file, indent=4, ensure_ascii=False)

            # 实时打印
            print(f"\rIter {i_iter+1}, Batch {i_batch+1}/{num_train_batches}, Steps: {total_steps if args.structure_generator == 'rl' else 'N/A'}, Overall Acc: {overall_accuracy:.4f}   ", end="", flush=True)

        # --- 迭代结束 ---
        # ... (迭代结束后的打印信息，与原来类似) ...
        iter_accuracy = iter_solved / iter_executed if iter_executed > 0 else 0
        print(f"\n{'='*30} 迭代 {i_iter + 1} 结束 {'='*30}")
        print(f"迭代耗时: {time.time() - iteration_start_ts:.3f}s")
        print(f"迭代准确率: {iter_accuracy:.4f}")
        print(f"累计总准确率: {overall_accuracy:.4f}")
        if args.structure_generator == 'rl':
             print("最终 Agent EMA 表现分数:")
             for idx, score in enumerate(agent_ema_perf):
                 print(f"  Agent {idx}: {score:.4f}")


    # --- 所有迭代结束后 ---
    print("\n训练/评估完成。")
    if args.structure_generator == 'rl' and rl_trainer:
        model_save_path = Path(args.result_file).with_suffix('.pt')
        try:
            rl_trainer.save(str(model_save_path))
            print(f"RL 模型已保存至: {model_save_path}")
        except Exception as e:
            print(f"保存 RL 模型失败: {e}")

# --- get_kwargs (保持不变) ---
def get_kwargs(mode: str, N: int):
    # ... (此函数代码完全不变) ...
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

