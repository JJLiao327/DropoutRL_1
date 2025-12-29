# 文件名: experiments/run-sync.py
# 描述: 完全同步化的主执行脚本

import sys
import os
import argparse
import json
import time
from pathlib import Path
import torch
import copy
from typing import List, Union
import numpy as np
import random

# <--- NEW: 导入 transformers 库 ---
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("警告: transformers 库未安装。如果需要使用 --structure_generator 'llm' 模式，请运行 'pip install transformers torch'.")
    AutoModelForCausalLM, AutoTokenizer = None, None

# --- 路径设置 (根据你的项目结构调整) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# <--- CHANGED: 导入修改后的 Graph 类 ---
from SPARCS.graph.graph import Graph
from SPARCS.tools.reader.readers import JSONLReader
from SPARCS.utils.globals import Time
from datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict
from datasets.aqua_dataset import aqua_data_process, aqua_get_predict

# --- Helper Functions (保持不变) ---
def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w', encoding='utf-8') as file:
            json.dump([], file)
    with open(result_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# --- Argument Parsing (保持不变, 仅简化描述) ---
def parse_args():
    parser = argparse.ArgumentParser(description="Agent Connection Learning (Synchronous Version)")
    # ... (所有参数定义与之前完全相同) ...
    # --- 数据集和结果文件 ---
    parser.add_argument("--dataset_json", type=str, default="datasets/gsm8k/test.jsonl", help="测试数据集路径")
    parser.add_argument("--result_file", type=str, default=None, help="结果保存文件路径 (可选, 默认自动生成)")
    parser.add_argument("--domain", type=str, default="gsm8k", choices=["gsm8k", "aqua"], help="任务领域 (数据集名称)")

    # --- LLM 和 Agent 配置 ---
    parser.add_argument("--llm_name", type=str, default="qwen/qwen-2.5-72b-instruct", help="执行任务的大语言模型名称")  #gpt-3.5-turbo
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'], help='Agent类型列表')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4], help='每种Agent类型的数量')
    parser.add_argument('--decision_method', type=str, default='FinalRefer', help='最终决策Agent的方法')
    parser.add_argument('--num_rounds', type=int, default=1, help="每个查询的Agent交互轮数")

    # --- 结构生成器配置 ---
    parser.add_argument('--structure_generator', type=str, default='fixed',
                        choices=['fixed', 'llm'],
                        help="选择通信结构的生成方式: 'fixed' (使用 --fixed_mode), 'llm' (大模型生成)")
    parser.add_argument('--fixed_mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain','Debate','Layered','Star'],
                        help="当 --structure_generator 为 'fixed' 时，使用的固定拓扑结构")

    # --- LLM 结构生成器特定参数 ---
    parser.add_argument('--llm_generator_path', type=str, default=None,
                        help="用于生成结构的本地LLM模型路径 (当 --structure_generator='llm' 时必须)")

    # --- 批处理 ---
    parser.add_argument('--batch_size', type=int, default=4, help="批处理大小 (同步模式下仅用于分批保存结果)")
    
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

# --- LLM 结构生成器函数 (保持不变) ---
def generate_structure_with_llm(
    task: str,
    agent_list: List[str],
    model,
    tokenizer
) -> Union[List[List[int]], None]:
    # ... (此函数代码完全不变) ...
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
        if not all([
            isinstance(used_indices, list),
            all(isinstance(i, int) for i in used_indices),
            isinstance(subset_matrix, list),
            len(subset_matrix) == len(used_indices),
            all(isinstance(row, list) and len(row) == len(used_indices) for row in subset_matrix)
        ]):
            print(f"警告: LLM 返回的 JSON 结构或类型无效。")
            return None

        # --- 成功，开始重建完整 n*n 矩阵 ---
        full_matrix = [[0] * num_agents for _ in range(num_agents)]
        for i, source_row_idx in enumerate(used_indices):
            for j, target_col_idx in enumerate(used_indices):
                if i == j: continue
                if 0 <= source_row_idx < num_agents and 0 <= target_col_idx < num_agents:
                    full_matrix[source_row_idx][target_col_idx] = subset_matrix[i][j]
                else:
                    print(f"警告: LLM 返回了越界的索引 {source_row_idx} 或 {target_col_idx} (总智能体数: {num_agents})。")

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
# <--- CHANGED: 移除 async ---
def main():
    args = parse_args()

    # --- 数据加载 (保持不变) ---
    try:
        dataset = JSONLReader.parse_file(args.dataset_json)
        if args.domain == "gsm8k":
            dataset = gsm_data_process(dataset)
        elif args.domain == "aqua":
            dataset = aqua_data_process(dataset)
    except Exception as e:
        print(f"错误: 加载或处理 '{args.dataset_json}' 时出错 - {e}")
        sys.exit(1)

    # --- Agent 和 Graph 初始化 (保持不变) ---
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    num_agents = len(agent_names)
    print(f"总 Agent 数量: {num_agents}")
    print(f"Agent 名称列表: {agent_names}")

    initial_kwargs = get_kwargs(args.fixed_mode, num_agents)
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

    # --- 初始化结构生成器 (保持不变) ---
    llm_generator_model, llm_generator_tokenizer = None, None
    if args.structure_generator == 'llm':
        # ... (LLM 加载代码不变) ...
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

    # --- 评估循环 ---
    total_solved, total_executed = 0, 0
    initial_mask_np = np.array(initial_kwargs["fixed_spatial_masks"], dtype=np.float32)
    np.fill_diagonal(initial_mask_np, 0)
    last_spatial_mask = initial_mask_np
    
    data_to_save = load_result(args.result_file)

    # <--- CHANGED: 简化为单层循环，不再有批处理并发 ---
    for i_record, record in enumerate(dataset):
        task = record["task"]
        input_dict = {"task": task}

        # --- 结构生成决策步骤 ---
        if args.structure_generator == 'llm':
            print(f"\n[Record {i_record+1}/{len(dataset)}] 使用 LLM 生成结构...")
            llm_generated_matrix = generate_structure_with_llm(task, agent_names, llm_generator_model, llm_generator_tokenizer)
            if llm_generated_matrix is not None:
                current_spatial_mask_tensor = torch.tensor(llm_generated_matrix, dtype=torch.float32)
                last_spatial_mask = current_spatial_mask_tensor.cpu().numpy()
            else:
                print("LLM 生成失败，回退到默认结构。")
                current_spatial_mask_tensor = torch.from_numpy(last_spatial_mask).float()
        else: # 'fixed' mode
            current_spatial_mask_tensor = torch.from_numpy(last_spatial_mask).float()

        # --- 执行 Agent 交互 (同步调用) ---
        realized_graph = copy.deepcopy(base_graph)
        realized_graph.set_runtime_masks(spatial_masks=current_spatial_mask_tensor,
                                         temporal_masks=fixed_temporal_masks)
        
        # <--- CHANGED: 直接调用 run() 并等待其完成 ---
        raw_result = realized_graph.run(input_dict, args.num_rounds)

        # --- 结果解析和评估 ---
        if isinstance(raw_result, tuple) and len(raw_result) == 2: final_answer_list, _ = raw_result
        elif isinstance(raw_result, list): final_answer_list = raw_result
        else: final_answer_list = ["<Error processing result>"]
        
        final_answer_str = final_answer_list[0] if final_answer_list else "<No Answer>"

        if args.domain == "gsm8k":
            predict_answer = gsm_get_predict(final_answer_str)
            try: is_solved = float(predict_answer) == float(record["answer"])
            except (ValueError, TypeError): is_solved = False
        elif args.domain == "aqua":
            predict_answer = aqua_get_predict(final_answer_str)
            is_solved = predict_answer == record["answer"]
        
        total_solved += float(is_solved)
        total_executed += 1
        overall_accuracy = total_solved / total_executed

        # --- 记录结果 ---
        updated_item = {
            "Task_ID": len(data_to_save),
            "Structure_Generator": args.structure_generator,
            "Question": record["task"],
            "True Answer": record["answer"],
            "Response": final_answer_str,
            "Predicted Answer": predict_answer,
            "Solved": bool(is_solved),
            "Overall Accuracy": f"{overall_accuracy:.4f}",
        }
        data_to_save.append(updated_item)

        # --- 定期保存结果 ---
        if (i_record + 1) % args.batch_size == 0 or (i_record + 1) == len(dataset):
            with open(args.result_file, 'w', encoding='utf-8') as file:
                json.dump(data_to_save, file, indent=4, ensure_ascii=False)
            print(f"\rProcessed: {total_executed}/{len(dataset)}, Overall Accuracy: {overall_accuracy:.4f} (Results saved) ", end="", flush=True)
        else:
            print(f"\rProcessed: {total_executed}/{len(dataset)}, Overall Accuracy: {overall_accuracy:.4f}                  ", end="", flush=True)


    print(f"\n\n{'='*30} 评估完成 {'='*30}")
    print(f"总计处理任务数: {total_executed}")
    print(f"最终准确率: {overall_accuracy:.4f}")
    print(f"详细结果已保存至: {args.result_file}")


def get_kwargs(mode: str, N: int) -> dict:
    """
    根据给定的模式和 Agent 数量生成初始化的参数字典。
    """
    if N == 0:
        return {
            "fixed_spatial_masks": [],
            "fixed_temporal_masks": [],
            "node_kwargs": []
        }

    # --- 生成空间掩码 (Spatial Mask) ---
    spatial_mask_data = []
    if mode == "FullConnected":
        spatial_mask_data = [[1 if i != j else 0 for j in range(N)] for i in range(N)]
    elif mode == "Chain":
        spatial_mask_data = [[0] * N for _ in range(N)]
        if N > 1:
            for i in range(N - 1):
                spatial_mask_data[i][i+1] = 1
    elif mode == "Star":
        spatial_mask_data = [[0] * N for _ in range(N)]
        if N > 0:
            for j in range(1, N):
                spatial_mask_data[0][j] = 1
    elif mode == "Layered":
        spatial_mask_data = [[0] * N for _ in range(N)]
        if N > 1:
            num_layer1 = (N + 1) // 2
            for i in range(num_layer1):
                for j in range(num_layer1, N):
                    spatial_mask_data[i][j] = 1
    elif mode in ["DirectAnswer", "Debate", "Random"]: # Random 模式也从无连接开始
        spatial_mask_data = [[0] * N for _ in range(N)]
    else:
        print(f"警告: 未知的 fixed_mode '{mode}', 将使用 FullConnected 作为默认值。")
        spatial_mask_data = [[1 if i != j else 0 for j in range(N)] for i in range(N)]
    
    # 如果是 Random 模式，随机生成连接
    if mode == "Random":
        for i in range(N):
            for j in range(N):
                if i != j:
                    spatial_mask_data[i][j] = random.choice([0, 1])

    # --- 生成时间掩码 (Temporal Mask) ---
    # 默认情况下，所有节点都可以从上一轮的所有节点获取信息
    temporal_mask_data = [[1] * N for _ in range(N)]

    # --- 生成节点特定参数 ---
    node_kwargs = [{} for _ in range(N)]

    return {
        "fixed_spatial_masks": spatial_mask_data,
        "fixed_temporal_masks": temporal_mask_data,
        "node_kwargs": node_kwargs
    }



# --- Entry Point ---
if __name__ == '__main__':
    # <--- CHANGED: 移除所有 asyncio 相关代码 ---
    try:
        main()
    except KeyboardInterrupt:
        print("\n检测到中断信号，程序退出。")
    except Exception as e:
        print(f"\n程序运行时发生未捕获的异常: {e}")
        import traceback
        traceback.print_exc()

