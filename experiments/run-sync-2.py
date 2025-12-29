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

# --- 路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# --- 导入模块 ---
from SPARCS.graph.graph import Graph
from SPARCS.tools.reader.readers import JSONLReader
from SPARCS.utils.globals import Time
from datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict
from datasets.aqua_dataset import aqua_data_process, aqua_get_predict
from datasets.mmlu_dataset import mmlu_data_process, mmlu_get_predict  # ✅ 使用函数接口

from SPARCS.llm.gpt_chat import GPTChat
from SPARCS.llm.format import Message

# --- LLM结构生成 ---
def generate_structure_with_llm(task: str, agent_list: List[str], gptchat_instance: GPTChat) -> Union[List[List[int]], None]:
    num_agents = len(agent_list)
    agent_descriptions = "\n".join([f"- Agent {i}: {name}" for i, name in enumerate(agent_list)])

    prompt = f"""
我正在解决一个任务，需要为一组智能体设计通信结构。请根据任务描述，生成一个合理的通信方案。

任务描述: "{task}"

可用智能体列表 (共 {num_agents} 个):
{agent_descriptions}

请按以下步骤操作：
1. 选择智能体: 从上面的列表中选择解决此任务所需的一个智能体子集。
2. 设计通信: 为你选择的智能体子集设计一个通信矩阵。矩阵 M[i][j] = 1 表示第 i 个智能体可以向第 j 个智能体发送信息。对角线必须为0。

⚠️ 约束要求：
- 通信矩阵必须构成一个前向无环图（DAG），禁止出现闭环或自我依赖。
- 结构中必须明确信息传递的终点 Agent，因为最终输出将直接来自这个 Agent。
- 请避免将信息传递给 "FinalRefer"、"Reflector"、"Checker" 等评估角色，它们不会生成有效的最终输出。
- 最终一个被执行的 Agent（无出边者）必须是一个能独立给出结果的 Agent（如 "MathSolver" 或 "Mathematical Analyst"）

请直接输出 JSON 格式，不要包含任何解释或思考过程。JSON 格式如下:
{{
  "reasoning": "简要说明你选择这些智能体和这种结构的理由。",
  "used_agent_indices": [0, 2, 3],
  "communication_matrix": [
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0]
  ]
}}
"""
    messages = [{"role": "user", "content": prompt}]
    try:
        response = gptchat_instance.gen(messages)
        data = json.loads(response)

        subset_matrix = data.get("communication_matrix")
        used_indices = data.get("used_agent_indices")

        if not all([
            isinstance(used_indices, list),
            all(isinstance(i, int) for i in used_indices),
            isinstance(subset_matrix, list),
            len(subset_matrix) == len(used_indices),
            all(isinstance(row, list) and len(row) == len(used_indices) for row in subset_matrix)
        ]):
            print(f"⚠️ 无效结构返回，跳过。")
            return None

        full_matrix = [[0] * num_agents for _ in range(num_agents)]
        for i, src in enumerate(used_indices):
            for j, tgt in enumerate(used_indices):
                if i != j:
                    full_matrix[src][tgt] = subset_matrix[i][j]

        print(f"✅ LLM 结构生成成功。理由: {data.get('reasoning', '无')}")
        return full_matrix

    except Exception as e:
        print(f"❌ 结构生成失败: {e}")
        return None

# --- 加载结果 ---
def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- 参数解析 ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json", type=str, default="datasets/gsm8k/test.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--domain", type=str, default="gsm8k", choices=["gsm8k", "aqua", "mmlu"])
    parser.add_argument("--llm_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--structure_llm_name", type=str, default=None)
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'])
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4])
    parser.add_argument('--decision_method', type=str, default='FinalRefer')
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--structure_generator', type=str, default='fixed', choices=['fixed', 'llm'])
    parser.add_argument('--fixed_mode', type=str, default='FullConnected')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    result_path = Path(project_root) / "result"
    os.makedirs(result_path, exist_ok=True)

    if len(args.agent_names) != len(args.agent_nums):
        parser.error("agent_names 和 agent_nums 长度不一致。")

    if args.result_file is None:
        current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        Time.instance().value = current_time
        result_dir = result_path / args.domain
        result_dir.mkdir(parents=True, exist_ok=True)
        args.result_file = result_dir / f"{args.domain}_{args.structure_generator}_{current_time}.json"
    else:
        args.result_file = Path(args.result_file)

    print(f"结果将保存至: {args.result_file}")
    return args

# --- 初始化结构 ---
def get_kwargs(mode: str, N: int) -> dict:
    if N == 0:
        return {"fixed_spatial_masks": [], "fixed_temporal_masks": [], "node_kwargs": []}
    spatial_mask_data = [[1 if i != j else 0 for j in range(N)] for i in range(N)]
    if mode == "Chain":
        spatial_mask_data = [[0]*N for _ in range(N)]
        for i in range(N-1): spatial_mask_data[i][i+1] = 1
    elif mode == "Star":
        spatial_mask_data = [[0]*N for _ in range(N)]
        for j in range(1, N): spatial_mask_data[0][j] = 1
    elif mode == "Layered":
        spatial_mask_data = [[0]*N for _ in range(N)]
        num_layer1 = (N + 1) // 2
        for i in range(num_layer1):
            for j in range(num_layer1, N):
                spatial_mask_data[i][j] = 1
    elif mode == "Random":
        spatial_mask_data = [[random.choice([0, 1]) if i != j else 0 for j in range(N)] for i in range(N)]
    elif mode in ["DirectAnswer", "Debate"]:
        spatial_mask_data = [[0]*N for _ in range(N)]

    temporal_mask_data = [[1]*N for _ in range(N)]
    node_kwargs = [{} for _ in range(N)]

    return {
        "fixed_spatial_masks": spatial_mask_data,
        "fixed_temporal_masks": temporal_mask_data,
        "node_kwargs": node_kwargs
    }

# --- 主程序 ---
def main():
    args = parse_args()

    dataset = JSONLReader.parse_file(args.dataset_json)
    if args.domain == "gsm8k":
        dataset = gsm_data_process(dataset)
    elif args.domain == "aqua":
        dataset = aqua_data_process(dataset)
    elif args.domain == "mmlu":
        dataset = mmlu_data_process(dataset)  # ✅ 使用函数接口处理 JSONL
    else:
        raise ValueError(f"不支持的 domain: {args.domain}")

    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    N = len(agent_names)
    print(f"Agent 数量: {N}")

    init_kwargs = get_kwargs(args.fixed_mode, N)
    temporal_mask = torch.tensor(init_kwargs["fixed_temporal_masks"], dtype=torch.float32)

    structure_model = args.structure_llm_name if args.structure_llm_name else args.llm_name
    gptchat_instance = GPTChat(structure_model) if args.structure_generator == "llm" else None

    graph = Graph(
        domain=args.domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        rounds=args.num_rounds,
        fixed_temporal_masks=temporal_mask,
        node_kwargs=init_kwargs["node_kwargs"],
        fixed_spatial_masks=init_kwargs["fixed_spatial_masks"]
    )

    last_mask_np = np.array(init_kwargs["fixed_spatial_masks"], dtype=np.float32)
    np.fill_diagonal(last_mask_np, 0)

    data_to_save = load_result(args.result_file)
    total_solved, total = 0, 0

    for i, record in enumerate(dataset):
        task = record["task"]

        if args.structure_generator == "llm":
            print(f"\n[{i+1}/{len(dataset)}] 使用 GPTChat API 生成结构 (模型: {structure_model})...")
            new_mask = generate_structure_with_llm(task, agent_names, gptchat_instance)
            current_mask = torch.tensor(new_mask if new_mask else last_mask_np, dtype=torch.float32)
        else:
            current_mask = torch.from_numpy(last_mask_np).float()

        g = copy.deepcopy(graph)
        g.set_runtime_masks(spatial_masks=current_mask, temporal_masks=temporal_mask)

        result = g.run({"task": task}, args.num_rounds)
        answer = result[0][0] if isinstance(result, tuple) else result[0]

        if args.domain == "gsm8k":
            pred = gsm_get_predict(answer)
            try:
                correct = float(pred) == float(record["answer"])
            except:
                correct = False
        elif args.domain == "aqua":
            pred = aqua_get_predict(answer)
            correct = pred == record["answer"]
        elif args.domain == "mmlu":
            pred = mmlu_get_predict(answer)
            correct = pred == record["answer"]
        else:
            raise ValueError(f"不支持的 domain: {args.domain}")

        total += 1
        total_solved += correct

        data_to_save.append({
            "Task_ID": len(data_to_save),
            "Structure_Generator": args.structure_generator,
            "Question": task,
            "True Answer": record["answer"],
            "Response": answer,
            "Predicted Answer": pred,
            "Solved": correct,
            "Overall Accuracy": f"{total_solved / total:.4f}"
        })

        if (i + 1) % args.batch_size == 0 or (i + 1) == len(dataset):
            with open(args.result_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            print(f"保存结果 ✓ 当前准确率: {total_solved / total:.4f}", end="\r")

    print(f"\n评估结束，共 {total} 条，总准确率: {total_solved / total:.4f}")
    print(f"结果保存在: {args.result_file}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"程序异常: {e}")
        import traceback
        traceback.print_exc()
