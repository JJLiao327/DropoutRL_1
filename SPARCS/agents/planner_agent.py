# 文件: SPARCS/agents/planner_agent.py

from typing import List, Any, Dict

from SPARCS.graph.node import Node
from SPARCS.agents.agent_registry import AgentRegistry
from SPARCS.llm.llm_registry import LLMRegistry
from SPARCS.prompt.prompt_set_registry import PromptSetRegistry

@AgentRegistry.register('Planner')
class Planner(Node):
    def __init__(self, id: str | None = None, role: str = None, domain: str = "", llm_name: str = ""):
        # ... (此方法代码保持不变) ...
        super().__init__(id, "Planner", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = "Planner" if role is None else role
        try:
            self.constraint = self.prompt_set.get_constraint(self.role)
        except KeyError:
            print(f"警告: 在领域 '{domain}' 的提示集中未找到角色 '{self.role}' 的约束。使用默认规划提示。")
            self.constraint = "你是一个顶级的任务规划专家。你的目标是分析一个复杂的问题，并将其分解成子任务或步骤，请简要给出你的理解即可。"

    def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs) -> List[Any]:
        # ... (此方法代码保持不变) ...
        system_prompt = self.constraint
        question = raw_inputs[0]["task"]
        try:
            user_prompt = self.prompt_set.get_planning_prompt(question=question, role=self.role)
        except AttributeError:
            user_prompt = f"请为以下任务制定一个简要的分步计划：\n\n任务：{question}"
        return [system_prompt, user_prompt]

    # <--- CHANGED: _execute 重命名为 _run_llm_sync 并调整参数 ---
    def _run_llm_sync(self, processed_inputs: List[Any], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        """
        实现 Node 基类要求的同步执行方法。
        """
        # <--- CHANGED: 不再调用 _process_inputs，直接使用传入的 processed_inputs ---
        system_prompt, user_prompt = processed_inputs
        
        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        
        # <--- CHANGED: 使用同步的 self.llm.gen ---
        response = self.llm.gen(message)
        
        return [response] # <--- CHANGED: 确保返回一个列表 ---

    async def _async_execute(self, processed_inputs: List[Any], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        # ... (此方法代码保持不变，作为参考) ...
        system_prompt, user_prompt = processed_inputs
        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response = await self.llm.agen(message)
        return [response] # <--- 建议也修改这里，确保返回列表 ---
