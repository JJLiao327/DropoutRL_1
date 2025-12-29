# 文件: .../agents/mathsolver.py (或其他定义 MathSolver 的文件)

from typing import List,Any,Dict

from SPARCS.graph.node import Node
from SPARCS.agents.agent_registry import AgentRegistry
from SPARCS.llm.llm_registry import LLMRegistry
from SPARCS.prompt.prompt_set_registry import PromptSetRegistry
from SPARCS.tools.coding.python_executor import execute_code_get_return
from datasets.gsm8k_dataset import gsm_get_predict

@AgentRegistry.register('MathSolver')
class MathSolver(Node):
    def __init__(self, id: str | None =None, role:str = None ,domain: str = "", llm_name: str = "",):
        super().__init__(id, "MathSolver" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role) 
        
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], **kwargs)->List[Any]:
        # ... (此方法代码保持不变) ...
        system_prompt = self.constraint
        spatial_str = ""
        temporal_str = ""
        user_prompt = self.prompt_set.get_answer_prompt(question=raw_inputs[0]["task"],role=self.role)
        if self.role == "Math Solver":
            user_prompt += "(Hint: The answer is near to"
            for id, info in spatial_info.items():
                user_prompt += " "+gsm_get_predict(info["output"])
            for id, info in temporal_info.items():
                user_prompt += " "+gsm_get_predict(info["output"])
            user_prompt += ")."
        else:
            for id, info in spatial_info.items():
                spatial_str += f"Agent {id} as a {info['role']} his answer to this question is:\n\n{info['output']}\n\n"
            for id, info in temporal_info.items():
                temporal_str += f"Agent {id} as a {info['role']} his answer to this question was:\n\n{info['output']}\n\n"
            user_prompt += f"At the same time, there are the following responses to the same question for your reference:\n\n{spatial_str} \n\n" if len(spatial_str) else ""
            user_prompt += f"In the last round of dialogue, there were the following responses to the same question for your reference: \n\n{temporal_str}" if len(temporal_str) else ""
        return system_prompt, user_prompt
    
    # <--- CHANGED: _execute 重命名为 _run_llm_sync 并调整参数 ---
    def _run_llm_sync(self, processed_inputs: List[Any],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """
        实现 Node 基类要求的同步执行方法。
        """
        # <--- CHANGED: 不再调用 _process_inputs，直接使用传入的 processed_inputs ---
        system_prompt, user_prompt = processed_inputs
        
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        
        # <--- CHANGED: 使用同步的 self.llm.gen ---
        response = self.llm.gen(message)
        
        # <--- NEW: 保持与异步版本逻辑一致 ---
        if self.role == "Programming Expert":
            answer = execute_code_get_return(response.lstrip("```python\n").rstrip("\n```"))
            response += f"\nthe answer is {answer}"
            
        return [response] # <--- CHANGED: 确保返回一个列表 ---

    async def _async_execute(self, processed_inputs:List[Any],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        # ... (此方法代码保持不变，作为参考) ...
        system_prompt, user_prompt = processed_inputs 
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        if self.role == "Programming Expert":
            answer = execute_code_get_return(response.lstrip("```python\n").rstrip("\n```"))
            response += f"\nthe answer is {answer}"
        return [response] # <--- 建议也修改这里，确保返回列表 ---
