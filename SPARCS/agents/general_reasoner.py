from typing import List, Any, Dict
from SPARCS.graph.node import Node
from SPARCS.agents.agent_registry import AgentRegistry
from SPARCS.llm.llm_registry import LLMRegistry

@AgentRegistry.register('GeneralReasoner')
class GeneralReasoner(Node):
    def __init__(self, id: str | None = None, role: str = "General Reasoning Agent", domain: str = "", llm_name: str = ""):
        super().__init__(id, "GeneralReasoner", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.role = role

        # ✅ 在此写死 System Prompt（constraint）
        self.system_prompt = (
            "You are a helpful, intelligent agent capable of general reasoning, logical thinking, "
            "and solving academic or factual questions clearly. "
            "Always choose the best answer using your understanding and any given context."
        )

    def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs) -> List[Any]:
        question = raw_inputs[0]["task"]
        user_prompt = f"As a {self.role}, answer the following question:\n\n{question}\n\n"

        # 加入空间信息
        if spatial_info:
            user_prompt += "Here are other agents' responses:\n"
            for id, info in spatial_info.items():
                user_prompt += f"\nAgent {id} ({info['role']}):\n{info['output']}\n"

        # 加入时间信息
        if temporal_info:
            user_prompt += "\nIn the previous round, agents said:\n"
            for id, info in temporal_info.items():
                user_prompt += f"\nAgent {id} ({info['role']}):\n{info['output']}\n"

        return self.system_prompt, user_prompt.strip()

    def _run_llm_sync(self, processed_inputs: List[Any], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        system_prompt, user_prompt = processed_inputs
        message = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        response = self.llm.gen(message)
        return [response]

    async def _async_execute(self, processed_inputs: List[Any], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        system_prompt, user_prompt = processed_inputs
        message = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        response = await self.llm.agen(message)
        return [response]
