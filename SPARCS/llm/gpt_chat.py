import asyncio
import aiohttp
from typing import List, Union, Optional, Dict
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
import async_timeout
import os
import torch

# ✅ 新增导入: transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("警告: transformers 未安装，本地模型功能将不可用。请运行: pip install transformers torch")
    AutoModelForCausalLM = None
    AutoTokenizer = None

# 假设您的项目结构中有这些模块
from SPARCS.llm.format import Message
from SPARCS.llm.llm import LLM
from SPARCS.llm.llm_registry import LLMRegistry

# --- API 配置 ---
load_dotenv()
MINE_BASE_URL = "https://openrouter.ai/api/v1"
MINE_API_KEYS = "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"

# --- 同步 API 调用函数 (保持不变) ---
@retry(wait=wait_random_exponential(max=60), stop=stop_after_attempt(3))
def chat_sync(model: str, msg: List[Dict], **kwargs) -> str:
    try:
        client = OpenAI(api_key=MINE_API_KEYS, base_url=MINE_BASE_URL)
        print(f"  [Debug] 发送给 API 的参数: {kwargs}")
        completion = client.chat.completions.create(
            model=model,
            messages=msg,
            timeout=120.0,
            **kwargs 
        )
        response_message = completion.choices[0].message.content
        if isinstance(response_message, str):
            return response_message
        else:
            raise ValueError("API 返回了非字符串格式的响应。")
    except Exception as e:
        print(f"错误: 同步 API 调用失败: {e}")
        raise

# --- 异步 API 调用函数 (保持不变) ---
@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(model: str, msg: List[Dict], **kwargs):
    api_kwargs = dict(api_key = MINE_API_KEYS, base_url = MINE_BASE_URL)
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(
                model=model,
                messages=msg,
                **kwargs
            )
        response_message = completion.choices[0].message.content
        if isinstance(response_message, str):
            return response_message
    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")

# --- GPTChat 类 (核心修改在这里) ---

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_local = False
        self.model = None
        self.tokenizer = None

        # ✅ 新增: 检查是否为本地模型路径
        if AutoModelForCausalLM and os.path.isdir(self.model_name):
            print(f"检测到本地模型路径，正在加载: {self.model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map="auto"  # 自动分配到 GPU 或 CPU
                )
                self.is_local = True
                print(f"✅ 本地模型 {self.model_name} 加载成功。")
            except Exception as e:
                print(f"❌ 本地模型加载失败: {e}. 将退回使用 API 模式。")
                self.is_local = False
        
        if not self.is_local:
            print(f"GPTChat 初始化完成，将使用 API 模型: {self.model_name}")

    def _prepare_local_generation(self, messages: List[Message], **kwargs):
        """为本地模型准备输入和生成参数"""
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        
        # 将 Message 对象转换为字典列表
        dict_messages = [msg if isinstance(msg, dict) else msg.dict() for msg in messages]

        inputs = self.tokenizer.apply_chat_template(
            dict_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        default_params = {
            'max_new_tokens': 2048,
            'do_sample': True,
            'temperature': 0.2,
            'top_p': 0.95,
        }
        default_params.update(kwargs)
        
        return inputs, default_params
    
    def gen(self, messages: List[Message], **kwargs) -> str:
        """同步生成方法，支持本地模型和 API"""
        if self.is_local:
            # --- 本地模型生成逻辑 ---
            print(f"GPTChat.gen (同步/本地) 被调用，模型: {self.model_name}")
            try:
                inputs, gen_kwargs = self._prepare_local_generation(messages, **kwargs)
                outputs = self.model.generate(inputs, **gen_kwargs)
                response_text = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                return response_text
            except Exception as e:
                print(f"错误: 在 GPTChat.gen (本地) 中生成失败: {e}")
                return f"<LOCAL_MODEL_ERROR: {e}>"
        else:
            # --- API 调用逻辑 (保持不变) ---
            print(f"GPTChat.gen (同步/API) 被调用，模型: {self.model_name}")
            if isinstance(messages, str):
                messages = [Message(role="user", content=messages)]
            
            dict_messages = [msg if isinstance(msg, dict) else msg.dict() for msg in messages]
            
            default_params = {'temperature': 0.2, 'top_p': 0.95, 'max_tokens': 2048}
            final_params = {**default_params, **kwargs}

            try:
                return chat_sync(self.model_name, dict_messages, **final_params)
            except Exception as e:
                print(f"错误: 在 GPTChat.gen (API) 中，chat_sync 最终失败: {e}")
                return f"<API_ERROR: {e}>"

    async def agen(self, messages: List[Message], **kwargs) -> str:
        """异步生成方法，支持本地模型和 API"""
        if self.is_local:
            # --- 本地模型异步生成 (通过在线程池中运行同步代码实现) ---
            print(f"GPTChat.agen (异步/本地) 被调用，模型: {self.model_name}")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,  # 使用默认的线程池执行器
                self.gen,  # 调用同步的 gen 方法
                messages,
                kwargs
            )
        else:
            # --- 异步 API 调用逻辑 (保持不变) ---
            print(f"GPTChat.agen (异步/API) 被调用，模型: {self.model_name}")
            if isinstance(messages, str):
                messages = [Message(role="user", content=messages)]
            
            dict_messages = [msg if isinstance(msg, dict) else msg.dict() for msg in messages]

            final_kwargs = {'temperature': 0.2, 'top_p': 0.95, 'max_tokens': 2048}
            final_kwargs.update(kwargs)

            return await achat(self.model_name, dict_messages, **final_kwargs)

