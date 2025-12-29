from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 使用本地模型路径
model_path = "/home/shareuser/JJ327/DropoutRL/Qwen3-8B"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 若支持 bfloat16 可改为 bfloat16，否则使用 float16
    device_map="auto",
    trust_remote_code=True
)

# 设置 prompt（问题）
prompt = """
Bill is trying to dig a well in his backyard. He can dig 4 feet\/hour through soil and half that fast through clay. If he has to dig through 24 feet of soil and 8 feet of clay, how long will it take him to dig the well?

这是一个问题。我现在有若干个智能体可供选用，请根据题目，给出这智能体的通信结构，数量是n,n的取值范围是1—5；智能体的类型为math_solver;  language_understading 在保持正确率的同时尽可能少通信。用一个n*n的矩阵代表通信结构，请注意每个智能体接受的是 {question} 和 {别的智能体的分析}。矩阵中：1 代表通信，0 代表不通信。

请输出 JSON 格式，格式参考如下：

{
"agent_count": n,
"agent_type": "",
"communication_matrix": n*n
}

请在简要分析后直接输出 JSON，不需要展开详细思考。

"""

# 构建 chat 格式
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # 如模型不支持可去掉
)

# 构建输入张量
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 文本生成
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=10240000,  # 可根据任务适当调整
    do_sample=False,
    temperature=0.7
)

# 提取生成部分（去掉 prompt 部分）
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# 尝试提取 thinking 和最终内容
try:
    end_think_token = tokenizer.convert_tokens_to_ids("</think>")
    index = len(output_ids) - output_ids[::-1].index(end_think_token)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# 打印输出
print("Thinking content:\n", thinking_content)
print("\nFinal content:\n", content)
