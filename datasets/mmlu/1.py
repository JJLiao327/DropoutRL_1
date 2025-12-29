import pandas as pd
import json
import numpy as np

# 输入 parquet 文件路径
input_path = "test-00000-of-00001.parquet"

# 输出 jsonl 文件路径
output_path = "converted_test.jsonl"

# 要保留的列
columns_to_keep = ['question', 'subject', 'choices', 'answer']

# 加载 parquet
df = pd.read_parquet(input_path)

# 保留所需字段
df = df[columns_to_keep]

# 将所有值转为原生 Python 类型（特别是 ndarray 转 list）
def convert_row(row):
    return {
        "question": str(row["question"]),
        "subject": str(row["subject"]),
        "choices": list(row["choices"]) if isinstance(row["choices"], (list, tuple, np.ndarray)) else row["choices"],
        "answer": str(row["answer"])
    }

# 写入 JSONL
with open(output_path, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        json_obj = convert_row(row)
        f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

print(f"✅ 成功转换为 JSONL，共 {len(df)} 条，文件保存到: {output_path}")
