import json

# 读取原始 JSON 文件（list of dicts）
with open('test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 写入为 .jsonl 文件（每行一个 JSON 对象）
with open('output.jsonl', 'w', encoding='utf-8') as f:
    for item in data:
        line = {
            "question": item["Body"].strip() + " " + item["Question"].strip(),
            "answer": str(item["Answer"]).strip()
        }
        f.write(json.dumps(line, ensure_ascii=False) + '\n')
