import glob
import pandas as pd
from typing import Union, List, Literal, Any, Dict
import numpy as np
from abc import ABC
import re

class MMLUDataset(ABC):
    def __init__(self,
        split: Union[Literal['dev'], Literal['val'], Literal['test']],
        ) -> None:

        self._split = split

        data_path = f"datasets/MMLU/data/{self._split}/"
        self._total_df: pd.DataFrame = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'mmlu'

    @staticmethod
    def _load_data(
        data_path: str,
        ) -> pd.DataFrame:

        rng = np.random.default_rng(888)

        csv_paths = glob.glob(data_path + "*.csv")
        csv_paths = sorted(csv_paths)
        print("Number of topics: ", len(csv_paths))

        names = ['question', 'A', 'B', 'C', 'D', 'correct_answer']

        total_df = pd.DataFrame(columns=names)
        for path in csv_paths:
            single_df = pd.read_csv(path, header=None,
                            names=names,encoding='utf-8')
            total_df = pd.concat([total_df, single_df])

        total_df = total_df.reset_index(drop=True)

        # Pseudorandom shuffle
        total_df = total_df.reindex(rng.permutation(total_df.index))

        print("Total number of questions: ", len(total_df))

        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.DataFrame:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_input(record: pd.DataFrame) -> Dict[str, Any]:
        demo_question = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
            )
        input_dict = {"task": demo_question}
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not isinstance(answer, str):
            raise Exception("Expected string")

        match = re.search(r"Answer:?\s*([A-D])", answer, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        answer = answer.strip().upper()
        if answer and answer[0] in "ABCD":
            return answer[0]
        return ""

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['correct_answer']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        return correct_answer


# ✅ 函数接口：用于 run-sync-2.py 处理 JSONL 格式的数据
def mmlu_data_process(data):
    """
    处理 JSONL 格式的 MMLU 数据集：
    {"question": "...", "choices": [...], "answer": "1"}
    """
    processed = []
    option_letters = ['A', 'B', 'C', 'D']
    for item in data:
        question = item.get("question", "")
        choices = item.get("choices", [])
        correct_index = int(item.get("answer", "0"))
        correct_letter = option_letters[correct_index] if correct_index < len(option_letters) else "A"

        task = f"{question}\n"
        for i, choice in enumerate(choices):
            task += f"Option {option_letters[i]}: {choice}\n"

        processed.append({
            "task": task.strip(),
            "answer": correct_letter
        })
    return processed


def mmlu_get_predict(response: str) -> str:
    """
    从模型输出中提取选项字母 A/B/C/D 作为预测结果
    """
    match = re.search(r"Answer:?\s*([A-D])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    response = response.strip().upper()
    if response and response[0] in "ABCD":
        return response[0]
    return ""
