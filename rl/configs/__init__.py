import yaml
import os

def load_rl_config(config_path: str) -> dict:
    """
    加载RL配置文件（YAML格式）

    Args:
        config_path (str): 配置文件路径

    Returns:
        dict: 配置字典
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在！")

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    return config
