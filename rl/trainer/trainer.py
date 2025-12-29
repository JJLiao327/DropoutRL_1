# RL/trainer/trainer.py
import torch
import torch.optim as optim
import numpy as np
from rl.ppo import ActorCritic, PPO # 从 RL.ppo 导入

class PPOTrainer:
    """
    使用 PPO 算法训练 Actor-Critic 网络的训练器。
    负责与环境交互、收集经验、计算优势和回报，并调用 PPO 更新。
    """
    def __init__(self, state_dim, action_dim, config):
        """
        初始化 PPOTrainer。

        Args:
            state_dim (int): 状态空间维度。
            action_dim (int): 动作空间维度 (N*N)。
            config (dict): 包含超参数的配置字典 (lr, gamma, gae_lambda, etc.)。
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[PPOTrainer] 使用设备: {self.device}", flush=True) # 添加设备信息日志
        self.state_dim = state_dim
        self.action_dim = action_dim # N*N

        # 从配置加载超参数
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95) # GAE lambda 参数
        self.lr = config.get("lr", 3e-4)
        self.clip_param = config.get("clip_param", 0.2)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        print(f"[PPOTrainer] 超参数: gamma={self.gamma}, gae_lambda={self.gae_lambda}, lr={self.lr}, clip_param={self.clip_param}, value_coef={self.value_loss_coef}, entropy_coef={self.entropy_coef}", flush=True) # 打印超参数

        # 创建策略网络 (ActorCritic)
        self.actor_critic = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)
        print(f"[PPOTrainer] ActorCritic 网络和 Adam 优化器已创建。", flush=True)

        # 创建 PPO 更新逻辑实例
        self.ppo = PPO(
            actor_critic=self.actor_critic,
            optimizer=self.optimizer,
            gamma=self.gamma,
            clip_param=self.clip_param,
            value_loss_coef=self.value_loss_coef,
            entropy_coef=self.entropy_coef
        )
        print(f"[PPOTrainer] PPO 更新逻辑已实例化。", flush=True)

        # 经验缓冲区 (Rollout Buffer)
        self.buffer = {
            'states': [],
            'actions': [],      # 存储 0/1 Mask 动作
            'log_probs': [],    # 存储采取该动作的对数概率
            'rewards': [],
            'values': [],       # 存储每个状态的价值估计
            'dones': [],        # 存储是否终止
        }
        print(f"[PPOTrainer] 经验缓冲区已初始化。", flush=True)

    def get_action_and_value(self, state, deterministic=False):
        """
        根据当前策略网络，为给定状态选择动作，并返回动作、对数概率和状态价值。

        Args:
            state (np.ndarray or torch.Tensor): 当前状态。
            deterministic (bool): 是否选择确定性动作 (用于评估)。

        Returns:
            tuple: (action, log_prob, value)
                   action (torch.Tensor): 选择的动作 (0/1 Mask), 形状 [action_dim]。
                   log_prob (torch.Tensor): 该动作的对数概率 (标量)。
                   value (torch.Tensor): 该状态的价值估计 (标量)。
        """
        # --- 添加日志：方法进入 ---
        print(f"[PPOTrainer | get_action] 输入状态形状: {state.shape}", flush=True) # 打印输入状态形状 (如果需要)

        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0) # 确保有 batch 维度

        with torch.no_grad():
            # --- 添加日志：调用网络前 ---
            print(f"[PPOTrainer | get_action] 调用 ActorCritic 网络...", flush=True)
            action_probs, value = self.actor_critic(state_tensor) # probs: [1, N*N], value: [1, 1]
            # --- 添加日志：获取网络输出后 ---
            print(f"[PPOTrainer | get_action] 网络输出 - action_probs 形状: {action_probs.shape}, value 形状: {value.shape}", flush=True)
            print(f"[PPOTrainer | get_action] 网络输出 - action_probs (前10): {action_probs.squeeze(0)[:10].cpu().numpy()}", flush=True) # 打印部分概率

            action_probs = action_probs.squeeze(0) # 移除 batch 维度 -> [N*N]
            value = value.squeeze(0).squeeze(-1)   # 移除所有多余维度 -> 标量

            # 基于概率创建 Bernoulli 分布
            dist = torch.distributions.Bernoulli(probs=action_probs)

            # 选择动作
            if deterministic:
                action = (action_probs > 0.5).float() # 确定性：取 >0.5
                # --- 添加日志：确定性动作 ---
                print(f"[PPOTrainer | get_action] 选择确定性动作 (基于 >0.5 阈值)。", flush=True)
            else:
                action = dist.sample() # 随机性：从分布中采样
                # --- 添加日志：随机采样动作 ---
                print(f"[PPOTrainer | get_action] 从 Bernoulli 分布随机采样动作。", flush=True)

            # 计算所选动作的对数概率
            # log_prob 需要对 N*N 个独立决策的 log_prob 求和
            log_prob = dist.log_prob(action).sum() # 标量

            # --- 添加日志：返回前 ---
            print(f"[PPOTrainer | get_action] 返回 - action (sum): {action.sum().item()}, log_prob: {log_prob.item():.4f}, value: {value.item():.4f}", flush=True) # 打印动作摘要和log_prob/value

        # 返回 CPU 上的张量，方便主程序处理和存储
        return action.cpu(), log_prob.cpu(), value.cpu()

    def store_experience(self, state, action, log_prob, reward, value, done):
        """
        将一次交互的经验存储到缓冲区中。
        (代码无逻辑变化)
        """
        self.buffer['states'].append(torch.tensor(state, dtype=torch.float32))
        self.buffer['actions'].append(action) # action 已经是 tensor
        self.buffer['log_probs'].append(log_prob) # log_prob 已经是 tensor
        self.buffer['rewards'].append(torch.tensor(reward, dtype=torch.float32))
        self.buffer['values'].append(value) # value 已经是 tensor
        self.buffer['dones'].append(torch.tensor(done, dtype=torch.float32))

    def _compute_gae_returns(self, last_value):
        """
        使用 GAE 计算优势和回报。
        (代码无逻辑变化)
        """
        rewards = torch.stack(self.buffer['rewards']).to(self.device)
        values = torch.stack(self.buffer['values']).to(self.device)
        dones = torch.stack(self.buffer['dones']).to(self.device)
        num_steps = len(rewards)
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0

        # 从后往前计算 GAE
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value.to(self.device) # 使用传入的 last_value
            else:
                next_non_terminal = 1.0 - dones[t+1] # 注意这里用 t+1 的 done
                next_value = values[t+1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages.cpu(), returns.cpu() # 返回 CPU 上的张量

    def update(self):
        """
        使用缓冲区中的经验计算 GAE 和回报，并调用 PPO 更新策略。
        """
        buffer_size = len(self.buffer['states'])
        if buffer_size == 0:
            print("[PPOTrainer | update] 缓冲区为空，跳过更新。", flush=True)
            return

        # --- 添加日志：更新开始 ---
        print(f"\n[PPOTrainer | update] 开始策略更新，缓冲区大小: {buffer_size}", flush=True)

        # 1. 计算最后一个状态的价值 (如果需要)
        with torch.no_grad():
            last_state_tensor = self.buffer['states'][-1].unsqueeze(0).to(self.device)
            _, last_value = self.actor_critic(last_state_tensor)
            last_value = last_value.squeeze()
            if self.buffer['dones'][-1].item() > 0.5: # 如果最后一步是终止状态
                last_value = torch.tensor(0.0).to(self.device)
            # --- 添加日志：最后一个状态价值 ---
            # print(f"[PPOTrainer | update] 估计的最后一个状态价值 (用于 GAE): {last_value.item():.4f}", flush=True)

        # 2. 计算 GAE 优势和回报
        # --- 添加日志：计算 GAE 前 ---
        print(f"[PPOTrainer | update] 计算 GAE 优势和回报...", flush=True)
        advantages, returns = self._compute_gae_returns(last_value)
        # --- 添加日志：计算 GAE 后 ---
        print(f"[PPOTrainer | update] GAE 计算完毕。Advantages mean: {advantages.mean().item():.4f}, Returns mean: {returns.mean().item():.4f}", flush=True)


        # 3. 准备 PPO 更新所需的数据 batch
        batch = {
            'states': torch.stack(self.buffer['states']),
            'actions': torch.stack(self.buffer['actions']),
            'old_log_probs': torch.stack(self.buffer['log_probs']),
            'returns': returns,
            'advantages': advantages,
        }
        # --- 添加日志：准备 Batch ---
        print(f"[PPOTrainer | update] PPO 更新 Batch 已准备好。", flush=True)

        # 4. 调用 PPO 进行更新
        # --- 添加日志：调用 PPO 更新前 ---
        print(f"[PPOTrainer | update] 调用 self.ppo.update() 进行网络参数更新...", flush=True)
        update_info = self.ppo.update([batch]) # PPO.update 内部会将数据移到 device，并返回可能的 loss 信息
        # --- 添加日志：调用 PPO 更新后 ---
        print(f"[PPOTrainer | update] self.ppo.update() 调用完成。", flush=True)
        # 可以打印 PPO.update 返回的损失信息 (如果它返回的话)
        # if update_info:
        #     print(f"[PPOTrainer | update] 更新损失信息: {update_info}", flush=True)


        # 5. 清空缓冲区
        self.buffer = {key: [] for key in self.buffer}
        # --- 添加日志：清空缓冲区 ---
        print(f"[PPOTrainer | update] 经验缓冲区已清空。", flush=True)
        print(f"[PPOTrainer | update] 策略更新完成。\n", flush=True)
    

    def save(self, path):
        """
        保存当前策略网络和优化器的参数到指定路径
        """
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"[PPOTrainer] 模型已保存到 {path}")

    def load(self, path):
        """
        从指定路径加载策略网络和优化器的参数
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[PPOTrainer] 模型已从 {path} 加载")


    # --- 兼容性接口 (可选) ---
    # def policy(self, state): ...
    # def select_action(self, state): ...
