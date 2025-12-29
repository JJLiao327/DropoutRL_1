# RL/ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F # 引入 F

# --- Actor-Critic 网络定义 (适配方案二：连续输出概率 + Bernoulli评估) ---
class ActorCritic(nn.Module):
    """
    Actor-Critic 网络结构。
    输入: state 特征向量
    输出:
        - actor: 每个连接的保留概率 (N*N 矩阵，值在 (0, 1))
        - critic: 状态价值估计 (标量)
    """
    def __init__(self, state_dim, action_dim):
        """
        初始化 Actor-Critic 网络。

        Args:
            state_dim (int): 输入状态特征的维度。
            action_dim (int): 输出动作空间的维度。对于 N*N Mask，这里是 N*N。
        """
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim # 保存 action_dim (N*N)

        # Actor 网络 (策略网络)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim), # 输出 N*N 个 logits
            nn.Sigmoid()                 # 将 logits 转换为 (0, 1) 之间的概率
        )

        # Critic 网络 (价值网络)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 输出单一的状态价值
        )

    def forward(self, state):
        """
        前向传播，计算每个连接的保留概率和状态价值。

        Args:
            state (torch.Tensor): 输入的状态张量，形状 [batch_size, state_dim]。

        Returns:
            tuple: (action_probs, state_value)
                   action_probs: Actor 网络输出的概率张量，形状 [batch_size, action_dim] (即 [batch_size, N*N])。
                   state_value: Critic 网络输出的状态价值张量，形状 [batch_size, 1]。
        """
        # 确保输入是 FloatTensor
        if not isinstance(state, torch.Tensor):
             state = torch.tensor(state, dtype=torch.float32)
        elif state.dtype != torch.float32:
             state = state.float()
        # 确保 state 是二维的 [batch_size, state_dim]
        if state.dim() == 1:
            state = state.unsqueeze(0) # 如果是单个样本，增加 batch 维度

        action_probs = self.actor(state) # 输出形状 [batch_size, N*N]
        state_value = self.critic(state)  # 输出形状 [batch_size, 1]
        return action_probs, state_value

    def evaluate_actions(self, state, actions):
        """
        在给定状态下评估采取特定动作 (0/1 Mask) 的价值、对数概率和熵。
        主要用于 PPO 的损失计算。

        Args:
            state (torch.Tensor): 输入的状态张量，形状 [batch_size, state_dim]。
            actions (torch.Tensor): 实际采取的动作张量 (0/1 Mask)，
                                     形状应为 [batch_size, action_dim] (即 [batch_size, N*N])。

        Returns:
            tuple: (state_value, action_log_probs, entropy)
                   state_value: Critic 对输入状态的价值评估，形状 [batch_size, 1]。
                   action_log_probs: 采取 actions 的联合对数概率 (标量或 [batch_size])。
                   entropy: 策略的熵 (标量或 [batch_size])。
        """
        action_probs, state_value = self.forward(state) # action_probs: [batch_size, N*N]

        # 确保 actions 的形状和类型正确
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)
        elif actions.dtype != torch.float32:
            actions = actions.float() # Bernoulli 的 log_prob 需要 float 输入
        if actions.dim() == 1 and actions.shape[0] == self.action_dim: # 单个样本动作
            actions = actions.unsqueeze(0) # 增加 batch 维度 -> [1, N*N]
        elif actions.dim() != 2 or actions.shape[1] != self.action_dim:
             raise ValueError(f"actions 的形状应为 [batch_size, {self.action_dim}], 但得到 {actions.shape}")

        # 使用 Bernoulli 分布来计算对数概率和熵
        # Clamp action_probs 防止概率为 0 或 1 导致 log(0) 或 log(1-1)
        clamped_probs = torch.clamp(action_probs, 1e-6, 1.0 - 1e-6)
        dist = torch.distributions.Bernoulli(probs=clamped_probs)

        # 计算联合对数概率：对每个连接的 log_prob 求和
        # dist.log_prob(actions) 输出形状 [batch_size, N*N]
        # 我们需要对最后一个维度 (N*N) 求和
        action_log_probs = dist.log_prob(actions).sum(dim=-1) # 输出形状 [batch_size]

        # 计算策略的总熵：对每个连接的熵求和
        entropy = dist.entropy().sum(dim=-1) # 输出形状 [batch_size]

        return state_value.squeeze(-1), action_log_probs, entropy # 返回 [batch_size] 形状的值

# --- PPO 算法实现 (需要微调以使用新的 evaluate_actions 返回值) ---
class PPO:
    """
    Proximal Policy Optimization (PPO) 算法的更新逻辑。
    """
    def __init__(self, actor_critic, optimizer, gamma=0.99, clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.01):
        """
        初始化 PPO 算法。
        (参数注释同上)
        """
        self.actor_critic = actor_critic
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.device = next(actor_critic.parameters()).device

    def update(self, rollouts):
        """
        使用收集到的经验 (rollouts) 更新 Actor-Critic 网络。
        (参数注释同上)
        """
        for batch in rollouts:
            states = batch['states'].to(self.device)
            actions = batch['actions'].to(self.device) # actions 是 0/1 Mask, [batch_size, N*N]
            old_log_probs = batch['old_log_probs'].to(self.device) # [batch_size]
            returns = batch['returns'].to(self.device)         # [batch_size]
            advantages = batch['advantages'].to(self.device)   # [batch_size]

            # --- 核心 PPO 损失计算 ---
            # 1. 使用当前策略评估旧经验中的动作，获取价值、新对数概率和熵
            state_values, new_log_probs, entropy = self.actor_critic.evaluate_actions(states, actions)
            # state_values: [batch_size], new_log_probs: [batch_size], entropy: [batch_size]

            # 2. 计算概率比率 ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # 3. 计算替代损失 (Surrogate Loss)
            surrogate_loss_1 = ratio * advantages
            surrogate_loss_2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surrogate_loss_1, surrogate_loss_2).mean() # 对 batch 取平均

            # 4. 计算价值损失 (Value Loss)
            # 确保 state_values 和 returns 形状匹配 (都应为 [batch_size])
            value_loss = F.mse_loss(state_values, returns)

            # 5. 计算熵损失 (Entropy Loss)
            entropy_loss = -entropy.mean() # 最大化熵，损失取负，对 batch 取平均

            # 6. 计算总损失
            total_loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

            # --- 反向传播和优化 ---
            self.optimizer.zero_grad()
            total_loss.backward()
            # nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5) # 可选梯度裁剪
            self.optimizer.step()

