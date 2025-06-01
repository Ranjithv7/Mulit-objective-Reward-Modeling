import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

class RewardType(Enum):
    SCALAR = "scalar"
    MULTI_OBJECTIVE = "multi_objective"
    DISTRIBUTIONAL = "distributional"
    PROCESS_LEVEL = "process_level"

@dataclass
class RewardOutput:
    rewards: torch.Tensor
    uncertainty: Optional[torch.Tensor] = None
    process_rewards: Optional[List[torch.Tensor]] = None
    objective_breakdown: Optional[Dict[str, torch.Tensor]] = None
    attention_weights: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None

@dataclass
class PreferenceData:
    chosen: torch.Tensor
    rejected: torch.Tensor
    chosen_attention_mask: Optional[torch.Tensor] = None
    rejected_attention_mask: Optional[torch.Tensor] = None
    margin: Optional[float] = None

class BaseRewardModel(nn.Module, ABC):
    def __init__(
        self,
        model_name_or_path: str,
        reward_type: RewardType = RewardType.SCALAR,
        num_objectives: int = 1,
        hidden_size: int = 768,
        dropout: float = 0.1,
        normalize_rewards: bool = True
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.reward_type = reward_type
        self.num_objectives = num_objectives
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.normalize_rewards = normalize_rewards
        
        self.reward_head = self._create_reward_head()
        self.dropout_layer = nn.Dropout(dropout)
        
    @abstractmethod
    def _create_reward_head(self) -> nn.Module:
        pass
    
    @abstractmethod
    def _encode_sequence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        return_dict: bool = True
    ) -> Union[torch.Tensor, RewardOutput]:
        pass
    
    def compute_preference_loss(
        self, 
        chosen_rewards: torch.Tensor, 
        rejected_rewards: torch.Tensor,
        margin: float = 0.0
    ) -> torch.Tensor:
        if self.reward_type == RewardType.MULTI_OBJECTIVE:
            return self._compute_multi_objective_preference_loss(chosen_rewards, rejected_rewards, margin)
        elif self.reward_type == RewardType.DISTRIBUTIONAL:
            return self._compute_distributional_preference_loss(chosen_rewards, rejected_rewards, margin)
        else:
            return self._compute_scalar_preference_loss(chosen_rewards, rejected_rewards, margin)
    
    def _compute_scalar_preference_loss(
        self, 
        chosen_rewards: torch.Tensor, 
        rejected_rewards: torch.Tensor,
        margin: float = 0.0
    ) -> torch.Tensor:
        return -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards - margin)).mean()
    
    def _compute_multi_objective_preference_loss(
        self, 
        chosen_rewards: torch.Tensor, 
        rejected_rewards: torch.Tensor,
        margin: float = 0.0
    ) -> torch.Tensor:
        reward_diff = chosen_rewards - rejected_rewards - margin
        objective_losses = -torch.log(torch.sigmoid(reward_diff))
        return objective_losses.mean()
    
    def _compute_distributional_preference_loss(
        self, 
        chosen_rewards: torch.Tensor, 
        rejected_rewards: torch.Tensor,
        margin: float = 0.0
    ) -> torch.Tensor:
        chosen_mean = chosen_rewards.mean(dim=-1)
        rejected_mean = rejected_rewards.mean(dim=-1)
        return self._compute_scalar_preference_loss(chosen_mean, rejected_mean, margin)
    
    def compute_ranking_loss(self, rewards: List[torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(len(rewards)):
            for j in range(i + 1, len(rewards)):
                better_reward = rewards[i]
                worse_reward = rewards[j]
                loss = -torch.log(torch.sigmoid(better_reward - worse_reward))
                total_loss += loss.mean()
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)
    
    def normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        if not self.normalize_rewards:
            return reward
            
        if self.reward_type == RewardType.MULTI_OBJECTIVE:
            return torch.tanh(reward)
        else:
            return torch.sigmoid(reward)
    
    def get_reward_statistics(self, rewards: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.reward_type == RewardType.MULTI_OBJECTIVE:
            return {
                f"objective_{i}_mean": rewards[:, i].mean() 
                for i in range(self.num_objectives)
            } | {
                f"objective_{i}_std": rewards[:, i].std() 
                for i in range(self.num_objectives)
            }
        else:
            return {
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
                "reward_min": rewards.min(),
                "reward_max": rewards.max()
            }
    
    def compute_uncertainty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(hidden_states[:, :1])
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        return None
    
    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if 'reward_head' not in name:
                param.requires_grad = False
    
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def get_model_config(self) -> Dict:
        return {
            "model_name_or_path": self.model_name_or_path,
            "reward_type": self.reward_type.value,
            "num_objectives": self.num_objectives,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "normalize_rewards": self.normalize_rewards
        }

class RewardHead(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_objectives: int = 1,
        head_type: str = "linear",
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_objectives = num_objectives
        self.head_type = head_type
        
        if head_type == "linear":
            self.head = nn.Linear(hidden_size, num_objectives)
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_objectives)
            )
        elif head_type == "residual":
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_objectives)
            )
            self.residual = nn.Linear(hidden_size, num_objectives)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.head_type == "residual":
            return self.head(hidden_states) + self.residual(hidden_states)
        return self.head(hidden_states)

class MultiHeadRewardModel(BaseRewardModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.objective_heads = nn.ModuleDict({
            f"head_{i}": RewardHead(self.hidden_size, 1, "mlp", self.dropout)
            for i in range(self.num_objectives)
        })
    
    def _create_reward_head(self) -> nn.Module:
        return nn.Identity()
    
    def compute_multi_head_rewards(self, hidden_states: torch.Tensor) -> torch.Tensor:
        objective_rewards = []
        for head in self.objective_heads.values():
            objective_rewards.append(head(hidden_states))
        return torch.cat(objective_rewards, dim=-1)

class PreferenceCollector:
    @staticmethod
    def create_preference_pairs(
        sequences: List[str],
        rewards: torch.Tensor,
        k_best: int = 5,
        k_worst: int = 5
    ) -> List[Tuple[str, str]]:
        sorted_indices = torch.argsort(rewards, descending=True)
        best_indices = sorted_indices[:k_best]
        worst_indices = sorted_indices[-k_worst:]
        
        pairs = []
        for best_idx in best_indices:
            for worst_idx in worst_indices:
                pairs.append((sequences[best_idx], sequences[worst_idx]))
        
        return pairs
    
    @staticmethod
    def bootstrap_preferences(
        preferences: List[PreferenceData],
        n_bootstrap: int = 1000
    ) -> List[PreferenceData]:
        indices = torch.randint(0, len(preferences), (n_bootstrap,))
        return [preferences[i] for i in indices]