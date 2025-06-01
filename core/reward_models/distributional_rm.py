import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import math

from .base_reward_model import BaseRewardModel, RewardOutput, RewardType
from .transformer_reward_model import TransformerRewardModel

class DistributionalRewardModel(TransformerRewardModel):
    def __init__(
        self,
        *args,
        distribution_type: str = "c51",
        num_atoms: int = 51,
        num_quantiles: int = 32,
        v_min: float = -10.0,
        v_max: float = 10.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.distribution_type = distribution_type
        self.num_atoms = num_atoms
        self.num_quantiles = num_quantiles
        self.v_min = v_min
        self.v_max = v_max
        
        if distribution_type == "c51":
            self.delta_z = (v_max - v_min) / (num_atoms - 1)
            self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        elif distribution_type == "qr_dqn":
            tau = torch.arange(0, num_quantiles + 1, dtype=torch.float32) / num_quantiles
            self.register_buffer('tau', (tau[:-1] + tau[1:]) / 2)
        elif distribution_type == "iqn":
            self.embedding_dim = 64
            self.quantile_embedding = nn.Linear(self.embedding_dim, self.hidden_size)
    
    def _create_reward_head(self) -> nn.Module:
        if self.distribution_type == "c51":
            return C51RewardHead(self.hidden_size, self.num_objectives, self.num_atoms)
        elif self.distribution_type == "qr_dqn":
            return QRDQNRewardHead(self.hidden_size, self.num_objectives, self.num_quantiles)
        elif self.distribution_type == "iqn":
            return IQNRewardHead(self.hidden_size, self.num_objectives, self.embedding_dim)
        elif self.distribution_type == "fqf":
            return FQFRewardHead(self.hidden_size, self.num_objectives, self.num_quantiles)
        else:
            raise ValueError(f"Unknown distribution_type: {self.distribution_type}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
        tau: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, RewardOutput]:
        pooled_output, backbone_outputs = self._encode_sequence(input_ids, attention_mask)
        pooled_output = self.dropout_layer(pooled_output)
        
        if self.distribution_type == "c51":
            logits = self.reward_head(pooled_output)
            probabilities = F.softmax(logits, dim=-1)
            expected_rewards = torch.sum(probabilities * self.support.unsqueeze(0).unsqueeze(0), dim=-1)
            distribution_params = probabilities
            
        elif self.distribution_type == "qr_dqn":
            quantiles = self.reward_head(pooled_output)
            expected_rewards = quantiles.mean(dim=-1)
            distribution_params = quantiles
            
        elif self.distribution_type == "iqn":
            if tau is None:
                tau = torch.rand(pooled_output.size(0), self.num_quantiles, device=pooled_output.device)
            quantiles = self.reward_head(pooled_output, tau)
            expected_rewards = quantiles.mean(dim=-1)
            distribution_params = quantiles
            
        elif self.distribution_type == "fqf":
            quantiles, quantile_fractions = self.reward_head(pooled_output)
            expected_rewards = torch.sum(quantiles * quantile_fractions, dim=-1)
            distribution_params = (quantiles, quantile_fractions)
        
        if self.normalize_rewards:
            expected_rewards = self.normalize_reward(expected_rewards)
        
        if not return_dict:
            return expected_rewards
        
        return RewardOutput(
            rewards=expected_rewards,
            hidden_states=pooled_output,
            objective_breakdown={"distribution_params": distribution_params}
        )
    
    def compute_distributional_loss(
        self,
        dist_pred: torch.Tensor,
        rewards: torch.Tensor,
        tau: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.distribution_type == "c51":
            return self._compute_c51_loss(dist_pred, rewards)
        elif self.distribution_type == "qr_dqn":
            return self._compute_quantile_loss(dist_pred, rewards, self.tau)
        elif self.distribution_type == "iqn":
            return self._compute_quantile_loss(dist_pred, rewards, tau)
        elif self.distribution_type == "fqf":
            quantiles, fractions = dist_pred
            return self._compute_fqf_loss(quantiles, fractions, rewards)
    
    def _compute_c51_loss(self, logits: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        batch_size = rewards.size(0)
        
        rewards_clipped = torch.clamp(rewards, self.v_min, self.v_max)
        
        b = (rewards_clipped - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        m = torch.zeros(batch_size, self.num_atoms, device=rewards.device)
        
        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=rewards.device).long().unsqueeze(1)
        
        m.view(-1).index_add_(0, (l + offset).view(-1), (u.float() - b).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1), (b - l.float()).view(-1))
        
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(m * log_probs).sum(dim=-1).mean()
        
        return loss
    
    def _compute_quantile_loss(
        self,
        quantiles: torch.Tensor,
        rewards: torch.Tensor,
        tau: torch.Tensor
    ) -> torch.Tensor:
        quantiles = quantiles.unsqueeze(-1)
        rewards = rewards.unsqueeze(-2)
        tau = tau.unsqueeze(-1)
        
        delta = rewards - quantiles
        huber_loss = F.smooth_l1_loss(quantiles, rewards, reduction='none')
        quantile_loss = torch.abs(tau - (delta < 0).float()) * huber_loss
        
        return quantile_loss.mean()
    
    def _compute_fqf_loss(
        self,
        quantiles: torch.Tensor,
        fractions: torch.Tensor,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        tau = torch.cumsum(fractions, dim=-1)
        tau_hat = F.pad(tau, (1, 0))[:, :-1]
        
        quantile_loss = self._compute_quantile_loss(quantiles, rewards, (tau + tau_hat) / 2)
        
        values_1 = quantiles[:, :-1]
        values_2 = quantiles[:, 1:]
        gradient_penalty = F.smooth_l1_loss(values_1, values_2, reduction='none').mean()
        
        return quantile_loss + 0.1 * gradient_penalty

class C51RewardHead(nn.Module):
    def __init__(self, hidden_size: int, num_objectives: int, num_atoms: int):
        super().__init__()
        self.num_objectives = num_objectives
        self.num_atoms = num_atoms
        
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_objectives * num_atoms)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(x)
        return logits.view(-1, self.num_objectives, self.num_atoms)

class QRDQNRewardHead(nn.Module):
    def __init__(self, hidden_size: int, num_objectives: int, num_quantiles: int):
        super().__init__()
        self.num_objectives = num_objectives
        self.num_quantiles = num_quantiles
        
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_objectives * num_quantiles)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantiles = self.head(x)
        return quantiles.view(-1, self.num_objectives, self.num_quantiles)

class IQNRewardHead(nn.Module):
    def __init__(self, hidden_size: int, num_objectives: int, embedding_dim: int):
        super().__init__()
        self.num_objectives = num_objectives
        self.embedding_dim = embedding_dim
        
        self.quantile_embedding = nn.Linear(embedding_dim, hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_objectives)
        )
    
    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        batch_size, num_quantiles = tau.shape
        
        i = torch.arange(1, self.embedding_dim + 1, dtype=torch.float32, device=x.device)
        cos_embedding = torch.cos(tau.unsqueeze(-1) * i.unsqueeze(0).unsqueeze(0) * math.pi)
        
        quantile_embedding = self.quantile_embedding(cos_embedding)
        
        x_expanded = x.unsqueeze(1).expand(-1, num_quantiles, -1)
        combined = x_expanded * quantile_embedding
        
        quantiles = self.head(combined)
        return quantiles

class FQFRewardHead(nn.Module):
    def __init__(self, hidden_size: int, num_objectives: int, num_quantiles: int):
        super().__init__()
        self.num_objectives = num_objectives
        self.num_quantiles = num_quantiles
        
        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_objectives * num_quantiles)
        )
        
        self.fraction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_objectives * num_quantiles),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quantiles = self.quantile_head(x)
        quantiles = quantiles.view(-1, self.num_objectives, self.num_quantiles)
        
        fractions = self.fraction_head(x)
        fractions = fractions.view(-1, self.num_objectives, self.num_quantiles)
        
        return quantiles, fractions

class RiskSensitiveRewardModel(DistributionalRewardModel):
    def __init__(self, *args, risk_measure: str = "cvar", alpha: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_measure = risk_measure
        self.alpha = alpha
    
    def compute_risk_measure(self, distribution: torch.Tensor) -> torch.Tensor:
        if self.risk_measure == "cvar":
            return self._compute_cvar(distribution)
        elif self.risk_measure == "var":
            return self._compute_var(distribution)
        elif self.risk_measure == "worst_case":
            return distribution.min(dim=-1)[0]
        elif self.risk_measure == "spectral":
            return self._compute_spectral_risk(distribution)
        else:
            raise ValueError(f"Unknown risk measure: {self.risk_measure}")
    
    def _compute_cvar(self, quantiles: torch.Tensor) -> torch.Tensor:
        sorted_quantiles, _ = torch.sort(quantiles, dim=-1)
        num_quantiles = quantiles.size(-1)
        cutoff = int(self.alpha * num_quantiles)
        return sorted_quantiles[:, :, :cutoff].mean(dim=-1)
    
    def _compute_var(self, quantiles: torch.Tensor) -> torch.Tensor:
        sorted_quantiles, _ = torch.sort(quantiles, dim=-1)
        num_quantiles = quantiles.size(-1)
        index = int(self.alpha * num_quantiles)
        return sorted_quantiles[:, :, index]
    
    def _compute_spectral_risk(self, quantiles: torch.Tensor) -> torch.Tensor:
        weights = torch.exp(-torch.arange(quantiles.size(-1), dtype=torch.float32, device=quantiles.device))
        weights = weights / weights.sum()
        
        sorted_quantiles, _ = torch.sort(quantiles, dim=-1)
        return torch.sum(sorted_quantiles * weights.unsqueeze(0).unsqueeze(0), dim=-1)