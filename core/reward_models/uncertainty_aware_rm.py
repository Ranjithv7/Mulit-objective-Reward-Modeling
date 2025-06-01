import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np

from reward_models.base_reward_model import BaseRewardModel, RewardOutput, RewardType
from .transformer_reward_model import TransformerRewardModel

class UncertaintyAwareRM(TransformerRewardModel):
    def __init__(
        self,
        *args,
        uncertainty_method: str = "mc_dropout",
        num_mc_samples: int = 10,
        epistemic_uncertainty: bool = True,
        aleatoric_uncertainty: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.uncertainty_method = uncertainty_method
        self.num_mc_samples = num_mc_samples
        self.epistemic_uncertainty = epistemic_uncertainty
        self.aleatoric_uncertainty = aleatoric_uncertainty
        
        if uncertainty_method == "variational":
            self.reward_head = VariationalRewardHead(
                self.hidden_size, 
                self.num_objectives, 
                self.dropout
            )
        elif uncertainty_method == "ensemble":
            self.reward_head = EnsembleRewardHead(
                self.hidden_size,
                self.num_objectives,
                num_heads=5,
                dropout=self.dropout
            )
        elif uncertainty_method == "evidential":
            self.reward_head = EvidentialRewardHead(
                self.hidden_size,
                self.num_objectives
            )
        
        if self.aleatoric_uncertainty:
            self.aleatoric_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size // 2, self.num_objectives)
            )
    
    def _create_reward_head(self) -> nn.Module:
        if self.uncertainty_method == "mc_dropout":
            return MCDropoutRewardHead(
                self.hidden_size,
                self.num_objectives,
                self.dropout
            )
        else:
            return super()._create_reward_head()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
        return_uncertainty: bool = True
    ) -> Union[torch.Tensor, RewardOutput]:
        pooled_output, backbone_outputs = self._encode_sequence(input_ids, attention_mask)
        pooled_output = self.dropout_layer(pooled_output)
        
        if self.uncertainty_method == "mc_dropout":
            rewards, epistemic_uncertainty = self._mc_dropout_inference(pooled_output)
        elif self.uncertainty_method == "variational":
            rewards, kl_loss = self.reward_head(pooled_output)
            epistemic_uncertainty = self._compute_variational_uncertainty(pooled_output)
        elif self.uncertainty_method == "ensemble":
            rewards, epistemic_uncertainty = self.reward_head(pooled_output)
        elif self.uncertainty_method == "evidential":
            rewards, epistemic_uncertainty, aleatoric_uncertainty = self.reward_head(pooled_output)
        else:
            rewards = self.reward_head(pooled_output)
            epistemic_uncertainty = torch.zeros_like(rewards)
        
        if self.aleatoric_uncertainty and hasattr(self, 'aleatoric_head'):
            aleatoric_log_var = self.aleatoric_head(pooled_output)
            aleatoric_uncertainty = torch.exp(0.5 * aleatoric_log_var)
        else:
            aleatoric_uncertainty = torch.zeros_like(rewards)
        
        total_uncertainty = torch.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        
        if self.normalize_rewards:
            rewards = self.normalize_reward(rewards)
        
        if not return_dict:
            return rewards
        
        return RewardOutput(
            rewards=rewards,
            uncertainty=total_uncertainty if return_uncertainty else None,
            hidden_states=pooled_output
        )
    
    def _mc_dropout_inference(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train()
        
        samples = []
        for _ in range(self.num_mc_samples):
            sample_reward = self.reward_head(hidden_states)
            samples.append(sample_reward)
        
        self.eval()
        
        samples = torch.stack(samples, dim=0)
        mean_reward = samples.mean(dim=0)
        epistemic_uncertainty = samples.std(dim=0)
        
        return mean_reward, epistemic_uncertainty
    
    def _compute_variational_uncertainty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        samples = []
        for _ in range(self.num_mc_samples):
            sample_reward, _ = self.reward_head(hidden_states)
            samples.append(sample_reward)
        
        samples = torch.stack(samples, dim=0)
        return samples.std(dim=0)
    
    def compute_uncertainty_loss(self, predictions: torch.Tensor, targets: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        precision = 1.0 / (uncertainty**2 + 1e-8)
        mse_loss = (predictions - targets)**2
        uncertainty_loss = 0.5 * (precision * mse_loss + torch.log(uncertainty**2 + 1e-8))
        return uncertainty_loss.mean()

class MCDropoutRewardHead(nn.Module):
    def __init__(self, hidden_size: int, num_objectives: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_objectives = num_objectives
        
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_objectives)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class VariationalRewardHead(nn.Module):
    def __init__(self, hidden_size: int, num_objectives: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_objectives = num_objectives
        
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_objectives)
        )
        
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_objectives)
        )
        
        self.prior_mean = nn.Parameter(torch.zeros(num_objectives))
        self.prior_logvar = nn.Parameter(torch.zeros(num_objectives))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        reward = mean + eps * std
        
        posterior = Normal(mean, std)
        prior = Normal(self.prior_mean.expand_as(mean), torch.exp(0.5 * self.prior_logvar).expand_as(std))
        kl_loss = kl_divergence(posterior, prior).sum(dim=-1).mean()
        
        return reward, kl_loss

class EnsembleRewardHead(nn.Module):
    def __init__(self, hidden_size: int, num_objectives: int = 1, num_heads: int = 5, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_objectives)
            ) for _ in range(num_heads)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = torch.stack([head(x) for head in self.heads], dim=0)
        
        mean_prediction = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.std(dim=0)
        
        return mean_prediction, epistemic_uncertainty

class EvidentialRewardHead(nn.Module):
    def __init__(self, hidden_size: int, num_objectives: int = 1):
        super().__init__()
        self.num_objectives = num_objectives
        
        self.evidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 4 * num_objectives)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        evidence = self.evidence_head(x)
        evidence = evidence.view(-1, self.num_objectives, 4)
        
        gamma, nu, alpha, beta = evidence[..., 0], evidence[..., 1], evidence[..., 2], evidence[..., 3]
        
        gamma = F.softplus(gamma)
        nu = F.softplus(nu) + 1
        alpha = F.softplus(alpha) + 1
        beta = F.softplus(beta)
        
        mean = gamma
        aleatoric_uncertainty = beta / (alpha - 1)
        epistemic_uncertainty = beta / ((alpha - 1) * nu)
        
        return mean, epistemic_uncertainty, aleatoric_uncertainty

class DeepEnsembleRM(nn.Module):
    def __init__(
        self,
        base_model_class: type,
        num_models: int = 5,
        *args,
        **kwargs
    ):
        super().__init__()
        self.num_models = num_models
        
        self.models = nn.ModuleList([
            base_model_class(*args, **kwargs) 
            for _ in range(num_models)
        ])
    
    def forward(self, *args, **kwargs) -> RewardOutput:
        outputs = [model(*args, **kwargs) for model in self.models]
        
        rewards = torch.stack([out.rewards for out in outputs], dim=0)
        mean_reward = rewards.mean(dim=0)
        epistemic_uncertainty = rewards.std(dim=0)
        
        return RewardOutput(
            rewards=mean_reward,
            uncertainty=epistemic_uncertainty
        )
    
    def compute_disagreement(self, *args, **kwargs) -> torch.Tensor:
        outputs = [model(*args, **kwargs) for model in self.models]
        rewards = torch.stack([out.rewards for out in outputs], dim=0)
        
        pairwise_disagreements = []
        for i in range(self.num_models):
            for j in range(i + 1, self.num_models):
                disagreement = torch.abs(rewards[i] - rewards[j])
                pairwise_disagreements.append(disagreement)
        
        return torch.stack(pairwise_disagreements, dim=0).mean(dim=0)

class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_rho = nn.Parameter(torch.randn(out_features) * 0.1)
        
        self.prior_std = prior_std
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)
        
        weight = self.weight_mu + weight_std * weight_eps
        bias = self.bias_mu + bias_std * bias_eps
        
        output = F.linear(x, weight, bias)
        
        kl_weight = self._kl_divergence(self.weight_mu, weight_std, self.prior_std)
        kl_bias = self._kl_divergence(self.bias_mu, bias_std, self.prior_std)
        kl_loss = kl_weight + kl_bias
        
        return output, kl_loss
    
    def _kl_divergence(self, mu: torch.Tensor, std: torch.Tensor, prior_std: float) -> torch.Tensor:
        var = std**2
        prior_var = prior_std**2
        
        kl = 0.5 * (torch.log(prior_var / var) + (var + mu**2) / prior_var - 1)
        return kl.sum()

class BayesianRewardHead(nn.Module):
    def __init__(self, hidden_size: int, num_objectives: int = 1, prior_std: float = 1.0):
        super().__init__()
        self.layer1 = BayesianLinear(hidden_size, hidden_size // 2, prior_std)
        self.layer2 = BayesianLinear(hidden_size // 2, num_objectives, prior_std)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, kl1 = self.layer1(x)
        x = F.relu(x)
        x, kl2 = self.layer2(x)
        
        total_kl = kl1 + kl2
        return x, total_kl

class UncertaintyCalibration:
    @staticmethod
    def compute_calibration_error(uncertainties: torch.Tensor, errors: torch.Tensor, n_bins: int = 10) -> float:
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (uncertainties > bin_lower.item()) & (uncertainties <= bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = (errors[in_bin] <= uncertainties[in_bin]).float().mean()
                avg_uncertainty_in_bin = uncertainties[in_bin].mean()
                ece += torch.abs(avg_uncertainty_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()
    
    @staticmethod
    def temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        return logits / temperature
    
    @staticmethod
    def platt_scaling(scores: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
        from sklearn.linear_model import LogisticRegression
        
        lr = LogisticRegression()
        lr.fit(scores.cpu().numpy().reshape(-1, 1), labels.cpu().numpy())
        
        return lr.coef_[0][0], lr.intercept_[0]

class UncertaintyMetrics:
    @staticmethod
    def mutual_information(predictions: torch.Tensor) -> torch.Tensor:
        mean_pred = predictions.mean(dim=0)
        entropy_mean = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(dim=-1)
        
        mean_entropy = 0
        for pred in predictions:
            entropy = -(pred * torch.log(pred + 1e-8)).sum(dim=-1)
            mean_entropy += entropy
        mean_entropy /= len(predictions)
        
        return entropy_mean - mean_entropy
    
    @staticmethod
    def predictive_entropy(predictions: torch.Tensor) -> torch.Tensor:
        mean_pred = predictions.mean(dim=0)
        return -(mean_pred * torch.log(mean_pred + 1e-8)).sum(dim=-1)
    
    @staticmethod
    def variance_of_expected(predictions: torch.Tensor) -> torch.Tensor:
        return predictions.var(dim=0).sum(dim=-1)
    
    @staticmethod
    def expected_pairwise_kl(predictions: torch.Tensor) -> torch.Tensor:
        n_samples = predictions.shape[0]
        total_kl = 0
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                kl_div = F.kl_div(
                    F.log_softmax(predictions[i], dim=-1),
                    F.softmax(predictions[j], dim=-1),
                    reduction='sum'
                )
                total_kl += kl_div
        
        return total_kl / (n_samples * (n_samples - 1) / 2)