import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
import copy
import math
import numpy as np

from .base_reward_model import BaseRewardModel, RewardOutput, RewardType
from .transformer_reward_model import TransformerRewardModel

class EnsembleRewardModel(nn.Module):
    def __init__(
        self,
        base_model_class: type,
        num_models: int = 5,
        ensemble_method: str = "deep_ensemble",
        diversity_regularization: float = 0.1,
        *args,
        **kwargs
    ):
        super().__init__()
        self.num_models = num_models
        self.ensemble_method = ensemble_method
        self.diversity_regularization = diversity_regularization
        
        if ensemble_method == "deep_ensemble":
            self.models = self._create_deep_ensemble(base_model_class, *args, **kwargs)
        elif ensemble_method == "snapshot_ensemble":
            self.models = self._create_snapshot_ensemble(base_model_class, *args, **kwargs)
        elif ensemble_method == "fast_geometric":
            self.models = self._create_fast_geometric_ensemble(base_model_class, *args, **kwargs)
        elif ensemble_method == "bayesian_ensemble":
            self.models = self._create_bayesian_ensemble(base_model_class, *args, **kwargs)
        else:
            raise ValueError(f"Unknown ensemble_method: {ensemble_method}")
        
        self.aggregation_weights = nn.Parameter(torch.ones(num_models) / num_models)
        self.meta_learner = MetaLearner(len(self.models), kwargs.get('hidden_size', 768))
    
    def _create_deep_ensemble(self, base_model_class: type, *args, **kwargs) -> nn.ModuleList:
        models = nn.ModuleList()
        for i in range(self.num_models):
            kwargs_copy = kwargs.copy()
            kwargs_copy['dropout'] = kwargs.get('dropout', 0.1) + np.random.uniform(-0.05, 0.05)
            model = base_model_class(*args, **kwargs_copy)
            self._initialize_weights_differently(model, i)
            models.append(model)
        return models
    
    def _create_snapshot_ensemble(self, base_model_class: type, *args, **kwargs) -> nn.ModuleList:
        base_model = base_model_class(*args, **kwargs)
        models = nn.ModuleList([copy.deepcopy(base_model) for _ in range(self.num_models)])
        return models
    
    def _create_fast_geometric_ensemble(self, base_model_class: type, *args, **kwargs) -> nn.ModuleList:
        base_model = base_model_class(*args, **kwargs)
        models = nn.ModuleList()
        
        for i in range(self.num_models):
            model = copy.deepcopy(base_model)
            self._apply_geometric_transformation(model, i)
            models.append(model)
        
        return models
    
    def _create_bayesian_ensemble(self, base_model_class: type, *args, **kwargs) -> nn.ModuleList:
        models = nn.ModuleList()
        for i in range(self.num_models):
            kwargs_copy = kwargs.copy()
            kwargs_copy['uncertainty_method'] = 'variational'
            model = base_model_class(*args, **kwargs_copy)
            models.append(model)
        return models
    
    def _initialize_weights_differently(self, model: nn.Module, seed: int):
        torch.manual_seed(seed * 42)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _apply_geometric_transformation(self, model: nn.Module, index: int):
        angle = 2 * math.pi * index / self.num_models
        rotation_factor = 0.1 * math.cos(angle)
        
        with torch.no_grad():
            for param in model.parameters():
                param.data += rotation_factor * torch.randn_like(param.data) * param.data.std()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
        return_ensemble_info: bool = True
    ) -> Union[torch.Tensor, RewardOutput]:
        outputs = []
        for model in self.models:
            output = model(input_ids, attention_mask, return_dict=True)
            outputs.append(output)
        
        if self.ensemble_method == "weighted_average":
            aggregated_reward = self._weighted_average_aggregation(outputs)
        elif self.ensemble_method == "meta_learning":
            aggregated_reward = self._meta_learning_aggregation(outputs, input_ids, attention_mask)
        else:
            aggregated_reward = self._default_aggregation(outputs)
        
        ensemble_uncertainty = self._compute_ensemble_uncertainty(outputs)
        ensemble_diversity = self._compute_ensemble_diversity(outputs)
        
        if not return_dict:
            return aggregated_reward
        
        ensemble_info = {
            "individual_rewards": [out.rewards for out in outputs],
            "ensemble_uncertainty": ensemble_uncertainty,
            "ensemble_diversity": ensemble_diversity,
            "aggregation_weights": F.softmax(self.aggregation_weights, dim=0)
        } if return_ensemble_info else None
        
        return RewardOutput(
            rewards=aggregated_reward,
            uncertainty=ensemble_uncertainty,
            objective_breakdown=ensemble_info
        )
    
    def _default_aggregation(self, outputs: List[RewardOutput]) -> torch.Tensor:
        rewards = torch.stack([out.rewards for out in outputs], dim=0)
        weights = F.softmax(self.aggregation_weights, dim=0)
        return torch.sum(rewards * weights.view(-1, 1, 1), dim=0)
    
    def _weighted_average_aggregation(self, outputs: List[RewardOutput]) -> torch.Tensor:
        rewards = torch.stack([out.rewards for out in outputs], dim=0)
        
        if hasattr(outputs[0], 'uncertainty') and outputs[0].uncertainty is not None:
            uncertainties = torch.stack([out.uncertainty for out in outputs], dim=0)
            precision_weights = 1.0 / (uncertainties + 1e-8)
            precision_weights = precision_weights / precision_weights.sum(dim=0, keepdim=True)
            return torch.sum(rewards * precision_weights, dim=0)
        else:
            return rewards.mean(dim=0)
    
    def _meta_learning_aggregation(
        self,
        outputs: List[RewardOutput],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        rewards = torch.stack([out.rewards for out in outputs], dim=0)
        hidden_states = torch.stack([out.hidden_states for out in outputs], dim=0)
        
        meta_weights = self.meta_learner(hidden_states, rewards)
        meta_weights = F.softmax(meta_weights, dim=0)
        
        return torch.sum(rewards * meta_weights, dim=0)
    
    def _compute_ensemble_uncertainty(self, outputs: List[RewardOutput]) -> torch.Tensor:
        rewards = torch.stack([out.rewards for out in outputs], dim=0)
        
        epistemic_uncertainty = rewards.var(dim=0)
        
        if hasattr(outputs[0], 'uncertainty') and outputs[0].uncertainty is not None:
            aleatoric_uncertainties = torch.stack([out.uncertainty for out in outputs], dim=0)
            aleatoric_uncertainty = aleatoric_uncertainties.mean(dim=0)
        else:
            aleatoric_uncertainty = torch.zeros_like(epistemic_uncertainty)
        
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        return total_uncertainty
    
    def _compute_ensemble_diversity(self, outputs: List[RewardOutput]) -> torch.Tensor:
        rewards = torch.stack([out.rewards for out in outputs], dim=0)
        
        pairwise_distances = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                distance = F.mse_loss(rewards[i], rewards[j], reduction='none')
                pairwise_distances.append(distance)
        
        if pairwise_distances:
            diversity = torch.stack(pairwise_distances, dim=0).mean(dim=0)
        else:
            diversity = torch.zeros_like(rewards[0])
        
        return diversity
    
    def compute_ensemble_loss(
        self,
        outputs: List[RewardOutput],
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        individual_losses = []
        for output in outputs:
            loss = F.mse_loss(output.rewards, targets)
            individual_losses.append(loss)
        
        ensemble_output = self._default_aggregation(outputs)
        ensemble_loss = F.mse_loss(ensemble_output, targets)
        
        diversity_loss = self._compute_diversity_loss(outputs)
        
        total_loss = ensemble_loss + self.diversity_regularization * diversity_loss
        
        return {
            "ensemble_loss": ensemble_loss,
            "diversity_loss": diversity_loss,
            "total_loss": total_loss,
            "individual_losses": individual_losses
        }
    
    def _compute_diversity_loss(self, outputs: List[RewardOutput]) -> torch.Tensor:
        rewards = torch.stack([out.rewards for out in outputs], dim=0)
        
        mean_reward = rewards.mean(dim=0)
        diversity_penalty = 0
        
        for i, reward in enumerate(rewards):
            diversity_penalty += F.mse_loss(reward, mean_reward)
        
        return -diversity_penalty / len(rewards)

class MetaLearner(nn.Module):
    def __init__(self, num_models: int, hidden_size: int):
        super().__init__()
        self.num_models = num_models
        
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        batch_size = hidden_states.size(1)
        
        attended_features, _ = self.attention(
            hidden_states.transpose(0, 1),
            hidden_states.transpose(0, 1),
            hidden_states.transpose(0, 1)
        )
        
        attended_features = attended_features.transpose(0, 1)
        
        combined_features = torch.cat([
            attended_features,
            rewards.unsqueeze(-1)
        ], dim=-1)
        
        weights = self.weight_predictor(combined_features).squeeze(-1)
        
        return weights

class SnapshotEnsembleScheduler:
    def __init__(
        self,
        num_snapshots: int,
        total_epochs: int,
        lr_min: float = 1e-6,
        lr_max: float = 1e-3
    ):
        self.num_snapshots = num_snapshots
        self.total_epochs = total_epochs
        self.lr_min = lr_min
        self.lr_max = lr_max
        
        self.epochs_per_cycle = total_epochs // num_snapshots
        self.snapshots = []
    
    def get_lr(self, epoch: int) -> float:
        cycle = epoch // self.epochs_per_cycle
        epoch_in_cycle = epoch % self.epochs_per_cycle
        
        lr = self.lr_min + (self.lr_max - self.lr_min) * (
            1 + math.cos(math.pi * epoch_in_cycle / self.epochs_per_cycle)
        ) / 2
        
        return lr
    
    def should_save_snapshot(self, epoch: int) -> bool:
        return (epoch + 1) % self.epochs_per_cycle == 0

class AdaptiveEnsemble(EnsembleRewardModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_tracker = PerformanceTracker(self.num_models)
        self.adaptive_weights = True
    
    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        
        if self.training and hasattr(output, 'objective_breakdown'):
            individual_rewards = output.objective_breakdown.get('individual_rewards', [])
            self.performance_tracker.update(individual_rewards)
        
        return output
    
    def update_ensemble_weights(self):
        if self.adaptive_weights:
            performance_scores = self.performance_tracker.get_performance_scores()
            self.aggregation_weights.data = F.softmax(
                torch.tensor(performance_scores),
                dim=0
            )

class PerformanceTracker:
    def __init__(self, num_models: int, window_size: int = 100):
        self.num_models = num_models
        self.window_size = window_size
        self.performance_history = [[] for _ in range(num_models)]
    
    def update(self, individual_rewards: List[torch.Tensor]):
        for i, reward in enumerate(individual_rewards):
            if i < self.num_models:
                reward_mean = reward.mean().item()
                self.performance_history[i].append(reward_mean)
                
                if len(self.performance_history[i]) > self.window_size:
                    self.performance_history[i].pop(0)
    
    def get_performance_scores(self) -> List[float]:
        scores = []
        for history in self.performance_history:
            if history:
                score = np.mean(history[-self.window_size:])
                scores.append(score)
            else:
                scores.append(0.0)
        return scores

class DiversityRegularizer:
    @staticmethod
    def compute_diversity_metrics(
        outputs: List[RewardOutput]
    ) -> Dict[str, torch.Tensor]:
        rewards = torch.stack([out.rewards for out in outputs], dim=0)
        
        pairwise_correlations = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                correlation = torch.corrcoef(
                    torch.stack([
                        rewards[i].flatten(),
                        rewards[j].flatten()
                    ])
                )[0, 1]
                if not torch.isnan(correlation):
                    pairwise_correlations.append(correlation)
        
        diversity_metrics = {
            "mean_correlation": torch.stack(pairwise_correlations).mean() if pairwise_correlations else torch.tensor(0.0),
            "reward_variance": rewards.var(dim=0).mean(),
            "prediction_entropy": -torch.sum(
                F.softmax(rewards, dim=0) * F.log_softmax(rewards, dim=0),
                dim=0
            ).mean()
        }
        
        return diversity_metrics
    
    @staticmethod
    def diversity_loss(
        outputs: List[RewardOutput],
        target_diversity: float = 0.3
    ) -> torch.Tensor:
        metrics = DiversityRegularizer.compute_diversity_metrics(outputs)
        
        correlation_penalty = F.relu(metrics["mean_correlation"] - target_diversity)
        variance_bonus = -F.relu(target_diversity - metrics["reward_variance"])
        
        return correlation_penalty + variance_bonus

class EnsembleCalibration:
    @staticmethod
    def temperature_scale_ensemble(
        ensemble_logits: torch.Tensor,
        temperature: torch.Tensor
    ) -> torch.Tensor:
        return ensemble_logits / temperature.unsqueeze(-1)
    
    @staticmethod
    def compute_ensemble_confidence(
        outputs: List[RewardOutput],
        method: str = "entropy"
    ) -> torch.Tensor:
        rewards = torch.stack([out.rewards for out in outputs], dim=0)
        
        if method == "entropy":
            probs = F.softmax(rewards, dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=0)
            confidence = 1 - entropy / math.log(len(outputs))
        elif method == "variance":
            variance = rewards.var(dim=0)
            confidence = 1 / (1 + variance)
        elif method == "max_agreement":
            max_prob = F.softmax(rewards, dim=0).max(dim=0)[0]
            confidence = max_prob
        else:
            raise ValueError(f"Unknown confidence method: {method}")
        
        return confidence