import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
from scipy.optimize import minimize

from reward_models.base_reward_model import BaseRewardModel, RewardOutput, PreferenceData

class BradleyTerryModel(nn.Module):
    def __init__(
        self,
        reward_model: BaseRewardModel,
        temperature: float = 1.0,
        preference_strength: float = 1.0,
        use_margin: bool = True,
        margin_schedule: str = "constant"
    ):
        super().__init__()
        self.reward_model = reward_model
        self.temperature = temperature
        self.preference_strength = preference_strength
        self.use_margin = use_margin
        self.margin_schedule = margin_schedule
        
        if use_margin:
            self.margin = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('margin', torch.tensor(0.0))
    
    def forward(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        chosen_rewards = self.reward_model(chosen_ids, chosen_mask, return_dict=False)
        rejected_rewards = self.reward_model(rejected_ids, rejected_mask, return_dict=False)
        
        preference_prob = self.compute_preference_probability(chosen_rewards, rejected_rewards)
        
        if not return_dict:
            return preference_prob
        
        return {
            "preference_probability": preference_prob,
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards,
            "reward_difference": chosen_rewards - rejected_rewards,
            "margin": self.margin
        }
    
    def compute_preference_probability(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor
    ) -> torch.Tensor:
        reward_diff = (chosen_rewards - rejected_rewards - self.margin) / self.temperature
        return torch.sigmoid(reward_diff * self.preference_strength)
    
    def compute_loss(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
        preference_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output = self.forward(chosen_ids, chosen_mask, rejected_ids, rejected_mask, return_dict=True)
        
        if preference_labels is None:
            preference_labels = torch.ones_like(output["preference_probability"])
        
        loss = F.binary_cross_entropy(
            output["preference_probability"],
            preference_labels,
            reduction='mean'
        )
        
        return loss
    
    def update_margin(self, epoch: int, total_epochs: int):
        if self.margin_schedule == "linear":
            self.margin.data = torch.tensor(0.5 * (1 - epoch / total_epochs))
        elif self.margin_schedule == "cosine":
            self.margin.data = torch.tensor(0.5 * (1 + math.cos(math.pi * epoch / total_epochs)) / 2)
        elif self.margin_schedule == "exponential":
            self.margin.data = torch.tensor(0.5 * math.exp(-epoch / total_epochs))

class MultiObjectiveBradleyTerry(BradleyTerryModel):
    def __init__(
        self,
        reward_model: BaseRewardModel,
        num_objectives: int,
        objective_weights: Optional[torch.Tensor] = None,
        **kwargs
    ):
        super().__init__(reward_model, **kwargs)
        self.num_objectives = num_objectives
        
        if objective_weights is None:
            objective_weights = torch.ones(num_objectives) / num_objectives
        self.register_buffer('objective_weights', objective_weights)
        
        self.objective_margins = nn.Parameter(torch.ones(num_objectives) * 0.5)
    
    def compute_preference_probability(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor
    ) -> torch.Tensor:
        objective_diffs = (chosen_rewards - rejected_rewards - self.objective_margins) / self.temperature
        weighted_diff = torch.sum(objective_diffs * self.objective_weights, dim=-1)
        return torch.sigmoid(weighted_diff * self.preference_strength)

class HierarchicalBradleyTerry(BradleyTerryModel):
    def __init__(
        self,
        reward_model: BaseRewardModel,
        hierarchy_levels: List[str] = ["local", "global"],
        level_weights: Optional[torch.Tensor] = None,
        **kwargs
    ):
        super().__init__(reward_model, **kwargs)
        self.hierarchy_levels = hierarchy_levels
        self.num_levels = len(hierarchy_levels)
        
        if level_weights is None:
            level_weights = torch.ones(self.num_levels) / self.num_levels
        self.register_buffer('level_weights', level_weights)
        
        self.level_projections = nn.ModuleDict({
            level: nn.Linear(reward_model.hidden_size, 1)
            for level in hierarchy_levels
        })
    
    def forward(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        chosen_output = self.reward_model(chosen_ids, chosen_mask, return_dict=True)
        rejected_output = self.reward_model(rejected_ids, rejected_mask, return_dict=True)
        
        level_preferences = {}
        total_preference = 0
        
        for i, level in enumerate(self.hierarchy_levels):
            if level == "local":
                chosen_level_reward = chosen_output.rewards
                rejected_level_reward = rejected_output.rewards
            elif level == "global":
                chosen_level_reward = self.level_projections[level](chosen_output.hidden_states)
                rejected_level_reward = self.level_projections[level](rejected_output.hidden_states)
            
            level_prob = self.compute_preference_probability(chosen_level_reward, rejected_level_reward)
            level_preferences[level] = level_prob
            total_preference += self.level_weights[i] * level_prob
        
        if not return_dict:
            return total_preference
        
        return {
            "preference_probability": total_preference,
            "level_preferences": level_preferences,
            "chosen_rewards": chosen_output.rewards,
            "rejected_rewards": rejected_output.rewards
        }

class DynamicBradleyTerry(BradleyTerryModel):
    def __init__(
        self,
        reward_model: BaseRewardModel,
        confidence_threshold: float = 0.8,
        adaptation_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(reward_model, **kwargs)
        self.confidence_threshold = confidence_threshold
        self.adaptation_rate = adaptation_rate
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(reward_model.hidden_size, reward_model.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(reward_model.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.dynamic_temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        chosen_output = self.reward_model(chosen_ids, chosen_mask, return_dict=True)
        rejected_output = self.reward_model(rejected_ids, rejected_mask, return_dict=True)
        
        chosen_confidence = self.confidence_estimator(chosen_output.hidden_states)
        rejected_confidence = self.confidence_estimator(rejected_output.hidden_states)
        
        avg_confidence = (chosen_confidence + rejected_confidence) / 2
        
        if avg_confidence.mean() < self.confidence_threshold:
            effective_temperature = self.dynamic_temperature * (1 + self.adaptation_rate)
        else:
            effective_temperature = self.dynamic_temperature
        
        reward_diff = (chosen_output.rewards - rejected_output.rewards - self.margin) / effective_temperature
        preference_prob = torch.sigmoid(reward_diff * self.preference_strength)
        
        if not return_dict:
            return preference_prob
        
        return {
            "preference_probability": preference_prob,
            "chosen_rewards": chosen_output.rewards,
            "rejected_rewards": rejected_output.rewards,
            "chosen_confidence": chosen_confidence,
            "rejected_confidence": rejected_confidence,
            "effective_temperature": effective_temperature
        }

class BayesianBradleyTerry(BradleyTerryModel):
    def __init__(
        self,
        reward_model: BaseRewardModel,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        num_samples: int = 10,
        **kwargs
    ):
        super().__init__(reward_model, **kwargs)
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.num_samples = num_samples
        
        self.posterior_mean = nn.Parameter(torch.tensor(prior_mean))
        self.posterior_logvar = nn.Parameter(torch.tensor(math.log(prior_variance)))
    
    def sample_parameters(self) -> torch.Tensor:
        std = torch.exp(0.5 * self.posterior_logvar)
        eps = torch.randn(self.num_samples, device=self.posterior_mean.device)
        return self.posterior_mean + eps * std
    
    def forward(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        chosen_rewards = self.reward_model(chosen_ids, chosen_mask, return_dict=False)
        rejected_rewards = self.reward_model(rejected_ids, rejected_mask, return_dict=False)
        
        sampled_params = self.sample_parameters()
        
        preference_probs = []
        for param in sampled_params:
            reward_diff = (chosen_rewards - rejected_rewards - self.margin) / self.temperature
            prob = torch.sigmoid(reward_diff * param)
            preference_probs.append(prob)
        
        preference_probs = torch.stack(preference_probs, dim=0)
        mean_prob = preference_probs.mean(dim=0)
        uncertainty = preference_probs.std(dim=0)
        
        if not return_dict:
            return mean_prob
        
        return {
            "preference_probability": mean_prob,
            "uncertainty": uncertainty,
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards,
            "posterior_samples": sampled_params
        }
    
    def compute_kl_loss(self) -> torch.Tensor:
        posterior_var = torch.exp(self.posterior_logvar)
        
        kl_div = 0.5 * (
            (self.posterior_mean - self.prior_mean)**2 / self.prior_variance +
            posterior_var / self.prior_variance -
            1 - 
            self.posterior_logvar + 
            math.log(self.prior_variance)
        )
        
        return kl_div

class ThurstoneCase5(nn.Module):
    def __init__(
        self,
        reward_model: BaseRewardModel,
        noise_variance: float = 1.0
    ):
        super().__init__()
        self.reward_model = reward_model
        self.noise_variance = noise_variance
    
    def forward(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        chosen_rewards = self.reward_model(chosen_ids, chosen_mask, return_dict=False)
        rejected_rewards = self.reward_model(rejected_ids, rejected_mask, return_dict=False)
        
        diff = chosen_rewards - rejected_rewards
        preference_prob = torch.distributions.Normal(
            diff, 
            math.sqrt(2 * self.noise_variance)
        ).cdf(torch.zeros_like(diff))
        
        if not return_dict:
            return preference_prob
        
        return {
            "preference_probability": preference_prob,
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards,
            "reward_difference": diff
        }

class BradleyTerryTrainer:
    def __init__(
        self,
        model: BradleyTerryModel,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.warmup_steps = warmup_steps
        self.step_count = 0
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.model.compute_loss(
            batch["chosen_ids"],
            batch["chosen_mask"],
            batch["rejected_ids"],
            batch["rejected_mask"],
            batch.get("preference_labels")
        )
        
        if isinstance(self.model, BayesianBradleyTerry):
            kl_loss = self.model.compute_kl_loss()
            total_loss = loss + 0.01 * kl_loss
        else:
            total_loss = loss
            kl_loss = torch.tensor(0.0)
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self._update_learning_rate()
        self.optimizer.step()
        
        self.step_count += 1
        
        with torch.no_grad():
            output = self.model(
                batch["chosen_ids"],
                batch["chosen_mask"],
                batch["rejected_ids"],
                batch["rejected_mask"],
                return_dict=True
            )
            
            accuracy = (output["preference_probability"] > 0.5).float().mean()
        
        return {
            "loss": loss.item(),
            "total_loss": total_loss.item(),
            "kl_loss": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            "accuracy": accuracy.item(),
            "margin": self.model.margin.item() if hasattr(self.model, 'margin') else 0.0
        }
    
    def _update_learning_rate(self):
        if self.step_count < self.warmup_steps:
            lr_scale = self.step_count / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_scale

class PreferenceDatasetGenerator:
    @staticmethod
    def generate_synthetic_preferences(
        reward_model: BaseRewardModel,
        texts: List[str],
        tokenizer,
        noise_scale: float = 0.1,
        num_pairs: int = 1000
    ) -> List[PreferenceData]:
        preferences = []
        
        for _ in range(num_pairs):
            idx1, idx2 = np.random.choice(len(texts), 2, replace=False)
            
            inputs1 = tokenizer(texts[idx1], return_tensors="pt", truncation=True, padding=True)
            inputs2 = tokenizer(texts[idx2], return_tensors="pt", truncation=True, padding=True)
            
            with torch.no_grad():
                reward1 = reward_model(inputs1["input_ids"], inputs1["attention_mask"], return_dict=False)
                reward2 = reward_model(inputs2["input_ids"], inputs2["attention_mask"], return_dict=False)
            
            noise1 = torch.randn_like(reward1) * noise_scale
            noise2 = torch.randn_like(reward2) * noise_scale
            
            noisy_reward1 = reward1 + noise1
            noisy_reward2 = reward2 + noise2
            
            if noisy_reward1 > noisy_reward2:
                chosen = inputs1
                rejected = inputs2
            else:
                chosen = inputs2
                rejected = inputs1
            
            preferences.append(PreferenceData(
                chosen=chosen["input_ids"],
                rejected=rejected["input_ids"],
                chosen_attention_mask=chosen["attention_mask"],
                rejected_attention_mask=rejected["attention_mask"]
            ))
        
        return preferences
    
    @staticmethod
    def create_preference_rankings(
        items: List[str],
        true_scores: torch.Tensor,
        num_comparisons: int = 100
    ) -> List[Tuple[int, int, float]]:
        rankings = []
        
        for _ in range(num_comparisons):
            idx1, idx2 = np.random.choice(len(items), 2, replace=False)
            
            score_diff = true_scores[idx1] - true_scores[idx2]
            preference_strength = torch.sigmoid(score_diff).item()
            
            rankings.append((idx1, idx2, preference_strength))
        
        return rankings

class BradleyTerryAnalyzer:
    @staticmethod
    def compute_win_probability_matrix(
        reward_model: BaseRewardModel,
        texts: List[str],
        tokenizer
    ) -> torch.Tensor:
        n = len(texts)
        win_matrix = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    inputs_i = tokenizer(texts[i], return_tensors="pt", truncation=True, padding=True)
                    inputs_j = tokenizer(texts[j], return_tensors="pt", truncation=True, padding=True)
                    
                    with torch.no_grad():
                        reward_i = reward_model(inputs_i["input_ids"], inputs_i["attention_mask"], return_dict=False)
                        reward_j = reward_model(inputs_j["input_ids"], inputs_j["attention_mask"], return_dict=False)
                    
                    win_prob = torch.sigmoid(reward_i - reward_j)
                    win_matrix[i, j] = win_prob.item()
        
        return win_matrix
    
    @staticmethod
    def compute_ranking_from_preferences(
        win_matrix: torch.Tensor,
        method: str = "iterative"
    ) -> torch.Tensor:
        n = win_matrix.size(0)
        
        if method == "iterative":
            scores = torch.ones(n)
            
            for _ in range(100):
                new_scores = torch.zeros(n)
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            prob = win_matrix[i, j]
                            new_scores[i] += prob / (scores[i] / (scores[i] + scores[j]) + 1e-8)
                
                new_scores = new_scores / new_scores.sum()
                
                if torch.allclose(scores, new_scores, atol=1e-6):
                    break
                
                scores = new_scores
            
            return scores
        
        elif method == "eigenvalue":
            A = win_matrix + win_matrix.t() + torch.eye(n) * 1e-6
            eigenvals, eigenvecs = torch.linalg.eig(A)
            
            max_idx = torch.argmax(eigenvals.real)
            ranking = eigenvecs[:, max_idx].real
            
            return F.softmax(ranking, dim=0)
        
        else:
            raise ValueError(f"Unknown ranking method: {method}")