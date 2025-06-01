import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from reward_models.base_reward_model import BaseRewardModel, RewardOutput

class StepwiseRewardAssigner:
    def __init__(
        self,
        assignment_strategy: str = "progressive",
        step_weight_decay: float = 0.9,
        minimum_step_reward: float = 0.1,
        correctness_bonus: float = 0.2
    ):
        self.assignment_strategy = assignment_strategy
        self.step_weight_decay = step_weight_decay
        self.minimum_step_reward = minimum_step_reward
        self.correctness_bonus = correctness_bonus
    
    def assign_step_rewards(
        self,
        final_reward: torch.Tensor,
        step_correctness: torch.Tensor,
        step_importance: torch.Tensor,
        step_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.assignment_strategy == "progressive":
            return self._progressive_assignment(final_reward, step_correctness, step_importance, step_mask)
        elif self.assignment_strategy == "uniform":
            return self._uniform_assignment(final_reward, step_correctness, step_mask)
        elif self.assignment_strategy == "importance_weighted":
            return self._importance_weighted_assignment(final_reward, step_correctness, step_importance, step_mask)
        elif self.assignment_strategy == "exponential_decay":
            return self._exponential_decay_assignment(final_reward, step_correctness, step_mask)
        elif self.assignment_strategy == "critical_path":
            return self._critical_path_assignment(final_reward, step_correctness, step_importance, step_mask)
        else:
            raise ValueError(f"Unknown assignment strategy: {self.assignment_strategy}")
    
    def _progressive_assignment(
        self,
        final_reward: torch.Tensor,
        step_correctness: torch.Tensor,
        step_importance: torch.Tensor,
        step_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps = step_correctness.shape
        step_rewards = torch.zeros_like(step_correctness)
        
        for b in range(batch_size):
            valid_steps = step_mask[b].sum().item()
            if valid_steps == 0:
                continue
            
            base_reward = final_reward[b] / valid_steps
            
            for s in range(int(valid_steps)):
                # Progressive weighting: later steps get higher weight
                progress_weight = (s + 1) / valid_steps
                correctness_factor = step_correctness[b, s]
                importance_factor = step_importance[b, s]
                
                step_reward = base_reward * progress_weight * correctness_factor * importance_factor
                step_reward = max(step_reward, self.minimum_step_reward)
                
                if correctness_factor > 0.8:  # High correctness bonus
                    step_reward += self.correctness_bonus
                
                step_rewards[b, s] = step_reward
        
        return step_rewards
    
    def _uniform_assignment(
        self,
        final_reward: torch.Tensor,
        step_correctness: torch.Tensor,
        step_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps = step_correctness.shape
        step_rewards = torch.zeros_like(step_correctness)
        
        for b in range(batch_size):
            valid_steps = step_mask[b].sum().item()
            if valid_steps > 0:
                uniform_reward = final_reward[b] / valid_steps
                step_rewards[b, :int(valid_steps)] = uniform_reward * step_correctness[b, :int(valid_steps)]
        
        return step_rewards
    
    def _importance_weighted_assignment(
        self,
        final_reward: torch.Tensor,
        step_correctness: torch.Tensor,
        step_importance: torch.Tensor,
        step_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps = step_correctness.shape
        step_rewards = torch.zeros_like(step_correctness)
        
        for b in range(batch_size):
            valid_mask = step_mask[b].bool()
            if valid_mask.sum() == 0:
                continue
            
            valid_importance = step_importance[b][valid_mask]
            valid_correctness = step_correctness[b][valid_mask]
            
            # Normalize importance weights
            importance_weights = valid_importance / (valid_importance.sum() + 1e-8)
            
            # Assign rewards based on importance and correctness
            step_reward_values = final_reward[b] * importance_weights * valid_correctness
            step_rewards[b][valid_mask] = step_reward_values
        
        return step_rewards
    
    def _exponential_decay_assignment(
        self,
        final_reward: torch.Tensor,
        step_correctness: torch.Tensor,
        step_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps = step_correctness.shape
        step_rewards = torch.zeros_like(step_correctness)
        
        for b in range(batch_size):
            valid_steps = step_mask[b].sum().item()
            if valid_steps == 0:
                continue
            
            # Exponential decay weights (recent steps get higher rewards)
            decay_weights = torch.tensor([
                self.step_weight_decay ** (valid_steps - s - 1)
                for s in range(int(valid_steps))
            ], device=step_correctness.device)
            
            # Normalize weights
            decay_weights = decay_weights / decay_weights.sum()
            
            # Assign rewards
            for s in range(int(valid_steps)):
                step_rewards[b, s] = (
                    final_reward[b] * decay_weights[s] * step_correctness[b, s]
                )
        
        return step_rewards
    
    def _critical_path_assignment(
        self,
        final_reward: torch.Tensor,
        step_correctness: torch.Tensor,
        step_importance: torch.Tensor,
        step_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps = step_correctness.shape
        step_rewards = torch.zeros_like(step_correctness)
        
        for b in range(batch_size):
            valid_mask = step_mask[b].bool()
            if valid_mask.sum() == 0:
                continue
            
            # Identify critical path (steps with high importance and correctness)
            critical_scores = step_importance[b] * step_correctness[b]
            critical_threshold = critical_scores.quantile(0.7)
            
            critical_steps = (critical_scores >= critical_threshold) & valid_mask
            non_critical_steps = (critical_scores < critical_threshold) & valid_mask
            
            # Assign higher rewards to critical steps
            critical_reward_fraction = 0.8
            non_critical_reward_fraction = 0.2
            
            num_critical = critical_steps.sum().item()
            num_non_critical = non_critical_steps.sum().item()
            
            if num_critical > 0:
                critical_reward = (final_reward[b] * critical_reward_fraction) / num_critical
                step_rewards[b][critical_steps] = critical_reward
            
            if num_non_critical > 0:
                non_critical_reward = (final_reward[b] * non_critical_reward_fraction) / num_non_critical
                step_rewards[b][non_critical_steps] = non_critical_reward
        
        return step_rewards

class AdaptiveStepRewardModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_objectives: int = 1,
        adaptation_strategy: str = "learned"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_objectives = num_objectives
        self.adaptation_strategy = adaptation_strategy
        
        if adaptation_strategy == "learned":
            self.step_weight_predictor = LearnedStepWeighting(hidden_size)
        elif adaptation_strategy == "attention":
            self.step_attention = StepAttentionWeighting(hidden_size)
        elif adaptation_strategy == "meta":
            self.meta_learner = MetaStepLearner(hidden_size)
        
        self.step_value_estimator = StepValueEstimator(hidden_size, num_objectives)
    
    def forward(
        self,
        step_features: torch.Tensor,
        step_context: torch.Tensor,
        final_reward: torch.Tensor,
        step_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps, _ = step_features.shape
        
        if self.adaptation_strategy == "learned":
            step_weights = self.step_weight_predictor(step_features, step_context, final_reward)
        elif self.adaptation_strategy == "attention":
            step_weights = self.step_attention(step_features, step_context)
        elif self.adaptation_strategy == "meta":
            step_weights = self.meta_learner(step_features, step_context, final_reward)
        else:
            step_weights = torch.ones(batch_size, num_steps, device=step_features.device)
        
        # Estimate individual step values
        step_values = self.step_value_estimator(step_features)
        
        # Combine weights and values
        step_rewards = step_weights.unsqueeze(-1) * step_values
        
        # Apply mask
        step_rewards = step_rewards * step_mask.unsqueeze(-1)
        
        # Normalize to match final reward
        total_predicted = step_rewards.sum(dim=1)
        normalization_factor = final_reward.unsqueeze(-1) / (total_predicted + 1e-8)
        step_rewards = step_rewards * normalization_factor.unsqueeze(1)
        
        return step_rewards

class LearnedStepWeighting(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, hidden_size),  # step + context + final_reward
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        step_features: torch.Tensor,
        step_context: torch.Tensor,
        final_reward: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps, hidden_size = step_features.shape
        
        # Expand final reward to match step dimensions
        final_reward_expanded = final_reward.unsqueeze(1).unsqueeze(2).expand(batch_size, num_steps, 1)
        
        # Combine features
        combined_features = torch.cat([
            step_features,
            step_context,
            final_reward_expanded
        ], dim=-1)
        
        # Predict weights
        encoded = self.context_encoder(combined_features)
        weights = self.weight_predictor(encoded).squeeze(-1)
        
        # Normalize weights to sum to 1 across steps
        weights = F.softmax(weights, dim=1)
        
        return weights

class StepAttentionWeighting(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        
        self.weight_projection = nn.Linear(hidden_size, 1)
    
    def forward(
        self,
        step_features: torch.Tensor,
        step_context: torch.Tensor
    ) -> torch.Tensor:
        # Use context as query, steps as key/value
        attended_features, attention_weights = self.attention(
            step_context, step_features, step_features
        )
        
        # Project to weights
        weights = self.weight_projection(attended_features).squeeze(-1)
        weights = F.softmax(weights, dim=1)
        
        return weights

class MetaStepLearner(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.meta_network = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),  # context + final_reward
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.step_weight_generator = nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self,
        step_features: torch.Tensor,
        step_context: torch.Tensor,
        final_reward: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps, hidden_size = step_features.shape
        
        # Generate meta-parameters
        context_mean = step_context.mean(dim=1)
        meta_input = torch.cat([context_mean, final_reward.unsqueeze(-1)], dim=-1)
        meta_params = self.meta_network(meta_input)
        
        # Generate step-specific weights
        weight_params = self.step_weight_generator(meta_params)
        
        # Compute weights as dot product with step features
        weights = torch.sum(
            step_features * weight_params.unsqueeze(1),
            dim=-1
        )
        
        weights = F.softmax(weights, dim=1)
        
        return weights

class StepValueEstimator(nn.Module):
    def __init__(self, hidden_size: int, num_objectives: int = 1):
        super().__init__()
        
        self.value_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_objectives)
        )
    
    def forward(self, step_features: torch.Tensor) -> torch.Tensor:
        return self.value_network(step_features)

class ReasoningStepRewardModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        step_types: List[str] = ["premise", "inference", "conclusion"],
        type_specific_weights: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.step_types = step_types
        self.type_specific_weights = type_specific_weights
        
        self.step_type_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, len(step_types)),
            nn.Softmax(dim=-1)
        )
        
        if type_specific_weights:
            self.type_specific_networks = nn.ModuleDict({
                step_type: nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, 1)
                ) for step_type in step_types
            })
        
        self.reasoning_quality_estimator = ReasoningQualityEstimator(hidden_size)
    
    def forward(
        self,
        step_features: torch.Tensor,
        reasoning_chain: torch.Tensor,
        step_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps, hidden_size = step_features.shape
        
        # Classify step types
        step_type_probs = self.step_type_classifier(step_features)
        
        # Compute type-specific rewards
        if self.type_specific_weights:
            type_rewards = torch.zeros(batch_size, num_steps, 1, device=step_features.device)
            
            for i, step_type in enumerate(self.step_types):
                type_mask = step_type_probs[:, :, i].unsqueeze(-1)
                type_reward = self.type_specific_networks[step_type](step_features)
                type_rewards += type_mask * type_reward
        else:
            type_rewards = torch.ones(batch_size, num_steps, 1, device=step_features.device)
        
        # Estimate reasoning quality
        quality_scores = self.reasoning_quality_estimator(step_features, reasoning_chain)
        
        # Combine type and quality
        step_rewards = type_rewards.squeeze(-1) * quality_scores
        
        # Apply mask
        step_rewards = step_rewards * step_mask
        
        return step_rewards

class ReasoningQualityEstimator(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.coherence_estimator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        self.novelty_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.relevance_attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, batch_first=True
        )
    
    def forward(
        self,
        step_features: torch.Tensor,
        reasoning_chain: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps, hidden_size = step_features.shape
        
        # Coherence: how well each step fits with the reasoning chain
        chain_context = reasoning_chain.mean(dim=1, keepdim=True).expand(-1, num_steps, -1)
        coherence_input = torch.cat([step_features, chain_context], dim=-1)
        coherence_scores = self.coherence_estimator(coherence_input).squeeze(-1)
        
        # Novelty: how novel/creative each step is
        novelty_scores = self.novelty_estimator(step_features).squeeze(-1)
        
        # Relevance: attention-based relevance to the overall reasoning
        relevance_features, _ = self.relevance_attention(
            step_features, reasoning_chain, reasoning_chain
        )
        relevance_scores = torch.norm(relevance_features, dim=-1)
        relevance_scores = F.sigmoid(relevance_scores)
        
        # Combine quality metrics
        quality_scores = (coherence_scores + novelty_scores + relevance_scores) / 3
        
        return quality_scores

class DynamicStepRewardAdjuster:
    def __init__(
        self,
        adjustment_strategies: List[str] = ["difficulty", "progress", "error_correction"],
        base_adjustment_rate: float = 0.1
    ):
        self.adjustment_strategies = adjustment_strategies
        self.base_adjustment_rate = base_adjustment_rate
    
    def adjust_step_rewards(
        self,
        step_rewards: torch.Tensor,
        step_metadata: Dict[str, torch.Tensor],
        adjustment_context: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        adjusted_rewards = step_rewards.clone()
        
        for strategy in self.adjustment_strategies:
            if strategy == "difficulty":
                adjusted_rewards = self._difficulty_adjustment(
                    adjusted_rewards,
                    step_metadata.get("difficulty_scores"),
                    adjustment_context
                )
            elif strategy == "progress":
                adjusted_rewards = self._progress_adjustment(
                    adjusted_rewards,
                    step_metadata.get("progress_indicators"),
                    adjustment_context
                )
            elif strategy == "error_correction":
                adjusted_rewards = self._error_correction_adjustment(
                    adjusted_rewards,
                    step_metadata.get("error_indicators"),
                    adjustment_context
                )
        
        return adjusted_rewards
    
    def _difficulty_adjustment(
        self,
        step_rewards: torch.Tensor,
        difficulty_scores: Optional[torch.Tensor],
        context: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if difficulty_scores is None:
            return step_rewards
        
        # Higher rewards for more difficult steps
        difficulty_bonus = difficulty_scores * self.base_adjustment_rate
        return step_rewards + difficulty_bonus
    
    def _progress_adjustment(
        self,
        step_rewards: torch.Tensor,
        progress_indicators: Optional[torch.Tensor],
        context: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if progress_indicators is None:
            return step_rewards
        
        # Bonus for steps that make significant progress
        progress_bonus = progress_indicators * self.base_adjustment_rate * 2
        return step_rewards + progress_bonus
    
    def _error_correction_adjustment(
        self,
        step_rewards: torch.Tensor,
        error_indicators: Optional[torch.Tensor],
        context: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if error_indicators is None:
            return step_rewards
        
        # Penalty for steps that introduce errors
        error_penalty = error_indicators * self.base_adjustment_rate * 1.5
        return step_rewards - error_penalty

class StepRewardValidator:
    def __init__(self, validation_thresholds: Dict[str, float]):
        self.validation_thresholds = validation_thresholds
    
    def validate_step_rewards(
        self,
        step_rewards: torch.Tensor,
        step_mask: torch.Tensor,
        final_reward: torch.Tensor
    ) -> Dict[str, bool]:
        validation_results = {}
        
        # Check reward conservation
        predicted_total = step_rewards.sum(dim=1)
        conservation_error = torch.abs(predicted_total - final_reward) / (torch.abs(final_reward) + 1e-8)
        validation_results["conservation"] = (conservation_error < self.validation_thresholds.get("conservation", 0.1)).all().item()
        
        # Check reward monotonicity (if expected)
        if "monotonicity" in self.validation_thresholds:
            reward_diffs = step_rewards[:, 1:] - step_rewards[:, :-1]
            monotonic = (reward_diffs >= -self.validation_thresholds["monotonicity"]).all()
            validation_results["monotonicity"] = monotonic.item()
        
        # Check reward range
        min_reward = step_rewards[step_mask.bool()].min()
        max_reward = step_rewards[step_mask.bool()].max()
        validation_results["range"] = (
            min_reward >= self.validation_thresholds.get("min_reward", -10.0) and
            max_reward <= self.validation_thresholds.get("max_reward", 10.0)
        )
        
        return validation_results