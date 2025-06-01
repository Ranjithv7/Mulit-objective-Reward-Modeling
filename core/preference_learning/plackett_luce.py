import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np

from reward_models.base_reward_model import BaseRewardModel, RewardOutput

class PlackettLuceModel(nn.Module):
    def __init__(
        self,
        reward_model: BaseRewardModel,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_gumbel: bool = False
    ):
        super().__init__()
        self.reward_model = reward_model
        self.temperature = temperature
        self.top_k = top_k
        self.use_gumbel = use_gumbel
    
    def forward(
        self,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
        ranking: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = input_ids_list[0].size(0)
        num_items = len(input_ids_list)
        
        rewards = []
        for i in range(num_items):
            reward = self.reward_model(input_ids_list[i], attention_mask_list[i], return_dict=False)
            rewards.append(reward)
        
        rewards = torch.stack(rewards, dim=1)
        
        if self.use_gumbel:
            ranking_probs = self._gumbel_plackett_luce(rewards)
        else:
            ranking_probs = self._standard_plackett_luce(rewards)
        
        if not return_dict:
            return ranking_probs
        
        predicted_ranking = torch.argsort(rewards, dim=1, descending=True)
        
        return {
            "ranking_probabilities": ranking_probs,
            "rewards": rewards,
            "predicted_ranking": predicted_ranking,
            "top_item_prob": ranking_probs[:, 0] if ranking_probs.size(1) > 0 else None
        }
    
    def _standard_plackett_luce(self, rewards: torch.Tensor) -> torch.Tensor:
        scaled_rewards = rewards / self.temperature
        
        if self.top_k is not None:
            top_k = min(self.top_k, rewards.size(1))
            top_rewards, top_indices = torch.topk(scaled_rewards, top_k, dim=1)
            
            batch_size = rewards.size(0)
            ranking_probs = torch.zeros(batch_size, top_k, device=rewards.device)
            
            remaining_rewards = top_rewards.clone()
            
            for pos in range(top_k):
                probs = F.softmax(remaining_rewards, dim=1)
                ranking_probs[:, pos] = probs[:, 0]
                
                if pos < top_k - 1:
                    remaining_rewards = remaining_rewards[:, 1:]
        else:
            batch_size, num_items = rewards.shape
            ranking_probs = torch.zeros(batch_size, num_items, device=rewards.device)
            
            sorted_rewards, sorted_indices = torch.sort(scaled_rewards, dim=1, descending=True)
            remaining_rewards = sorted_rewards.clone()
            
            for pos in range(num_items):
                if remaining_rewards.size(1) > 0:
                    probs = F.softmax(remaining_rewards, dim=1)
                    ranking_probs[:, pos] = probs[:, 0]
                    
                    if pos < num_items - 1:
                        remaining_rewards = remaining_rewards[:, 1:]
        
        return ranking_probs
    
    def _gumbel_plackett_luce(self, rewards: torch.Tensor) -> torch.Tensor:
        gumbel_noise = torch.distributions.Gumbel(0, 1).sample(rewards.shape).to(rewards.device)
        gumbel_rewards = rewards + gumbel_noise * self.temperature
        
        return F.softmax(gumbel_rewards, dim=1)
    
    def compute_ranking_loss(
        self,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
        true_ranking: torch.Tensor
    ) -> torch.Tensor:
        output = self.forward(input_ids_list, attention_mask_list, return_dict=True)
        rewards = output["rewards"]
        
        batch_size, num_items = rewards.shape
        loss = 0.0
        
        for batch_idx in range(batch_size):
            batch_rewards = rewards[batch_idx]
            batch_ranking = true_ranking[batch_idx]
            
            remaining_items = list(range(num_items))
            
            for pos in range(num_items):
                if len(remaining_items) > 1:
                    current_item = batch_ranking[pos].item()
                    
                    if current_item in remaining_items:
                        remaining_rewards = batch_rewards[remaining_items]
                        current_item_idx = remaining_items.index(current_item)
                        
                        log_prob = F.log_softmax(remaining_rewards / self.temperature, dim=0)
                        loss -= log_prob[current_item_idx]
                        
                        remaining_items.remove(current_item)
        
        return loss / batch_size

class HierarchicalPlackettLuce(PlackettLuceModel):
    def __init__(
        self,
        reward_model: BaseRewardModel,
        num_levels: int = 2,
        level_sizes: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__(reward_model, **kwargs)
        self.num_levels = num_levels
        self.level_sizes = level_sizes or [4, 2]
        
        self.level_predictors = nn.ModuleList([
            nn.Linear(reward_model.hidden_size, 1)
            for _ in range(num_levels)
        ])
    
    def forward(
        self,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
        ranking: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        num_items = len(input_ids_list)
        
        hidden_states = []
        base_rewards = []
        
        for i in range(num_items):
            output = self.reward_model(input_ids_list[i], attention_mask_list[i], return_dict=True)
            hidden_states.append(output.hidden_states)
            base_rewards.append(output.rewards)
        
        hidden_states = torch.stack(hidden_states, dim=1)
        base_rewards = torch.stack(base_rewards, dim=1)
        
        level_rankings = []
        current_items = list(range(num_items))
        
        for level in range(self.num_levels):
            if len(current_items) <= 1:
                break
            
            level_size = min(self.level_sizes[level], len(current_items))
            
            level_hidden = hidden_states[:, current_items, :]
            level_rewards = self.level_predictors[level](level_hidden).squeeze(-1)
            
            level_probs = self._standard_plackett_luce(level_rewards)
            level_rankings.append(level_probs)
            
            if level < self.num_levels - 1:
                top_indices = torch.topk(level_rewards, level_size, dim=1)[1]
                current_items = [current_items[idx] for idx in top_indices[0].tolist()]
        
        final_ranking = level_rankings[-1] if level_rankings else torch.zeros(1, 1)
        
        if not return_dict:
            return final_ranking
        
        return {
            "hierarchical_rankings": level_rankings,
            "final_ranking": final_ranking,
            "base_rewards": base_rewards,
            "hidden_states": hidden_states
        }

class MixtureOfPlackettLuce(PlackettLuceModel):
    def __init__(
        self,
        reward_model: BaseRewardModel,
        num_components: int = 3,
        **kwargs
    ):
        super().__init__(reward_model, **kwargs)
        self.num_components = num_components
        
        self.component_weights = nn.Parameter(torch.ones(num_components) / num_components)
        self.component_predictors = nn.ModuleList([
            nn.Linear(reward_model.hidden_size, 1)
            for _ in range(num_components)
        ])
        
        self.gate_network = nn.Sequential(
            nn.Linear(reward_model.hidden_size, reward_model.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(reward_model.hidden_size // 2, num_components),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
        ranking: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        num_items = len(input_ids_list)
        batch_size = input_ids_list[0].size(0)
        
        hidden_states = []
        for i in range(num_items):
            output = self.reward_model(input_ids_list[i], attention_mask_list[i], return_dict=True)
            hidden_states.append(output.hidden_states)
        
        hidden_states = torch.stack(hidden_states, dim=1)
        context = hidden_states.mean(dim=1)
        
        gate_weights = self.gate_network(context)
        
        component_rankings = []
        for comp in range(self.num_components):
            comp_rewards = self.component_predictors[comp](hidden_states).squeeze(-1)
            comp_probs = self._standard_plackett_luce(comp_rewards)
            component_rankings.append(comp_probs)
        
        component_rankings = torch.stack(component_rankings, dim=2)
        
        mixture_ranking = torch.sum(
            component_rankings * gate_weights.unsqueeze(1).unsqueeze(1),
            dim=2
        )
        
        if not return_dict:
            return mixture_ranking
        
        return {
            "mixture_ranking": mixture_ranking,
            "component_rankings": component_rankings,
            "gate_weights": gate_weights,
            "hidden_states": hidden_states
        }

class BayesianPlackettLuce(PlackettLuceModel):
    def __init__(
        self,
        reward_model: BaseRewardModel,
        prior_precision: float = 1.0,
        num_samples: int = 10,
        **kwargs
    ):
        super().__init__(reward_model, **kwargs)
        self.prior_precision = prior_precision
        self.num_samples = num_samples
        
        self.posterior_mean = nn.Parameter(torch.zeros(reward_model.hidden_size))
        self.posterior_precision = nn.Parameter(torch.ones(reward_model.hidden_size) * prior_precision)
    
    def sample_parameters(self) -> torch.Tensor:
        std = 1.0 / torch.sqrt(self.posterior_precision)
        samples = []
        
        for _ in range(self.num_samples):
            eps = torch.randn_like(self.posterior_mean)
            sample = self.posterior_mean + eps * std
            samples.append(sample)
        
        return torch.stack(samples, dim=0)
    
    def forward(
        self,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
        ranking: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        num_items = len(input_ids_list)
        
        hidden_states = []
        for i in range(num_items):
            output = self.reward_model(input_ids_list[i], attention_mask_list[i], return_dict=True)
            hidden_states.append(output.hidden_states)
        
        hidden_states = torch.stack(hidden_states, dim=1)
        
        parameter_samples = self.sample_parameters()
        
        sample_rankings = []
        for sample in parameter_samples:
            sample_rewards = torch.sum(hidden_states * sample.unsqueeze(0).unsqueeze(0), dim=-1)
            sample_probs = self._standard_plackett_luce(sample_rewards)
            sample_rankings.append(sample_probs)
        
        sample_rankings = torch.stack(sample_rankings, dim=0)
        mean_ranking = sample_rankings.mean(dim=0)
        ranking_uncertainty = sample_rankings.std(dim=0)
        
        if not return_dict:
            return mean_ranking
        
        return {
            "mean_ranking": mean_ranking,
            "ranking_uncertainty": ranking_uncertainty,
            "sample_rankings": sample_rankings,
            "parameter_samples": parameter_samples
        }

class PartialRankingPL(PlackettLuceModel):
    def __init__(
        self,
        reward_model: BaseRewardModel,
        missing_penalty: float = 0.1,
        **kwargs
    ):
        super().__init__(reward_model, **kwargs)
        self.missing_penalty = missing_penalty
    
    def forward(
        self,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
        partial_ranking: torch.Tensor,
        ranking_mask: torch.Tensor,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        num_items = len(input_ids_list)
        
        rewards = []
        for i in range(num_items):
            reward = self.reward_model(input_ids_list[i], attention_mask_list[i], return_dict=False)
            rewards.append(reward)
        
        rewards = torch.stack(rewards, dim=1)
        
        batch_size = rewards.size(0)
        ranking_probs = torch.zeros(batch_size, num_items, device=rewards.device)
        
        for batch_idx in range(batch_size):
            observed_positions = ranking_mask[batch_idx].nonzero().flatten()
            observed_items = partial_ranking[batch_idx][observed_positions]
            
            if len(observed_positions) > 0:
                observed_rewards = rewards[batch_idx][observed_items]
                observed_probs = self._standard_plackett_luce(observed_rewards.unsqueeze(0)).squeeze(0)
                
                for i, pos in enumerate(observed_positions):
                    ranking_probs[batch_idx, pos] = observed_probs[i]
                
                missing_positions = (~ranking_mask[batch_idx]).nonzero().flatten()
                if len(missing_positions) > 0:
                    uniform_prob = self.missing_penalty / len(missing_positions)
                    ranking_probs[batch_idx, missing_positions] = uniform_prob
        
        if not return_dict:
            return ranking_probs
        
        return {
            "ranking_probabilities": ranking_probs,
            "rewards": rewards,
            "observed_positions": observed_positions,
            "missing_penalty": self.missing_penalty
        }

class PlackettLuceTrainer:
    def __init__(
        self,
        model: PlackettLuceModel,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    def train_step(
        self,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
        true_rankings: torch.Tensor
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.model.compute_ranking_loss(
            input_ids_list,
            attention_mask_list,
            true_rankings
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        with torch.no_grad():
            output = self.model(input_ids_list, attention_mask_list, return_dict=True)
            predicted_rankings = torch.argsort(output["rewards"], dim=1, descending=True)
            
            kendall_tau = self._compute_kendall_tau(predicted_rankings, true_rankings)
            spearman_rho = self._compute_spearman_correlation(predicted_rankings, true_rankings)
        
        return {
            "loss": loss.item(),
            "kendall_tau": kendall_tau,
            "spearman_rho": spearman_rho
        }
    
    def _compute_kendall_tau(self, pred_rankings: torch.Tensor, true_rankings: torch.Tensor) -> float:
        batch_size, num_items = pred_rankings.shape
        total_tau = 0.0
        
        for b in range(batch_size):
            concordant = 0
            discordant = 0
            
            for i in range(num_items):
                for j in range(i + 1, num_items):
                    pred_order = (pred_rankings[b, i] < pred_rankings[b, j])
                    true_order = (true_rankings[b, i] < true_rankings[b, j])
                    
                    if pred_order == true_order:
                        concordant += 1
                    else:
                        discordant += 1
            
            tau = (concordant - discordant) / (concordant + discordant) if (concordant + discordant) > 0 else 0
            total_tau += tau
        
        return total_tau / batch_size
    
    def _compute_spearman_correlation(self, pred_rankings: torch.Tensor, true_rankings: torch.Tensor) -> float:
        batch_size = pred_rankings.size(0)
        total_rho = 0.0
        
        for b in range(batch_size):
            pred_ranks = torch.argsort(torch.argsort(pred_rankings[b]))
            true_ranks = torch.argsort(torch.argsort(true_rankings[b]))
            
            pred_ranks = pred_ranks.float()
            true_ranks = true_ranks.float()
            
            pred_mean = pred_ranks.mean()
            true_mean = true_ranks.mean()
            
            numerator = torch.sum((pred_ranks - pred_mean) * (true_ranks - true_mean))
            denominator = torch.sqrt(torch.sum((pred_ranks - pred_mean)**2) * torch.sum((true_ranks - true_mean)**2))
            
            rho = numerator / (denominator + 1e-8)
            total_rho += rho.item()
        
        return total_rho / batch_size

class RankingMetrics:
    @staticmethod
    def normalized_discounted_cumulative_gain(
        predicted_ranking: torch.Tensor,
        true_relevance: torch.Tensor,
        k: Optional[int] = None
    ) -> float:
        if k is None:
            k = predicted_ranking.size(-1)
        
        dcg = RankingMetrics._compute_dcg(predicted_ranking, true_relevance, k)
        ideal_ranking = torch.argsort(true_relevance, descending=True)
        idcg = RankingMetrics._compute_dcg(ideal_ranking, true_relevance, k)
        
        return (dcg / (idcg + 1e-8)).item()
    
    @staticmethod
    def _compute_dcg(ranking: torch.Tensor, relevance: torch.Tensor, k: int) -> torch.Tensor:
        dcg = 0.0
        for i in range(min(k, ranking.size(-1))):
            item_idx = ranking[i]
            rel = relevance[item_idx]
            dcg += (2**rel - 1) / math.log2(i + 2)
        return torch.tensor(dcg)
    
    @staticmethod
    def mean_reciprocal_rank(
        predicted_rankings: torch.Tensor,
        true_rankings: torch.Tensor
    ) -> float:
        batch_size = predicted_rankings.size(0)
        mrr = 0.0
        
        for b in range(batch_size):
            true_best = true_rankings[b, 0]
            predicted_rank = (predicted_rankings[b] == true_best).nonzero(as_tuple=True)[0]
            
            if len(predicted_rank) > 0:
                mrr += 1.0 / (predicted_rank[0].item() + 1)
        
        return mrr / batch_size
    
    @staticmethod
    def precision_at_k(
        predicted_ranking: torch.Tensor,
        true_ranking: torch.Tensor,
        k: int
    ) -> float:
        top_k_pred = predicted_ranking[:k]
        top_k_true = true_ranking[:k]
        
        intersection = len(set(top_k_pred.tolist()) & set(top_k_true.tolist()))
        return intersection / k