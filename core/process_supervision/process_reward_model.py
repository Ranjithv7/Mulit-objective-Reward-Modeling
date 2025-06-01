import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from reward_models.base_reward_model import BaseRewardModel, RewardOutput, RewardType
from reward_models.transformer_reward_model import TransformerRewardModel

class ProcessRewardModel(TransformerRewardModel):
    def __init__(
        self,
        *args,
        step_granularity: str = "token",
        process_supervision_type: str = "dense",
        outcome_weight: float = 0.5,
        process_weight: float = 0.5,
        step_pooling: str = "attention",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.step_granularity = step_granularity
        self.process_supervision_type = process_supervision_type
        self.outcome_weight = outcome_weight
        self.process_weight = process_weight
        self.step_pooling = step_pooling
        
        self.process_head = ProcessHead(
            self.hidden_size,
            self.num_objectives,
            step_pooling
        )
        
        self.outcome_head = nn.Linear(self.hidden_size, self.num_objectives)
        
        if process_supervision_type == "sparse":
            self.step_selector = StepSelector(self.hidden_size)
        
        self.step_confidence = StepConfidenceEstimator(self.hidden_size)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: Optional[torch.Tensor] = None,
        step_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, RewardOutput]:
        pooled_output, backbone_outputs = self._encode_sequence(input_ids, attention_mask)
        hidden_states = backbone_outputs.last_hidden_state
        
        if self.step_granularity == "sentence":
            step_features, step_boundaries = self._extract_sentence_steps(hidden_states, attention_mask)
        elif self.step_granularity == "reasoning":
            step_features, step_boundaries = self._extract_reasoning_steps(hidden_states, attention_mask, step_boundaries)
        else:  # token level
            step_features = hidden_states
            step_boundaries = torch.arange(hidden_states.size(1)).unsqueeze(0).expand(hidden_states.size(0), -1)
        
        step_rewards = self.process_head(step_features, attention_mask)
        outcome_reward = self.outcome_head(pooled_output)
        
        step_confidences = self.step_confidence(step_features)
        
        if self.process_supervision_type == "sparse":
            selected_steps = self.step_selector(step_features, step_confidences)
            step_rewards = step_rewards * selected_steps.unsqueeze(-1)
        
        combined_reward = (
            self.outcome_weight * outcome_reward +
            self.process_weight * self._aggregate_step_rewards(step_rewards, attention_mask)
        )
        
        if self.normalize_rewards:
            combined_reward = self.normalize_reward(combined_reward)
            step_rewards = self.normalize_reward(step_rewards)
        
        if not return_dict:
            return combined_reward
        
        return RewardOutput(
            rewards=combined_reward,
            process_rewards=[step_rewards],
            hidden_states=pooled_output,
            objective_breakdown={
                "outcome_reward": outcome_reward,
                "step_rewards": step_rewards,
                "step_confidences": step_confidences,
                "step_boundaries": step_boundaries
            }
        )
    
    def _extract_sentence_steps(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Simple sentence segmentation based on punctuation tokens
        # In practice, use proper sentence segmentation
        sentence_ends = self._find_sentence_boundaries(attention_mask)
        
        max_sentences = max(len(ends) for ends in sentence_ends)
        sentence_features = torch.zeros(batch_size, max_sentences, hidden_size, device=hidden_states.device)
        sentence_boundaries = torch.zeros(batch_size, max_sentences, dtype=torch.long, device=hidden_states.device)
        
        for b, ends in enumerate(sentence_ends):
            start = 0
            for i, end in enumerate(ends):
                if i < max_sentences:
                    sentence_repr = hidden_states[b, start:end+1].mean(dim=0)
                    sentence_features[b, i] = sentence_repr
                    sentence_boundaries[b, i] = end
                    start = end + 1
        
        return sentence_features, sentence_boundaries
    
    def _extract_reasoning_steps(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if step_boundaries is not None:
            return self._extract_explicit_steps(hidden_states, step_boundaries)
        else:
            return self._extract_implicit_steps(hidden_states, attention_mask)
    
    def _extract_explicit_steps(
        self,
        hidden_states: torch.Tensor,
        step_boundaries: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        max_steps = step_boundaries.size(1)
        
        step_features = torch.zeros(batch_size, max_steps, hidden_size, device=hidden_states.device)
        
        for b in range(batch_size):
            start = 0
            for s in range(max_steps):
                end = step_boundaries[b, s].item()
                if end > start and end < seq_len:
                    step_repr = hidden_states[b, start:end+1].mean(dim=0)
                    step_features[b, s] = step_repr
                    start = end + 1
        
        return step_features, step_boundaries
    
    def _extract_implicit_steps(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use attention patterns to identify reasoning steps
        step_detector = ImplicitStepDetector(self.hidden_size)
        step_probabilities = step_detector(hidden_states, attention_mask)
        
        # Extract steps based on high attention scores
        step_threshold = 0.7
        step_boundaries = (step_probabilities > step_threshold).nonzero(as_tuple=True)
        
        return hidden_states, step_boundaries[1].unsqueeze(0)
    
    def _find_sentence_boundaries(self, attention_mask: torch.Tensor) -> List[List[int]]:
        # Simplified sentence boundary detection
        # In practice, use proper NLP tools
        batch_boundaries = []
        
        for b in range(attention_mask.size(0)):
            mask = attention_mask[b]
            seq_len = mask.sum().item()
            
            # Mock sentence boundaries every ~20 tokens
            boundaries = list(range(19, seq_len, 20))
            if boundaries[-1] != seq_len - 1:
                boundaries.append(seq_len - 1)
            
            batch_boundaries.append(boundaries)
        
        return batch_boundaries
    
    def _aggregate_step_rewards(self, step_rewards: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.step_pooling == "mean":
            return step_rewards.mean(dim=1)
        elif self.step_pooling == "max":
            return step_rewards.max(dim=1)[0]
        elif self.step_pooling == "weighted_mean":
            weights = F.softmax(step_rewards.sum(dim=-1), dim=1)
            return torch.sum(step_rewards * weights.unsqueeze(-1), dim=1)
        else:  # attention pooling
            attention_weights = F.softmax(step_rewards.sum(dim=-1), dim=1)
            return torch.sum(step_rewards * attention_weights.unsqueeze(-1), dim=1)
    
    def compute_process_loss(
        self,
        step_rewards: torch.Tensor,
        step_labels: torch.Tensor,
        step_mask: torch.Tensor
    ) -> torch.Tensor:
        masked_step_rewards = step_rewards * step_mask.unsqueeze(-1)
        masked_step_labels = step_labels * step_mask.unsqueeze(-1)
        
        step_loss = F.mse_loss(masked_step_rewards, masked_step_labels, reduction='none')
        step_loss = step_loss * step_mask.unsqueeze(-1)
        
        return step_loss.sum() / (step_mask.sum() + 1e-8)

class ProcessHead(nn.Module):
    def __init__(self, hidden_size: int, num_objectives: int = 1, pooling_type: str = "attention"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_objectives = num_objectives
        self.pooling_type = pooling_type
        
        self.step_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_objectives)
        )
        
        if pooling_type == "attention":
            self.attention_pool = nn.MultiheadAttention(
                hidden_size, num_heads=8, batch_first=True
            )
    
    def forward(self, step_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "attention":
            attended_features, _ = self.attention_pool(
                step_features, step_features, step_features,
                key_padding_mask=~attention_mask.bool()
            )
            step_rewards = self.step_encoder(attended_features)
        else:
            step_rewards = self.step_encoder(step_features)
        
        return step_rewards

class StepSelector(nn.Module):
    def __init__(self, hidden_size: int, selection_threshold: float = 0.5):
        super().__init__()
        self.selection_threshold = selection_threshold
        
        self.selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, step_features: torch.Tensor, step_confidences: torch.Tensor) -> torch.Tensor:
        selection_scores = self.selector(step_features).squeeze(-1)
        
        # Combine selection scores with confidence
        combined_scores = selection_scores * step_confidences
        
        # Binary selection based on threshold
        selected_steps = (combined_scores > self.selection_threshold).float()
        
        return selected_steps

class StepConfidenceEstimator(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, step_features: torch.Tensor) -> torch.Tensor:
        return self.confidence_estimator(step_features).squeeze(-1)

class ImplicitStepDetector(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.step_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.context_attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, batch_first=True
        )
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Use self-attention to capture context
        attended_states, attention_weights = self.context_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Detect step boundaries based on attention patterns
        step_probabilities = self.step_detector(attended_states).squeeze(-1)
        
        return step_probabilities

class HierarchicalProcessModel(ProcessRewardModel):
    def __init__(
        self,
        *args,
        hierarchy_levels: List[str] = ["token", "phrase", "sentence", "paragraph"],
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.hierarchy_levels = hierarchy_levels
        
        self.level_processors = nn.ModuleDict({
            level: ProcessHead(self.hidden_size, self.num_objectives)
            for level in hierarchy_levels
        })
        
        self.level_aggregator = nn.Sequential(
            nn.Linear(self.hidden_size * len(hierarchy_levels), self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_objectives)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True
    ) -> Union[torch.Tensor, RewardOutput]:
        pooled_output, backbone_outputs = self._encode_sequence(input_ids, attention_mask)
        hidden_states = backbone_outputs.last_hidden_state
        
        level_rewards = {}
        level_features = []
        
        for level in self.hierarchy_levels:
            if level == "token":
                level_input = hidden_states
            elif level == "phrase":
                level_input = self._extract_phrase_features(hidden_states, attention_mask)
            elif level == "sentence":
                level_input, _ = self._extract_sentence_steps(hidden_states, attention_mask)
            elif level == "paragraph":
                level_input = self._extract_paragraph_features(hidden_states, attention_mask)
            
            level_reward = self.level_processors[level](level_input, attention_mask)
            level_rewards[level] = level_reward
            level_features.append(level_input.mean(dim=1))
        
        # Aggregate across levels
        combined_features = torch.cat(level_features, dim=-1)
        hierarchical_reward = self.level_aggregator(combined_features)
        
        if not return_dict:
            return hierarchical_reward
        
        return RewardOutput(
            rewards=hierarchical_reward,
            hidden_states=pooled_output,
            objective_breakdown=level_rewards
        )
    
    def _extract_phrase_features(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Extract phrase-level features (every 5-10 tokens)
        phrase_length = 7
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        num_phrases = seq_len // phrase_length
        phrase_features = torch.zeros(batch_size, num_phrases, hidden_size, device=hidden_states.device)
        
        for i in range(num_phrases):
            start = i * phrase_length
            end = min((i + 1) * phrase_length, seq_len)
            phrase_features[:, i] = hidden_states[:, start:end].mean(dim=1)
        
        return phrase_features
    
    def _extract_paragraph_features(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Extract paragraph-level features (every 50-100 tokens)
        paragraph_length = 75
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        num_paragraphs = max(1, seq_len // paragraph_length)
        paragraph_features = torch.zeros(batch_size, num_paragraphs, hidden_size, device=hidden_states.device)
        
        for i in range(num_paragraphs):
            start = i * paragraph_length
            end = min((i + 1) * paragraph_length, seq_len)
            paragraph_features[:, i] = hidden_states[:, start:end].mean(dim=1)
        
        return paragraph_features

class VerificationAugmentedPRM(ProcessRewardModel):
    def __init__(
        self,
        *args,
        verification_model: Optional[nn.Module] = None,
        verification_weight: float = 0.3,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        if verification_model is None:
            self.verification_model = StepVerifier(self.hidden_size)
        else:
            self.verification_model = verification_model
        
        self.verification_weight = verification_weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True
    ) -> Union[torch.Tensor, RewardOutput]:
        # Get base process rewards
        base_output = super().forward(input_ids, attention_mask, return_dict=True)
        
        # Verify each step
        step_rewards = base_output.objective_breakdown["step_rewards"]
        verification_scores = self.verification_model(step_rewards, attention_mask)
        
        # Combine process rewards with verification
        verified_step_rewards = step_rewards * verification_scores.unsqueeze(-1)
        
        # Re-aggregate with verification
        verified_reward = (
            self.outcome_weight * base_output.objective_breakdown["outcome_reward"] +
            self.process_weight * self._aggregate_step_rewards(verified_step_rewards, attention_mask)
        )
        
        if not return_dict:
            return verified_reward
        
        return RewardOutput(
            rewards=verified_reward,
            process_rewards=[verified_step_rewards],
            hidden_states=base_output.hidden_states,
            objective_breakdown={
                **base_output.objective_breakdown,
                "verification_scores": verification_scores,
                "verified_step_rewards": verified_step_rewards
            }
        )

class StepVerifier(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.verifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, step_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        verification_scores = self.verifier(step_features).squeeze(-1)
        
        # Mask out padded positions
        verification_scores = verification_scores * attention_mask.float()
        
        return verification_scores

class ProcessRewardTrainer:
    def __init__(
        self,
        model: ProcessRewardModel,
        learning_rate: float = 1e-4,
        process_loss_weight: float = 0.5,
        outcome_loss_weight: float = 0.5
    ):
        self.model = model
        self.process_loss_weight = process_loss_weight
        self.outcome_loss_weight = outcome_loss_weight
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        output = self.model(
            batch["input_ids"],
            batch["attention_mask"],
            batch.get("step_boundaries"),
            batch.get("step_labels"),
            return_dict=True
        )
        
        # Outcome loss
        outcome_loss = F.mse_loss(
            output.rewards,
            batch["outcome_labels"]
        )
        
        # Process loss
        if "step_labels" in batch:
            process_loss = self.model.compute_process_loss(
                output.objective_breakdown["step_rewards"],
                batch["step_labels"],
                batch.get("step_mask", batch["attention_mask"])
            )
        else:
            process_loss = torch.tensor(0.0)
        
        # Combined loss
        total_loss = (
            self.outcome_loss_weight * outcome_loss +
            self.process_loss_weight * process_loss
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "outcome_loss": outcome_loss.item(),
            "process_loss": process_loss.item() if isinstance(process_loss, torch.Tensor) else process_loss
        }