import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from typing import Dict, List, Optional, Tuple, Union
import math
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from reward_models.base_reward_model import BaseRewardModel, RewardOutput, RewardType, RewardHead

class TransformerRewardModel(BaseRewardModel):
    def __init__(
        self,
        model_name_or_path: str,
        reward_type: RewardType = RewardType.SCALAR,
        num_objectives: int = 1,
        pooling_strategy: str = "last_token",
        hidden_size: Optional[int] = None,
        dropout: float = 0.1,
        normalize_rewards: bool = True,
        gradient_checkpointing: bool = False,
        use_flash_attention: bool = False
    ):
        self.pooling_strategy = pooling_strategy
        self.gradient_checkpointing = gradient_checkpointing
        self.use_flash_attention = use_flash_attention
        
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        if hidden_size is None:
            hidden_size = self.config.hidden_size
            
        super().__init__(
            model_name_or_path=model_name_or_path,
            reward_type=reward_type,
            num_objectives=num_objectives,
            hidden_size=hidden_size,
            dropout=dropout,
            normalize_rewards=normalize_rewards
        )
        
        self.backbone = AutoModel.from_pretrained(
            model_name_or_path,
            config=self.config,
            torch_dtype=torch.float32
        )
        
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.pooling_layer = self._create_pooling_layer()
        
    def _create_reward_head(self) -> nn.Module:
        if self.reward_type == RewardType.SCALAR:
            return RewardHead(self.hidden_size, 1, "mlp", self.dropout)
        elif self.reward_type == RewardType.MULTI_OBJECTIVE:
            return RewardHead(self.hidden_size, self.num_objectives, "mlp", self.dropout)
        elif self.reward_type == RewardType.DISTRIBUTIONAL:
            return DistributionalRewardHead(self.hidden_size, self.num_objectives, self.dropout)
        elif self.reward_type == RewardType.PROCESS_LEVEL:
            return ProcessRewardHead(self.hidden_size, self.num_objectives, self.dropout)
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")
    
    def _create_pooling_layer(self) -> nn.Module:
        if self.pooling_strategy == "last_token":
            return LastTokenPooling()
        elif self.pooling_strategy == "mean":
            return MeanPooling()
        elif self.pooling_strategy == "max":
            return MaxPooling()
        elif self.pooling_strategy == "attention":
            return AttentionPooling(self.hidden_size, self.dropout)
        elif self.pooling_strategy == "weighted_mean":
            return WeightedMeanPooling(self.hidden_size)
        else:
            raise ValueError(f"Unknown pooling_strategy: {self.pooling_strategy}")
    
    def _encode_sequence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
        
        hidden_states = outputs.last_hidden_state
        pooled_output = self.pooling_layer(hidden_states, attention_mask)
        
        return pooled_output, outputs
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
        return_hidden_states: bool = False
    ) -> Union[torch.Tensor, RewardOutput]:
        pooled_output, backbone_outputs = self._encode_sequence(input_ids, attention_mask)
        pooled_output = self.dropout_layer(pooled_output)
        
        if self.reward_type == RewardType.PROCESS_LEVEL:
            rewards, process_rewards = self.reward_head(
                pooled_output, 
                backbone_outputs.hidden_states, 
                attention_mask
            )
        else:
            rewards = self.reward_head(pooled_output)
            process_rewards = None
        
        if self.normalize_rewards:
            rewards = self.normalize_reward(rewards)
        
        if not return_dict:
            return rewards
        
        uncertainty = self.compute_uncertainty(pooled_output)
        attention_weights = self.get_attention_weights() if hasattr(self, '_attention_weights') else None
        
        return RewardOutput(
            rewards=rewards,
            uncertainty=uncertainty,
            process_rewards=process_rewards,
            hidden_states=pooled_output if return_hidden_states else None,
            attention_weights=attention_weights
        )
    
    def compute_uncertainty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        uncertainty_head = nn.Linear(self.hidden_size, 1).to(hidden_states.device)
        return torch.sigmoid(uncertainty_head(hidden_states))
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        return getattr(self, '_attention_weights', None)

class LastTokenPooling(nn.Module):
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        
        return hidden_states[torch.arange(batch_size), sequence_lengths]

class MeanPooling(nn.Module):
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

class MaxPooling(nn.Module):
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        hidden_states[input_mask_expanded == 0] = -1e9
        return torch.max(hidden_states, 1)[0]

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(hidden_states).squeeze(-1)
        attention_weights = attention_weights.masked_fill(attention_mask == 0, -1e9)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        pooled_output = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        return self.dropout(pooled_output)

class WeightedMeanPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        weighted_hidden = hidden_states * self.weight.unsqueeze(0).unsqueeze(0)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(weighted_hidden.size()).float()
        sum_embeddings = torch.sum(weighted_hidden * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

class DistributionalRewardHead(nn.Module):
    def __init__(self, hidden_size: int, num_objectives: int = 1, dropout: float = 0.1, num_atoms: int = 51):
        super().__init__()
        self.num_objectives = num_objectives
        self.num_atoms = num_atoms
        self.v_min = -10.0
        self.v_max = 10.0
        
        self.delta_z = (self.v_max - self.v_min) / (num_atoms - 1)
        self.register_buffer('support', torch.linspace(self.v_min, self.v_max, num_atoms))
        
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_objectives * num_atoms)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.head(hidden_states)
        logits = logits.view(-1, self.num_objectives, self.num_atoms)
        probabilities = F.softmax(logits, dim=-1)
        
        expected_rewards = torch.sum(probabilities * self.support.unsqueeze(0).unsqueeze(0), dim=-1)
        return expected_rewards

class ProcessRewardHead(nn.Module):
    def __init__(self, hidden_size: int, num_objectives: int = 1, dropout: float = 0.1):
        super().__init__()
        self.num_objectives = num_objectives
        
        self.final_reward_head = RewardHead(hidden_size, num_objectives, "mlp", dropout)
        self.step_reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_objectives)
        )
        
    def forward(
        self, 
        pooled_output: torch.Tensor, 
        all_hidden_states: Tuple[torch.Tensor, ...], 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        final_reward = self.final_reward_head(pooled_output)
        
        process_rewards = []
        for layer_hidden in all_hidden_states[-4:]:
            step_rewards = self.step_reward_head(layer_hidden)
            
            masked_rewards = step_rewards * attention_mask.unsqueeze(-1).float()
            process_rewards.append(masked_rewards)
        
        return final_reward, process_rewards

class MultiScaleTransformerRM(TransformerRewardModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.local_attention = nn.MultiheadAttention(
            self.hidden_size, 
            num_heads=8, 
            dropout=self.dropout,
            batch_first=True
        )
        
        self.global_attention = nn.MultiheadAttention(
            self.hidden_size,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
        
        self.fusion_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
    def _encode_sequence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled_output, backbone_outputs = super()._encode_sequence(input_ids, attention_mask)
        
        hidden_states = backbone_outputs.last_hidden_state
        
        local_features, _ = self.local_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~attention_mask.bool()
        )
        
        global_features, _ = self.global_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~attention_mask.bool()
        )
        
        combined_features = torch.cat([local_features, global_features], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        
        enhanced_pooled = self.pooling_layer(fused_features, attention_mask)
        
        return enhanced_pooled, backbone_outputs

class HierarchicalTransformerRM(TransformerRewardModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.sentence_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dropout=self.dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.document_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dropout=self.dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
    def _segment_into_sentences(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        segment_size = 64
        
        segments = []
        for i in range(0, seq_len, segment_size):
            end_idx = min(i + segment_size, seq_len)
            segment = hidden_states[:, i:end_idx, :]
            segment_mask = attention_mask[:, i:end_idx]
            
            if segment_mask.sum() > 0:
                segment_pooled = MeanPooling()(segment, segment_mask)
                segments.append(segment_pooled)
        
        if segments:
            return torch.stack(segments, dim=1)
        else:
            return hidden_states.mean(dim=1, keepdim=True)
    
    def _encode_sequence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled_output, backbone_outputs = super()._encode_sequence(input_ids, attention_mask)
        
        hidden_states = backbone_outputs.last_hidden_state
        
        sentence_features = self._segment_into_sentences(hidden_states, attention_mask)
        sentence_encoded = self.sentence_encoder(sentence_features)
        
        document_features = self.document_encoder(sentence_encoded)
        document_pooled = document_features.mean(dim=1)
        
        return document_pooled, backbone_outputs