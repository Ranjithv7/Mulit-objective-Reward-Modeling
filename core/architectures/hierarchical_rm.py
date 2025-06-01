import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math

from reward_models.base_reward_model import BaseRewardModel, RewardOutput, RewardType
from reward_models.transformer_reward_model import TransformerRewardModel

class HierarchicalRewardModel(TransformerRewardModel):
    def __init__(
        self,
        *args,
        hierarchy_levels: List[str] = ["word", "sentence", "paragraph", "document"],
        aggregation_method: str = "attention",
        level_weights: Optional[List[float]] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.hierarchy_levels = hierarchy_levels
        self.aggregation_method = aggregation_method
        self.num_levels = len(hierarchy_levels)
        
        if level_weights is None:
            level_weights = [1.0 / self.num_levels] * self.num_levels
        self.register_buffer('level_weights', torch.tensor(level_weights))
        
        self.level_encoders = nn.ModuleDict({
            level: HierarchicalEncoder(self.hidden_size, level)
            for level in hierarchy_levels
        })
        
        self.level_reward_heads = nn.ModuleDict({
            level: nn.Linear(self.hidden_size, self.num_objectives)
            for level in hierarchy_levels
        })
        
        if aggregation_method == "attention":
            self.aggregator = AttentionAggregator(self.hidden_size, self.num_levels)
        elif aggregation_method == "hierarchical":
            self.aggregator = HierarchicalAggregator(self.hidden_size, self.num_levels)
        elif aggregation_method == "graph":
            self.aggregator = GraphAggregator(self.hidden_size, self.num_levels)
        else:
            self.aggregator = WeightedAggregator(self.num_levels)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
        return_level_rewards: bool = False
    ) -> Union[torch.Tensor, RewardOutput]:
        pooled_output, backbone_outputs = self._encode_sequence(input_ids, attention_mask)
        hidden_states = backbone_outputs.last_hidden_state
        
        level_features = {}
        level_rewards = {}
        
        for level in self.hierarchy_levels:
            features = self.level_encoders[level](hidden_states, attention_mask)
            level_features[level] = features
            level_rewards[level] = self.level_reward_heads[level](features)
        
        aggregated_reward = self.aggregator(level_features, level_rewards)
        
        if self.normalize_rewards:
            aggregated_reward = self.normalize_reward(aggregated_reward)
        
        if not return_dict:
            return aggregated_reward
        
        return RewardOutput(
            rewards=aggregated_reward,
            hidden_states=pooled_output,
            objective_breakdown=level_rewards if return_level_rewards else None
        )

class HierarchicalEncoder(nn.Module):
    def __init__(self, hidden_size: int, level: str):
        super().__init__()
        self.level = level
        self.hidden_size = hidden_size
        
        if level == "word":
            self.encoder = WordLevelEncoder(hidden_size)
        elif level == "sentence":
            self.encoder = SentenceLevelEncoder(hidden_size)
        elif level == "paragraph":
            self.encoder = ParagraphLevelEncoder(hidden_size)
        elif level == "document":
            self.encoder = DocumentLevelEncoder(hidden_size)
        else:
            raise ValueError(f"Unknown hierarchy level: {level}")
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(hidden_states, attention_mask)

class WordLevelEncoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.self_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~attention_mask.bool()
        )
        
        normalized = self.layer_norm(attended + hidden_states)
        
        masked_states = normalized * attention_mask.unsqueeze(-1).float()
        pooled = masked_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).float()
        
        return pooled

class SentenceLevelEncoder(nn.Module):
    def __init__(self, hidden_size: int, sentence_length: int = 30):
        super().__init__()
        self.sentence_length = sentence_length
        
        self.sentence_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.sentence_pooling = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        sentence_features = []
        for i in range(0, seq_len, self.sentence_length):
            end_idx = min(i + self.sentence_length, seq_len)
            sentence = hidden_states[:, i:end_idx, :]
            sentence_mask = attention_mask[:, i:end_idx]
            
            if sentence_mask.sum() > 0:
                sentence_pooled = (sentence * sentence_mask.unsqueeze(-1).float()).sum(dim=1)
                sentence_pooled = sentence_pooled / (sentence_mask.sum(dim=1, keepdim=True).float() + 1e-8)
                sentence_features.append(sentence_pooled)
        
        if sentence_features:
            sentence_tensor = torch.stack(sentence_features, dim=1)
            transformed = self.sentence_transformer(sentence_tensor)
            document_features = transformed.mean(dim=1)
        else:
            document_features = hidden_states.mean(dim=1)
        
        return document_features

class ParagraphLevelEncoder(nn.Module):
    def __init__(self, hidden_size: int, paragraph_length: int = 100):
        super().__init__()
        self.paragraph_length = paragraph_length
        
        self.paragraph_encoder = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=2, batch_first=True, bidirectional=True
        )
        
        self.projection = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_pool = AttentionPooling(hidden_size)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        paragraph_features = []
        for i in range(0, seq_len, self.paragraph_length):
            end_idx = min(i + self.paragraph_length, seq_len)
            paragraph = hidden_states[:, i:end_idx, :]
            paragraph_mask = attention_mask[:, i:end_idx]
            
            if paragraph_mask.sum() > 0:
                lstm_out, _ = self.paragraph_encoder(paragraph)
                projected = self.projection(lstm_out)
                
                pooled = self.attention_pool(projected, paragraph_mask)
                paragraph_features.append(pooled)
        
        if paragraph_features:
            paragraph_tensor = torch.stack(paragraph_features, dim=1)
            document_features = paragraph_tensor.mean(dim=1)
        else:
            document_features = hidden_states.mean(dim=1)
        
        return document_features

class DocumentLevelEncoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.global_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )
        
        self.hierarchical_conv = nn.Conv1d(
            hidden_size, hidden_size,
            kernel_size=5, padding=2, groups=hidden_size
        )
        
        self.document_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                batch_first=True
            ),
            num_layers=3
        )
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        global_attended, _ = self.global_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~attention_mask.bool()
        )
        
        conv_features = self.hierarchical_conv(
            global_attended.transpose(1, 2)
        ).transpose(1, 2)
        
        combined = global_attended + conv_features
        
        document_features = self.document_transformer(combined)
        
        masked_features = document_features * attention_mask.unsqueeze(-1).float()
        pooled = masked_features.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).float()
        
        return pooled

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = self.attention(hidden_states).squeeze(-1)
        weights = weights.masked_fill(~mask.bool(), -1e9)
        weights = F.softmax(weights, dim=-1)
        
        return torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)

class AttentionAggregator(nn.Module):
    def __init__(self, hidden_size: int, num_levels: int):
        super().__init__()
        self.num_levels = num_levels
        
        self.level_attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, batch_first=True
        )
        
        self.level_weights = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_levels),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        level_features: Dict[str, torch.Tensor],
        level_rewards: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        features_stack = torch.stack(list(level_features.values()), dim=1)
        rewards_stack = torch.stack(list(level_rewards.values()), dim=1)
        
        attended_features, _ = self.level_attention(
            features_stack, features_stack, features_stack
        )
        
        global_context = attended_features.mean(dim=1)
        weights = self.level_weights(global_context)
        
        weighted_rewards = torch.sum(
            rewards_stack * weights.unsqueeze(-1),
            dim=1
        )
        
        return weighted_rewards

class HierarchicalAggregator(nn.Module):
    def __init__(self, hidden_size: int, num_levels: int):
        super().__init__()
        self.num_levels = num_levels
        
        self.bottom_up_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_levels - 1)
        ])
        
        self.top_down_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_levels - 1)
        ])
        
        self.fusion_layer = nn.Linear(hidden_size * num_levels, hidden_size)
    
    def forward(
        self,
        level_features: Dict[str, torch.Tensor],
        level_rewards: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        features_list = list(level_features.values())
        rewards_list = list(level_rewards.values())
        
        bottom_up_features = [features_list[0]]
        for i, layer in enumerate(self.bottom_up_layers):
            enhanced = layer(bottom_up_features[-1]) + features_list[i + 1]
            bottom_up_features.append(enhanced)
        
        top_down_features = [bottom_up_features[-1]]
        for i, layer in enumerate(reversed(self.top_down_layers)):
            enhanced = layer(top_down_features[-1]) + bottom_up_features[-(i + 2)]
            top_down_features.append(enhanced)
        
        top_down_features.reverse()
        
        fused_features = self.fusion_layer(torch.cat(top_down_features, dim=-1))
        
        enhanced_rewards = []
        for i, reward in enumerate(rewards_list):
            enhanced = reward + F.linear(top_down_features[i], torch.eye(reward.size(-1), device=reward.device))
            enhanced_rewards.append(enhanced)
        
        final_reward = torch.stack(enhanced_rewards, dim=1).mean(dim=1)
        
        return final_reward

class GraphAggregator(nn.Module):
    def __init__(self, hidden_size: int, num_levels: int):
        super().__init__()
        self.num_levels = num_levels
        
        self.graph_conv_layers = nn.ModuleList([
            GraphConvLayer(hidden_size, hidden_size)
            for _ in range(2)
        ])
        
        self.level_embeddings = nn.Embedding(num_levels, hidden_size)
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(
        self,
        level_features: Dict[str, torch.Tensor],
        level_rewards: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        batch_size = next(iter(level_features.values())).size(0)
        
        node_features = torch.stack(list(level_features.values()), dim=1)
        
        level_indices = torch.arange(self.num_levels, device=node_features.device)
        level_embs = self.level_embeddings(level_indices).unsqueeze(0).expand(batch_size, -1, -1)
        
        node_features = node_features + level_embs
        
        adj_matrix = self._create_hierarchy_adjacency().to(node_features.device)
        
        for layer in self.graph_conv_layers:
            node_features = layer(node_features, adj_matrix)
        
        rewards_stack = torch.stack(list(level_rewards.values()), dim=1)
        
        graph_weights = self.readout(node_features).squeeze(-1)
        graph_weights = F.softmax(graph_weights, dim=1)
        
        aggregated_reward = torch.sum(
            rewards_stack * graph_weights.unsqueeze(-1),
            dim=1
        )
        
        return aggregated_reward
    
    def _create_hierarchy_adjacency(self) -> torch.Tensor:
        adj = torch.eye(self.num_levels)
        
        for i in range(self.num_levels - 1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1
        
        return adj

class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(out_features)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.matmul(adj.unsqueeze(0), x)
        x = self.activation(x)
        x = self.layer_norm(x)
        return x

class WeightedAggregator(nn.Module):
    def __init__(self, num_levels: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_levels) / num_levels)
    
    def forward(
        self,
        level_features: Dict[str, torch.Tensor],
        level_rewards: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        rewards_stack = torch.stack(list(level_rewards.values()), dim=1)
        weights = F.softmax(self.weights, dim=0)
        
        weighted_reward = torch.sum(
            rewards_stack * weights.view(1, -1, 1),
            dim=1
        )
        
        return weighted_reward

class AdaptiveHierarchicalModel(HierarchicalRewardModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.complexity_estimator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.num_levels),
            nn.Sigmoid()
        )
        
        self.dynamic_weights = nn.Parameter(torch.ones(self.num_levels, self.num_levels))
    
    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        
        if isinstance(result, RewardOutput):
            complexity_scores = self.complexity_estimator(result.hidden_states)
            
            adaptive_weights = torch.matmul(complexity_scores, self.dynamic_weights)
            adaptive_weights = F.softmax(adaptive_weights, dim=-1)
            
            if result.objective_breakdown:
                rewards_stack = torch.stack(list(result.objective_breakdown.values()), dim=1)
                adaptive_reward = torch.sum(
                    rewards_stack * adaptive_weights.unsqueeze(-1),
                    dim=1
                )
                result.rewards = adaptive_reward
        
        return result

class ProgressiveHierarchy(nn.Module):
    def __init__(self, hidden_size: int, max_levels: int = 5):
        super().__init__()
        self.max_levels = max_levels
        self.hidden_size = hidden_size
        
        self.level_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            ) for _ in range(max_levels)
        ])
        
        self.level_processors = nn.ModuleList([
            HierarchicalEncoder(hidden_size, f"level_{i}")
            for i in range(max_levels)
        ])
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        current_features = hidden_states.mean(dim=1)
        
        for i, (gate, processor) in enumerate(zip(self.level_gates, self.level_processors)):
            gate_value = gate(current_features)
            
            if gate_value.mean() > 0.5 or i == self.max_levels - 1:
                return processor(hidden_states, attention_mask)
            
            current_features = processor(hidden_states, attention_mask)
        
        return current_features