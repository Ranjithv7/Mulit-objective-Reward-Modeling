import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math

from reward_models.base_reward_model import BaseRewardModel, RewardOutput, RewardType
from reward_models.transformer_reward_model import TransformerRewardModel

class CrossAttentionRewardModel(TransformerRewardModel):
    def __init__(
        self,
        *args,
        cross_attention_type: str = "query_response",
        num_cross_layers: int = 2,
        cross_attention_heads: int = 8,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.cross_attention_type = cross_attention_type
        self.num_cross_layers = num_cross_layers
        self.cross_attention_heads = cross_attention_heads
        
        if cross_attention_type == "query_response":
            self.cross_attention = QueryResponseCrossAttention(
                self.hidden_size, 
                cross_attention_heads,
                num_cross_layers
            )
        elif cross_attention_type == "multi_scale":
            self.cross_attention = MultiScaleCrossAttention(
                self.hidden_size,
                cross_attention_heads,
                num_cross_layers
            )
        elif cross_attention_type == "hierarchical":
            self.cross_attention = HierarchicalCrossAttention(
                self.hidden_size,
                cross_attention_heads,
                num_cross_layers
            )
        elif cross_attention_type == "temporal":
            self.cross_attention = TemporalCrossAttention(
                self.hidden_size,
                cross_attention_heads,
                num_cross_layers
            )
        else:
            raise ValueError(f"Unknown cross_attention_type: {cross_attention_type}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        query_ids: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        return_cross_attention: bool = False
    ) -> Union[torch.Tensor, RewardOutput]:
        pooled_output, backbone_outputs = self._encode_sequence(input_ids, attention_mask)
        
        if query_ids is not None and query_mask is not None:
            query_output, query_backbone = self._encode_sequence(query_ids, query_mask)
            
            cross_attended, cross_weights = self.cross_attention(
                backbone_outputs.last_hidden_state,
                query_backbone.last_hidden_state,
                attention_mask,
                query_mask
            )
            
            pooled_output = self.pooling_layer(cross_attended, attention_mask)
        
        pooled_output = self.dropout_layer(pooled_output)
        rewards = self.reward_head(pooled_output)
        
        if self.normalize_rewards:
            rewards = self.normalize_reward(rewards)
        
        if not return_dict:
            return rewards
        
        return RewardOutput(
            rewards=rewards,
            hidden_states=pooled_output,
            attention_weights=cross_weights if return_cross_attention and 'cross_weights' in locals() else None
        )

class QueryResponseCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.response_projection = nn.Linear(hidden_size, hidden_size)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(
        self,
        response_hidden: torch.Tensor,
        query_hidden: torch.Tensor,
        response_mask: torch.Tensor,
        query_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_proj = self.query_projection(query_hidden)
        response_proj = self.response_projection(response_hidden)
        
        cross_weights = None
        for layer in self.cross_attention_layers:
            response_proj, weights = layer(
                response_proj, query_proj, query_proj,
                key_padding_mask=~query_mask.bool()
            )
            cross_weights = weights
        
        query_attended, _ = layer(
            query_proj, response_proj, response_proj,
            key_padding_mask=~response_mask.bool()
        )
        
        fused = self.fusion_layer(torch.cat([response_proj, query_attended], dim=-1))
        
        return fused, cross_weights

class MultiScaleCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        
        self.scales = [1, 2, 4, 8]
        self.scale_attentions = nn.ModuleList([
            CrossAttentionLayer(hidden_size, max(1, num_heads // len(self.scales)))
            for _ in self.scales
        ])
        
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_size * len(self.scales), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.temporal_conv = nn.Conv1d(
            hidden_size, hidden_size,
            kernel_size=3, padding=1, groups=hidden_size
        )
    
    def forward(
        self,
        response_hidden: torch.Tensor,
        query_hidden: torch.Tensor,
        response_mask: torch.Tensor,
        query_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale_outputs = []
        all_weights = []
        
        for scale, attention in zip(self.scales, self.scale_attentions):
            if scale > 1:
                downsampled_query = self._downsample(query_hidden, scale)
                downsampled_mask = self._downsample_mask(query_mask, scale)
            else:
                downsampled_query = query_hidden
                downsampled_mask = query_mask
            
            attended, weights = attention(
                response_hidden, downsampled_query, downsampled_query,
                key_padding_mask=~downsampled_mask.bool()
            )
            
            if scale > 1:
                attended = self._upsample(attended, response_hidden.size(1))
            
            scale_outputs.append(attended)
            all_weights.append(weights)
        
        fused = self.scale_fusion(torch.cat(scale_outputs, dim=-1))
        
        temporal_enhanced = self.temporal_conv(fused.transpose(1, 2)).transpose(1, 2)
        output = fused + temporal_enhanced
        
        return output, all_weights[0]
    
    def _downsample(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        return F.avg_pool1d(
            x.transpose(1, 2),
            kernel_size=scale,
            stride=scale,
            padding=0
        ).transpose(1, 2)
    
    def _downsample_mask(self, mask: torch.Tensor, scale: int) -> torch.Tensor:
        return F.max_pool1d(
            mask.float().unsqueeze(1),
            kernel_size=scale,
            stride=scale,
            padding=0
        ).squeeze(1).bool()
    
    def _upsample(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        return F.interpolate(
            x.transpose(1, 2),
            size=target_length,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

class HierarchicalCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        
        self.word_level_attention = CrossAttentionLayer(hidden_size, num_heads)
        self.sentence_level_attention = CrossAttentionLayer(hidden_size, num_heads)
        self.document_level_attention = CrossAttentionLayer(hidden_size, num_heads)
        
        self.hierarchical_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(3)
        ])
        
        self.level_weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(
        self,
        response_hidden: torch.Tensor,
        query_hidden: torch.Tensor,
        response_mask: torch.Tensor,
        query_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        word_attended, word_weights = self.word_level_attention(
            response_hidden, query_hidden, query_hidden,
            key_padding_mask=~query_mask.bool()
        )
        
        sentence_features = self._extract_sentence_features(response_hidden, response_mask)
        query_sentence_features = self._extract_sentence_features(query_hidden, query_mask)
        
        sentence_attended, sentence_weights = self.sentence_level_attention(
            sentence_features, query_sentence_features, query_sentence_features
        )
        
        document_features = sentence_features.mean(dim=1, keepdim=True)
        query_document_features = query_sentence_features.mean(dim=1, keepdim=True)
        
        document_attended, document_weights = self.document_level_attention(
            document_features, query_document_features, query_document_features
        )
        
        word_fused = self.hierarchical_fusion[0](
            torch.cat([response_hidden, word_attended], dim=-1)
        )
        
        sentence_expanded = self._expand_to_word_level(sentence_attended, response_hidden.size(1))
        sentence_fused = self.hierarchical_fusion[1](
            torch.cat([word_fused, sentence_expanded], dim=-1)
        )
        
        document_expanded = document_attended.expand(-1, response_hidden.size(1), -1)
        document_fused = self.hierarchical_fusion[2](
            torch.cat([sentence_fused, document_expanded], dim=-1)
        )
        
        level_weights = F.softmax(self.level_weights, dim=0)
        final_output = (
            level_weights[0] * word_fused +
            level_weights[1] * sentence_fused +
            level_weights[2] * document_fused
        )
        
        return final_output, word_weights
    
    def _extract_sentence_features(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sentence_length = 20
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        sentences = []
        for i in range(0, seq_len, sentence_length):
            end_idx = min(i + sentence_length, seq_len)
            sentence = hidden_states[:, i:end_idx, :]
            sentence_mask = mask[:, i:end_idx]
            
            if sentence_mask.sum() > 0:
                sentence_pooled = (sentence * sentence_mask.unsqueeze(-1).float()).sum(dim=1)
                sentence_pooled = sentence_pooled / (sentence_mask.sum(dim=1, keepdim=True).float() + 1e-8)
                sentences.append(sentence_pooled)
        
        if sentences:
            return torch.stack(sentences, dim=1)
        else:
            return hidden_states.mean(dim=1, keepdim=True)
    
    def _expand_to_word_level(self, sentence_features: torch.Tensor, target_length: int) -> torch.Tensor:
        return F.interpolate(
            sentence_features.transpose(1, 2),
            size=target_length,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

class TemporalCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        
        self.past_attention = CrossAttentionLayer(hidden_size, num_heads)
        self.present_attention = CrossAttentionLayer(hidden_size, num_heads)
        self.future_attention = CrossAttentionLayer(hidden_size, num_heads)
        
        self.temporal_encoding = TemporalPositionalEncoding(hidden_size)
        
        self.temporal_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.recurrent_layer = nn.GRU(
            hidden_size, hidden_size,
            batch_first=True, bidirectional=True
        )
        
        self.recurrent_projection = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(
        self,
        response_hidden: torch.Tensor,
        query_hidden: torch.Tensor,
        response_mask: torch.Tensor,
        query_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        response_encoded = self.temporal_encoding(response_hidden)
        query_encoded = self.temporal_encoding(query_hidden)
        
        seq_len = response_hidden.size(1)
        third = seq_len // 3
        
        past_response = response_encoded[:, :third, :]
        present_response = response_encoded[:, third:2*third, :]
        future_response = response_encoded[:, 2*third:, :]
        
        past_query = query_encoded[:, :third, :] if query_encoded.size(1) >= third else query_encoded
        present_query = query_encoded
        future_query = query_encoded[:, -third:, :] if query_encoded.size(1) >= third else query_encoded
        
        past_attended, past_weights = self.past_attention(
            past_response, past_query, past_query,
            key_padding_mask=~query_mask[:, :past_query.size(1)].bool()
        )
        
        present_attended, present_weights = self.present_attention(
            present_response, present_query, present_query,
            key_padding_mask=~query_mask.bool()
        )
        
        future_attended, future_weights = self.future_attention(
            future_response, future_query, future_query,
            key_padding_mask=~query_mask[:, -future_query.size(1):].bool()
        )
        
        temporal_concat = torch.cat([
            past_attended,
            present_attended,
            future_attended
        ], dim=1)
        
        if temporal_concat.size(1) != seq_len:
            temporal_concat = F.interpolate(
                temporal_concat.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        recurrent_output, _ = self.recurrent_layer(temporal_concat)
        recurrent_projected = self.recurrent_projection(recurrent_output)
        
        final_output = response_encoded + recurrent_projected
        
        return final_output, present_weights

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attended, attention_weights = self.cross_attention(
            query, key, value,
            key_padding_mask=key_padding_mask
        )
        
        query = self.layer_norm1(query + self.dropout(attended))
        
        ff_output = self.feedforward(query)
        output = self.layer_norm2(query + self.dropout(ff_output))
        
        return output, attention_weights

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           (-math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :].to(x.device)

class AdaptiveCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        
        self.attention_types = nn.ModuleList([
            QueryResponseCrossAttention(hidden_size, num_heads, num_layers),
            MultiScaleCrossAttention(hidden_size, num_heads, num_layers),
            HierarchicalCrossAttention(hidden_size, num_heads, num_layers)
        ])
        
        self.attention_selector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(self.attention_types)),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        response_hidden: torch.Tensor,
        query_hidden: torch.Tensor,
        response_mask: torch.Tensor,
        query_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        response_pooled = response_hidden.mean(dim=1)
        query_pooled = query_hidden.mean(dim=1)
        
        combined_context = torch.cat([response_pooled, query_pooled], dim=-1)
        attention_weights = self.attention_selector(combined_context)
        
        outputs = []
        all_weights = []
        
        for attention_module in self.attention_types:
            output, weights = attention_module(
                response_hidden, query_hidden, 
                response_mask, query_mask
            )
            outputs.append(output)
            all_weights.append(weights)
        
        weighted_output = torch.zeros_like(outputs[0])
        for i, (output, weight) in enumerate(zip(outputs, attention_weights.unbind(1))):
            weighted_output += weight.view(-1, 1, 1) * output
        
        return weighted_output, all_weights[0]