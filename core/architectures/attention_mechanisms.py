import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math

class RewardAttentionFactory:
    @staticmethod
    def create_attention(
        attention_type: str,
        hidden_size: int,
        **kwargs
    ) -> nn.Module:
        attention_map = {
            "sparse": SparseAttention,
            "local": LocalAttention,
            "rotary": RotaryAttention,
            "relative": RelativePositionAttention,
            "multi_scale": MultiScaleAttention,
            "adaptive": AdaptiveAttention,
            "reward_aware": RewardAwareAttention,
            "contrastive": ContrastiveAttention,
            "temporal": TemporalAttention,
            "hierarchical": HierarchicalAttention
        }
        
        if attention_type not in attention_map:
            raise ValueError(f"Unknown attention_type: {attention_type}")
        
        return attention_map[attention_type](hidden_size, **kwargs)

class SparseAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        sparsity_factor: int = 4,
        top_k: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.sparsity_factor = sparsity_factor
        self.top_k = top_k
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.shape
        
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        sparse_mask = self._create_sparse_mask(seq_len, q.device)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(1), -1e9)
        
        scores = scores.masked_fill(~sparse_mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        if self.top_k is not None:
            top_k = min(self.top_k, seq_len)
            top_values, top_indices = torch.topk(scores, top_k, dim=-1)
            sparse_scores = torch.full_like(scores, -1e9)
            sparse_scores.scatter_(-1, top_indices, top_values)
            scores = sparse_scores
        
        attention_weights = F.softmax(scores, dim=-1)
        
        attended = torch.matmul(attention_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        output = self.out_proj(attended)
        
        return output, attention_weights.mean(dim=1)
    
    def _create_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        
        for i in range(seq_len):
            start = max(0, i - self.sparsity_factor)
            end = min(seq_len, i + self.sparsity_factor + 1)
            mask[i, start:end] = True
            
            if i % self.sparsity_factor == 0:
                mask[i, :] = True
                mask[:, i] = True
        
        return mask

class LocalAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        window_size: int = 32,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.shape
        
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attended = torch.zeros_like(q)
        attention_weights = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=q.device)
        
        for i in range(0, seq_len, self.window_size):
            end_idx = min(i + self.window_size, seq_len)
            
            q_window = q[:, :, i:end_idx, :]
            k_window = k[:, :, i:end_idx, :]
            v_window = v[:, :, i:end_idx, :]
            
            scores = torch.matmul(q_window, k_window.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                window_mask = attention_mask[:, i:end_idx]
                scores = scores.masked_fill(~window_mask.unsqueeze(1).unsqueeze(1), -1e9)
            
            weights = F.softmax(scores, dim=-1)
            window_attended = torch.matmul(weights, v_window)
            
            attended[:, :, i:end_idx, :] = window_attended
            attention_weights[:, :, i:end_idx, i:end_idx] = weights
        
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attended)
        
        return output, attention_weights.mean(dim=1)

class RotaryAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        max_position: int = 2048,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_position = max_position
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.scale = self.head_dim ** -0.5
        
        self.register_buffer('rotary_emb', self._create_rotary_embedding())
    
    def _create_rotary_embedding(self) -> torch.Tensor:
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        position = torch.arange(self.max_position).float()
        sinusoid_inp = torch.outer(position, inv_freq)
        
        emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return emb.unsqueeze(0).unsqueeze(0)
    
    def _apply_rotary_pos_emb(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(2)
        pos_emb = pos_emb[:, :, :seq_len, :].to(x.device)
        
        x_rot = x * pos_emb[:, :, :, :self.head_dim]
        x_pass = x * pos_emb[:, :, :, self.head_dim:]
        
        return x_rot + x_pass
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.shape
        
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        q = self._apply_rotary_pos_emb(q, self.rotary_emb)
        k = self._apply_rotary_pos_emb(k, self.rotary_emb)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(1), -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, v)
        
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attended)
        
        return output, attention_weights.mean(dim=1)

class RelativePositionAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        max_relative_position: int = 32,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_relative_position = max_relative_position
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.relative_k = nn.Embedding(2 * max_relative_position + 1, self.head_dim)
        self.relative_v = nn.Embedding(2 * max_relative_position + 1, self.head_dim)
        
        self.scale = self.head_dim ** -0.5
    
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
        positions = torch.clamp(positions, -self.max_relative_position, self.max_relative_position)
        return positions + self.max_relative_position
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.shape
        
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        relative_positions = self._get_relative_positions(seq_len).to(query.device)
        
        relative_k_emb = self.relative_k(relative_positions)
        relative_v_emb = self.relative_v(relative_positions)
        
        content_scores = torch.matmul(q, k.transpose(-2, -1))
        relative_scores = torch.matmul(
            q.unsqueeze(-2),
            relative_k_emb.transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        ).squeeze(-2)
        
        scores = (content_scores + relative_scores) * self.scale
        
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(1), -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        
        content_attended = torch.matmul(attention_weights, v)
        relative_attended = torch.matmul(
            attention_weights.unsqueeze(-1),
            relative_v_emb.unsqueeze(0).unsqueeze(0)
        ).squeeze(-2)
        
        attended = content_attended + relative_attended
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        output = self.out_proj(attended)
        
        return output, attention_weights.mean(dim=1)

class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        scales: List[int] = [1, 2, 4, 8],
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.scales = scales
        self.head_dim = hidden_size // num_heads
        
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_size // len(scales),
                max(1, num_heads // len(scales)),
                batch_first=True
            ) for _ in scales
        ])
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        self.scale_fusion = nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.shape
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q_splits = torch.split(q, self.hidden_size // len(self.scales), dim=-1)
        k_splits = torch.split(k, self.hidden_size // len(self.scales), dim=-1)
        v_splits = torch.split(v, self.hidden_size // len(self.scales), dim=-1)
        
        scale_outputs = []
        attention_weights_list = []
        
        for i, (scale, attention) in enumerate(zip(self.scales, self.scale_attentions)):
            if scale > 1:
                downsampled_k = F.avg_pool1d(
                    k_splits[i].transpose(1, 2),
                    kernel_size=scale, stride=scale
                ).transpose(1, 2)
                downsampled_v = F.avg_pool1d(
                    v_splits[i].transpose(1, 2),
                    kernel_size=scale, stride=scale
                ).transpose(1, 2)
                
                if attention_mask is not None:
                    downsampled_mask = F.max_pool1d(
                        attention_mask.float().unsqueeze(1),
                        kernel_size=scale, stride=scale
                    ).squeeze(1).bool()
                else:
                    downsampled_mask = None
            else:
                downsampled_k = k_splits[i]
                downsampled_v = v_splits[i]
                downsampled_mask = attention_mask
            
            attended, weights = attention(
                q_splits[i], downsampled_k, downsampled_v,
                key_padding_mask=~downsampled_mask if downsampled_mask is not None else None
            )
            
            if scale > 1:
                attended = F.interpolate(
                    attended.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            scale_outputs.append(attended)
            attention_weights_list.append(weights)
        
        fused_output = self.scale_fusion(torch.cat(scale_outputs, dim=-1))
        
        return fused_output, attention_weights_list[0]

class AdaptiveAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_attention_types: int = 3,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.attention_types = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads, batch_first=True),
            SparseAttention(hidden_size, num_heads),
            LocalAttention(hidden_size, num_heads)
        ])
        
        self.attention_selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, len(self.attention_types)),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context = query.mean(dim=1)
        selection_weights = self.attention_selector(context)
        
        outputs = []
        attention_weights_list = []
        
        for attention_module in self.attention_types:
            if isinstance(attention_module, nn.MultiheadAttention):
                attended, weights = attention_module(
                    query, key, value,
                    key_padding_mask=~attention_mask if attention_mask is not None else None
                )
            else:
                attended, weights = attention_module(query, key, value, attention_mask)
            
            outputs.append(attended)
            attention_weights_list.append(weights)
        
        weighted_output = torch.zeros_like(outputs[0])
        for i, (output, weight) in enumerate(zip(outputs, selection_weights.unbind(1))):
            weighted_output += weight.view(-1, 1, 1) * output
        
        return weighted_output, attention_weights_list[0]

class RewardAwareAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        reward_dim: int = 1,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.reward_dim = reward_dim
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.reward_proj = nn.Linear(reward_dim, hidden_size)
        self.reward_attention = nn.Linear(hidden_size, num_heads)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rewards: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.shape
        
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        reward_features = self.reward_proj(rewards)
        reward_bias = self.reward_attention(reward_features).unsqueeze(-1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + reward_bias
        
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(1), -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, v)
        
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attended)
        
        return output, attention_weights.mean(dim=1)

class ContrastiveAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        temperature: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.temperature = temperature
        
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        
        self.positive_proj = nn.Linear(hidden_size, hidden_size)
        self.negative_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positive_examples: torch.Tensor,
        negative_examples: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attended, attention_weights = self.attention(
            query, key, value,
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )
        
        query_pooled = query.mean(dim=1)
        positive_features = self.positive_proj(positive_examples.mean(dim=1))
        negative_features = self.negative_proj(negative_examples.mean(dim=1))
        
        positive_sim = F.cosine_similarity(query_pooled, positive_features, dim=-1) / self.temperature
        negative_sim = F.cosine_similarity(query_pooled, negative_features, dim=-1) / self.temperature
        
        contrastive_weights = F.softmax(torch.stack([positive_sim, negative_sim], dim=-1), dim=-1)
        contrastive_bias = contrastive_weights[:, 0].unsqueeze(-1).unsqueeze(-1)
        
        enhanced_attended = attended * (1 + contrastive_bias)
        
        return enhanced_attended, attention_weights

class TemporalAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        decay_factor: float = 0.9,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.decay_factor = decay_factor
        
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        
        self.temporal_proj = nn.Linear(hidden_size, num_heads)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.shape
        
        temporal_decay = torch.pow(
            self.decay_factor,
            torch.arange(seq_len, dtype=torch.float, device=query.device)
        ).flip(0)
        
        temporal_weights = self.temporal_proj(query)
        temporal_weights = temporal_weights * temporal_decay.unsqueeze(0).unsqueeze(-1)
        
        attended, attention_weights = self.attention(
            query, key, value,
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )
        
        temporal_attended = attended * F.sigmoid(temporal_weights).unsqueeze(-1)
        
        return temporal_attended, attention_weights

class HierarchicalAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_levels: int = 3,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_levels = num_levels
        
        self.level_attentions = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
            for _ in range(num_levels)
        ])
        
        self.level_fusion = nn.Linear(hidden_size * num_levels, hidden_size)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        level_outputs = []
        level_weights = []
        
        for level, attention in enumerate(self.level_attentions):
            level_query = self._create_level_features(query, level)
            level_key = self._create_level_features(key, level)
            level_value = self._create_level_features(value, level)
            
            attended, weights = attention(
                level_query, level_key, level_value,
                key_padding_mask=~attention_mask if attention_mask is not None else None
            )
            
            level_outputs.append(attended)
            level_weights.append(weights)
        
        fused_output = self.level_fusion(torch.cat(level_outputs, dim=-1))
        
        return fused_output, level_weights[0]
    
    def _create_level_features(self, x: torch.Tensor, level: int) -> torch.Tensor:
        if level == 0:
            return x
        
        pooling_size = 2 ** level
        return F.avg_pool1d(
            x.transpose(1, 2),
            kernel_size=pooling_size,
            stride=pooling_size,
            padding=0
        ).transpose(1, 2)