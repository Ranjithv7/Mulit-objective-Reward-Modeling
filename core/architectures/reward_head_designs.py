import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math

class RewardHeadFactory:
    @staticmethod
    def create_head(
        head_type: str,
        hidden_size: int,
        num_objectives: int = 1,
        **kwargs
    ) -> nn.Module:
        head_map = {
            "linear": LinearRewardHead,
            "mlp": MLPRewardHead,
            "residual": ResidualRewardHead,
            "attention": AttentionRewardHead,
            "mixture_experts": MixtureOfExpertsHead,
            "capsule": CapsuleRewardHead,
            "graph": GraphRewardHead,
            "transformer": TransformerRewardHead,
            "multi_scale": MultiScaleRewardHead,
            "adaptive": AdaptiveRewardHead
        }
        
        if head_type not in head_map:
            raise ValueError(f"Unknown head_type: {head_type}")
        
        return head_map[head_type](hidden_size, num_objectives, **kwargs)

class LinearRewardHead(nn.Module):
    def __init__(self, hidden_size: int, num_objectives: int = 1, **kwargs):
        super().__init__()
        self.head = nn.Linear(hidden_size, num_objectives)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class MLPRewardHead(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_objectives: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        **kwargs
    ):
        super().__init__()
        
        activation_fn = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "swish": nn.SiLU,
            "tanh": nn.Tanh
        }[activation]
        
        layers = []
        current_size = hidden_size
        
        for i in range(num_layers - 1):
            next_size = hidden_size // (2 ** (i + 1))
            layers.extend([
                nn.Linear(current_size, next_size),
                activation_fn(),
                nn.Dropout(dropout)
            ])
            current_size = next_size
        
        layers.append(nn.Linear(current_size, num_objectives))
        self.head = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class ResidualRewardHead(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_objectives: int = 1,
        num_blocks: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout) 
            for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_size, num_objectives)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        
        x = self.layer_norm(x)
        return self.output_layer(x)

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.layer_norm(x + residual)

class AttentionRewardHead(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_objectives: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        self.output_layer = nn.Linear(hidden_size, num_objectives)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        attended, _ = self.self_attention(x, x, x)
        x = self.layer_norm(x + attended)
        
        ff_output = self.feedforward(x)
        x = self.layer_norm(x + ff_output)
        
        x = x.squeeze(1) if x.size(1) == 1 else x.mean(dim=1)
        return self.output_layer(x)

class MixtureOfExpertsHead(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_objectives: int = 1,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.experts = nn.ModuleList([
            MLPRewardHead(hidden_size, num_objectives, dropout=dropout)
            for _ in range(num_experts)
        ])
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_experts)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        batch_size = x.size(0)
        selected_outputs = expert_outputs[
            torch.arange(batch_size).unsqueeze(1),
            top_k_indices
        ]
        
        weighted_output = torch.sum(
            selected_outputs * top_k_probs.unsqueeze(-1),
            dim=1
        )
        
        return weighted_output

class CapsuleRewardHead(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_objectives: int = 1,
        num_capsules: int = 8,
        capsule_dim: int = 16,
        num_routing: int = 3,
        **kwargs
    ):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.num_routing = num_routing
        
        self.primary_capsules = nn.Linear(hidden_size, num_capsules * capsule_dim)
        self.output_capsules = nn.Parameter(
            torch.randn(num_objectives, num_capsules, capsule_dim, capsule_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        primary_caps = self.primary_capsules(x)
        primary_caps = primary_caps.view(batch_size, self.num_capsules, self.capsule_dim)
        
        u_hat = torch.einsum('bic,oicj->boij', primary_caps, self.output_capsules)
        u_hat = u_hat.view(batch_size, -1, self.capsule_dim)
        
        b = torch.zeros(batch_size, u_hat.size(1), 1, device=x.device)
        
        for _ in range(self.num_routing):
            c = F.softmax(b, dim=1)
            s = torch.sum(c * u_hat, dim=1)
            v = self.squash(s)
            
            if _ < self.num_routing - 1:
                agreement = torch.sum(u_hat * v.unsqueeze(1), dim=-1, keepdim=True)
                b = b + agreement
        
        return torch.norm(v, dim=-1)
    
    def squash(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True)
        return (norm**2 / (1 + norm**2)) * (x / (norm + 1e-8))

class GraphRewardHead(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_objectives: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.graph_layers = nn.ModuleList([
            GraphConvLayer(hidden_size, hidden_size, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_size, num_objectives)
        self.global_pool = GlobalAttentionPool(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        adj_matrix = self._create_adjacency_matrix(x.size(1), x.device)
        
        for layer in self.graph_layers:
            x = layer(x, adj_matrix)
        
        x = self.global_pool(x)
        return self.output_layer(x)
    
    def _create_adjacency_matrix(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        adj = torch.eye(num_nodes, device=device)
        
        for i in range(num_nodes - 1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1
        
        return adj

class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.matmul(adj, x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.layer_norm(x)

class GlobalAttentionPool(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(x * weights, dim=1)

class TransformerRewardHead(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_objectives: int = 1,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(hidden_size, num_objectives)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.transformer(x)
        cls_output = x[:, 0]
        
        return self.output_layer(cls_output)

class MultiScaleRewardHead(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_objectives: int = 1,
        scales: List[int] = [1, 3, 5],
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.scales = scales
        
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size // len(scales), kernel_size=scale, padding=scale//2)
            for scale in scales
        ])
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_objectives)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = x.transpose(1, 2)
        
        scale_outputs = []
        for conv in self.scale_convs:
            scale_out = F.relu(conv(x))
            scale_outputs.append(scale_out.mean(dim=-1))
        
        fused = torch.cat(scale_outputs, dim=1)
        return self.fusion_layer(fused)

class AdaptiveRewardHead(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_objectives: int = 1,
        num_head_types: int = 3,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.head_types = nn.ModuleList([
            MLPRewardHead(hidden_size, num_objectives, dropout=dropout),
            AttentionRewardHead(hidden_size, num_objectives, dropout=dropout),
            ResidualRewardHead(hidden_size, num_objectives, dropout=dropout)
        ])
        
        self.head_selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, len(self.head_types)),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_weights = self.head_selector(x)
        
        head_outputs = torch.stack([
            head(x) for head in self.head_types
        ], dim=1)
        
        weighted_output = torch.sum(
            head_outputs * head_weights.unsqueeze(-1),
            dim=1
        )
        
        return weighted_output

class DynamicRewardHead(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_objectives: int = 1,
        max_layers: int = 5,
        layer_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.max_layers = max_layers
        self.layer_threshold = layer_threshold
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(max_layers)
        ])
        
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            ) for _ in range(max_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_size, num_objectives)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (layer, conf_head) in enumerate(zip(self.layers, self.confidence_heads)):
            x = layer(x)
            confidence = conf_head(x)
            
            if confidence.mean() > self.layer_threshold or i == self.max_layers - 1:
                break
        
        return self.output_layer(x)