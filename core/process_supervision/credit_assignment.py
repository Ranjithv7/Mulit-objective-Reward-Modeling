import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np

class TemporalCreditAssigner:
    def __init__(
        self,
        assignment_method: str = "exponential_decay",
        decay_factor: float = 0.9,
        eligibility_trace_lambda: float = 0.8,
        causal_window: int = 5
    ):
        self.assignment_method = assignment_method
        self.decay_factor = decay_factor
        self.eligibility_trace_lambda = eligibility_trace_lambda
        self.causal_window = causal_window
    
    def assign_credit(
        self,
        step_sequence: torch.Tensor,
        final_outcomes: torch.Tensor,
        step_contributions: torch.Tensor,
        step_dependencies: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.assignment_method == "exponential_decay":
            return self._exponential_decay_assignment(step_sequence, final_outcomes)
        elif self.assignment_method == "eligibility_traces":
            return self._eligibility_trace_assignment(step_sequence, final_outcomes, step_contributions)
        elif self.assignment_method == "causal_influence":
            return self._causal_influence_assignment(step_sequence, final_outcomes, step_dependencies)
        elif self.assignment_method == "attention_based":
            return self._attention_based_assignment(step_sequence, final_outcomes)
        elif self.assignment_method == "shapley_value":
            return self._shapley_value_assignment(step_sequence, final_outcomes, step_contributions)
        else:
            raise ValueError(f"Unknown assignment method: {self.assignment_method}")
    
    def _exponential_decay_assignment(
        self,
        step_sequence: torch.Tensor,
        final_outcomes: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps, feature_dim = step_sequence.shape
        credit_assignments = torch.zeros(batch_size, num_steps, device=step_sequence.device)
        
        for b in range(batch_size):
            outcome = final_outcomes[b]
            
            # Assign credit with exponential decay (recent steps get more credit)
            for t in range(num_steps):
                steps_from_end = num_steps - t - 1
                decay_weight = self.decay_factor ** steps_from_end
                credit_assignments[b, t] = outcome * decay_weight
            
            # Normalize to sum to final outcome
            if credit_assignments[b].sum() > 0:
                credit_assignments[b] = credit_assignments[b] * (outcome / credit_assignments[b].sum())
        
        return credit_assignments
    
    def _eligibility_trace_assignment(
        self,
        step_sequence: torch.Tensor,
        final_outcomes: torch.Tensor,
        step_contributions: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps, _ = step_sequence.shape
        credit_assignments = torch.zeros(batch_size, num_steps, device=step_sequence.device)
        
        for b in range(batch_size):
            eligibility_traces = torch.zeros(num_steps, device=step_sequence.device)
            
            for t in range(num_steps):
                # Update eligibility traces
                eligibility_traces *= self.decay_factor * self.eligibility_trace_lambda
                eligibility_traces[t] = step_contributions[b, t]
                
                # Assign credit based on eligibility
                if t == num_steps - 1:  # Final step
                    credit_assignments[b] = final_outcomes[b] * eligibility_traces
        
        return credit_assignments
    
    def _causal_influence_assignment(
        self,
        step_sequence: torch.Tensor,
        final_outcomes: torch.Tensor,
        step_dependencies: Optional[torch.Tensor]
    ) -> torch.Tensor:
        batch_size, num_steps, _ = step_sequence.shape
        credit_assignments = torch.zeros(batch_size, num_steps, device=step_sequence.device)
        
        if step_dependencies is None:
            # Create default causal dependencies (each step depends on previous ones)
            step_dependencies = torch.zeros(batch_size, num_steps, num_steps, device=step_sequence.device)
            for b in range(batch_size):
                for i in range(num_steps):
                    for j in range(max(0, i - self.causal_window), i):
                        step_dependencies[b, i, j] = 1.0 / (i - j)
        
        for b in range(batch_size):
            # Compute causal influence matrix
            causal_matrix = self._compute_causal_matrix(step_dependencies[b])
            
            # Assign credit based on causal influence
            outcome = final_outcomes[b]
            for t in range(num_steps):
                # Credit is proportional to causal influence on final outcome
                causal_influence = causal_matrix[num_steps - 1, t]  # Influence on final step
                credit_assignments[b, t] = outcome * causal_influence
        
        return credit_assignments
    
    def _attention_based_assignment(
        self,
        step_sequence: torch.Tensor,
        final_outcomes: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps, feature_dim = step_sequence.shape
        
        # Use self-attention to determine step importance
        attention_module = nn.MultiheadAttention(
            feature_dim, num_heads=8, batch_first=True
        ).to(step_sequence.device)
        
        with torch.no_grad():
            attended_sequence, attention_weights = attention_module(
                step_sequence, step_sequence, step_sequence
            )
            
            # Use attention weights from final step to assign credit
            final_step_attention = attention_weights[:, -1, :]  # Shape: (batch_size, num_steps)
            
            # Normalize and scale by final outcome
            credit_assignments = torch.zeros(batch_size, num_steps, device=step_sequence.device)
            for b in range(batch_size):
                attention_sum = final_step_attention[b].sum()
                if attention_sum > 0:
                    normalized_attention = final_step_attention[b] / attention_sum
                    credit_assignments[b] = final_outcomes[b] * normalized_attention
        
        return credit_assignments
    
    def _shapley_value_assignment(
        self,
        step_sequence: torch.Tensor,
        final_outcomes: torch.Tensor,
        step_contributions: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps, _ = step_sequence.shape
        credit_assignments = torch.zeros(batch_size, num_steps, device=step_sequence.device)
        
        for b in range(batch_size):
            # Approximate Shapley values through sampling
            shapley_values = self._approximate_shapley_values(
                step_contributions[b],
                final_outcomes[b],
                num_samples=100
            )
            credit_assignments[b] = shapley_values
        
        return credit_assignments
    
    def _compute_causal_matrix(self, dependencies: torch.Tensor) -> torch.Tensor:
        num_steps = dependencies.size(0)
        causal_matrix = torch.eye(num_steps, device=dependencies.device)
        
        # Compute transitive closure of causal relationships
        for k in range(num_steps):
            for i in range(num_steps):
                for j in range(num_steps):
                    causal_matrix[i, j] = max(
                        causal_matrix[i, j],
                        min(causal_matrix[i, k], causal_matrix[k, j])
                    )
        
        return causal_matrix
    
    def _approximate_shapley_values(
        self,
        step_contributions: torch.Tensor,
        final_outcome: torch.Tensor,
        num_samples: int = 100
    ) -> torch.Tensor:
        num_steps = step_contributions.size(0)
        shapley_values = torch.zeros(num_steps, device=step_contributions.device)
        
        for _ in range(num_samples):
            # Random permutation of steps
            permutation = torch.randperm(num_steps)
            
            for i in range(num_steps):
                step_idx = permutation[i]
                
                # Contribution when adding this step
                coalition_before = permutation[:i]
                coalition_after = permutation[:i+1]
                
                value_before = self._coalition_value(coalition_before, step_contributions, final_outcome)
                value_after = self._coalition_value(coalition_after, step_contributions, final_outcome)
                
                marginal_contribution = value_after - value_before
                shapley_values[step_idx] += marginal_contribution
        
        return shapley_values / num_samples
    
    def _coalition_value(
        self,
        coalition: torch.Tensor,
        step_contributions: torch.Tensor,
        final_outcome: torch.Tensor
    ) -> torch.Tensor:
        if len(coalition) == 0:
            return torch.tensor(0.0, device=step_contributions.device)
        
        # Simple coalition value: sum of contributions
        return step_contributions[coalition].sum()

class LearnedCreditAssigner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        assignment_layers: int = 3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Multi-head attention for step interactions
        self.step_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
            for _ in range(assignment_layers)
        ])
        
        # Credit prediction networks
        self.credit_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Outcome encoder
        self.outcome_encoder = nn.Linear(1, hidden_size)
    
    def forward(
        self,
        step_sequence: torch.Tensor,
        final_outcomes: torch.Tensor,
        step_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps, hidden_size = step_sequence.shape
        
        # Encode final outcome
        outcome_encoding = self.outcome_encoder(final_outcomes.unsqueeze(-1))
        outcome_encoding = outcome_encoding.unsqueeze(1).expand(-1, num_steps, -1)
        
        # Apply attention layers to capture step interactions
        attended_sequence = step_sequence
        for attention_layer in self.step_attention:
            attended_sequence, _ = attention_layer(
                attended_sequence, attended_sequence, attended_sequence,
                key_padding_mask=~step_mask.bool()
            )
        
        # Combine step features with outcome encoding
        combined_features = torch.cat([attended_sequence, outcome_encoding], dim=-1)
        
        # Predict credit for each step
        credit_scores = self.credit_predictor(combined_features).squeeze(-1)
        
        # Apply mask and normalize
        credit_scores = credit_scores * step_mask
        
        # Normalize to sum to final outcome
        total_credit = credit_scores.sum(dim=1, keepdim=True)
        normalized_credit = credit_scores * (final_outcomes.unsqueeze(1) / (total_credit + 1e-8))
        
        return normalized_credit

class CausalCreditAssigner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        causal_discovery_method: str = "attention",
        intervention_strength: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.causal_discovery_method = causal_discovery_method
        self.intervention_strength = intervention_strength
        
        if causal_discovery_method == "attention":
            self.causal_attention = CausalAttentionModule(hidden_size)
        elif causal_discovery_method == "gnn":
            self.causal_gnn = CausalGraphNetwork(hidden_size)
        
        self.intervention_network = InterventionNetwork(hidden_size)
        self.credit_estimator = CreditEstimationNetwork(hidden_size)
    
    def forward(
        self,
        step_sequence: torch.Tensor,
        final_outcomes: torch.Tensor,
        step_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Discover causal relationships
        if self.causal_discovery_method == "attention":
            causal_matrix = self.causal_attention(step_sequence, step_mask)
        elif self.causal_discovery_method == "gnn":
            causal_matrix = self.causal_gnn(step_sequence, step_mask)
        
        # Estimate credit through interventions
        credit_assignments = self._compute_interventional_credit(
            step_sequence, final_outcomes, causal_matrix, step_mask
        )
        
        return credit_assignments, causal_matrix
    
    def _compute_interventional_credit(
        self,
        step_sequence: torch.Tensor,
        final_outcomes: torch.Tensor,
        causal_matrix: torch.Tensor,
        step_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps, _ = step_sequence.shape
        credit_assignments = torch.zeros(batch_size, num_steps, device=step_sequence.device)
        
        for b in range(batch_size):
            for t in range(num_steps):
                if step_mask[b, t]:
                    # Create intervention by modifying step t
                    intervened_sequence = step_sequence[b:b+1].clone()
                    intervention = self.intervention_network(intervened_sequence[:, t:t+1])
                    intervened_sequence[:, t] = intervention.squeeze(1)
                    
                    # Predict counterfactual outcome
                    counterfactual_outcome = self._predict_outcome(
                        intervened_sequence, causal_matrix[b:b+1]
                    )
                    
                    # Credit is difference between actual and counterfactual
                    credit_assignments[b, t] = final_outcomes[b] - counterfactual_outcome
        
        return credit_assignments
    
    def _predict_outcome(
        self,
        sequence: torch.Tensor,
        causal_matrix: torch.Tensor
    ) -> torch.Tensor:
        # Simple outcome prediction based on causal relationships
        weighted_sequence = sequence * causal_matrix.unsqueeze(-1)
        return self.credit_estimator(weighted_sequence.mean(dim=1)).squeeze(-1)

class CausalAttentionModule(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, batch_first=True
        )
        
        self.causal_mask_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, step_sequence: torch.Tensor, step_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_steps, _ = step_sequence.shape
        
        # Compute attention weights
        _, attention_weights = self.attention(
            step_sequence, step_sequence, step_sequence,
            key_padding_mask=~step_mask.bool()
        )
        
        # Predict causal relationships
        causal_strengths = self.causal_mask_predictor(step_sequence).squeeze(-1)
        
        # Combine attention with causal prediction
        causal_matrix = attention_weights.mean(dim=1) * causal_strengths.unsqueeze(-1)
        
        # Enforce causal ordering (no backward causation)
        causal_mask = torch.tril(torch.ones(num_steps, num_steps, device=step_sequence.device))
        causal_matrix = causal_matrix * causal_mask.unsqueeze(0)
        
        return causal_matrix

class CausalGraphNetwork(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.graph_conv = nn.Sequential(
            GraphConvLayer(hidden_size, hidden_size),
            nn.ReLU(),
            GraphConvLayer(hidden_size, hidden_size)
        )
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, step_sequence: torch.Tensor, step_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_steps, _ = step_sequence.shape
        
        # Apply graph convolution
        graph_features = self.graph_conv(step_sequence)
        
        # Predict edges between all step pairs
        causal_matrix = torch.zeros(batch_size, num_steps, num_steps, device=step_sequence.device)
        
        for i in range(num_steps):
            for j in range(num_steps):
                if i != j:
                    edge_input = torch.cat([graph_features[:, i], graph_features[:, j]], dim=-1)
                    edge_strength = self.edge_predictor(edge_input).squeeze(-1)
                    causal_matrix[:, i, j] = edge_strength
        
        # Apply causal ordering constraint
        causal_mask = torch.tril(torch.ones(num_steps, num_steps, device=step_sequence.device), diagonal=-1)
        causal_matrix = causal_matrix * causal_mask.unsqueeze(0)
        
        return causal_matrix

class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple graph convolution (all-to-all connectivity)
        return self.activation(self.linear(x))

class InterventionNetwork(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.intervention_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
    
    def forward(self, step_features: torch.Tensor) -> torch.Tensor:
        # Generate intervention that modifies the step
        intervention = self.intervention_generator(step_features)
        return step_features + 0.1 * intervention  # Small perturbation

class CreditEstimationNetwork(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.estimator(features)

class TemporalCreditTrainer:
    def __init__(
        self,
        credit_assigner: nn.Module,
        learning_rate: float = 1e-4,
        credit_consistency_weight: float = 1.0,
        causal_regularization_weight: float = 0.1
    ):
        self.credit_assigner = credit_assigner
        self.credit_consistency_weight = credit_consistency_weight
        self.causal_regularization_weight = causal_regularization_weight
        
        self.optimizer = torch.optim.AdamW(
            credit_assigner.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        self.credit_assigner.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        if isinstance(self.credit_assigner, CausalCreditAssigner):
            credit_assignments, causal_matrix = self.credit_assigner(
                batch["step_sequence"],
                batch["final_outcomes"],
                batch["step_mask"]
            )
        else:
            credit_assignments = self.credit_assigner(
                batch["step_sequence"],
                batch["final_outcomes"],
                batch["step_mask"]
            )
            causal_matrix = None
        
        # Credit consistency loss
        predicted_total = credit_assignments.sum(dim=1)
        consistency_loss = F.mse_loss(predicted_total, batch["final_outcomes"])
        
        total_loss = self.credit_consistency_weight * consistency_loss
        
        # Causal regularization
        if causal_matrix is not None:
            # Encourage sparsity in causal relationships
            causal_sparsity_loss = torch.norm(causal_matrix, p=1) / causal_matrix.numel()
            total_loss += self.causal_regularization_weight * causal_sparsity_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.credit_assigner.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "causal_sparsity": causal_sparsity_loss.item() if causal_matrix is not None else 0.0
        }

class CreditAssignmentEvaluator:
    def __init__(self):
        pass
    
    def evaluate_assignment_quality(
        self,
        predicted_credits: torch.Tensor,
        ground_truth_credits: torch.Tensor,
        step_mask: torch.Tensor
    ) -> Dict[str, float]:
        # Correlation with ground truth
        valid_predictions = predicted_credits[step_mask.bool()]
        valid_ground_truth = ground_truth_credits[step_mask.bool()]
        
        correlation = torch.corrcoef(torch.stack([valid_predictions, valid_ground_truth]))[0, 1]
        
        # Mean absolute error
        mae = F.l1_loss(valid_predictions, valid_ground_truth)
        
        # Ranking correlation (Spearman)
        spearman_corr = self._compute_spearman_correlation(valid_predictions, valid_ground_truth)
        
        return {
            "pearson_correlation": correlation.item() if not torch.isnan(correlation) else 0.0,
            "mean_absolute_error": mae.item(),
            "spearman_correlation": spearman_corr
        }
    
    def _compute_spearman_correlation(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        def rank_tensor(x):
            return torch.argsort(torch.argsort(x)).float()
        
        pred_ranks = rank_tensor(pred)
        target_ranks = rank_tensor(target)
        
        correlation = torch.corrcoef(torch.stack([pred_ranks, target_ranks]))[0, 1]
        return correlation.item() if not torch.isnan(correlation) else 0.0