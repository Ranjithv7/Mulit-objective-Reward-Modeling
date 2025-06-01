import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
import math
import itertools

class CounterfactualReasoningModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intervention_types: List[str] = ["removal", "replacement", "addition"],
        counterfactual_depth: int = 3,
        causal_strength_threshold: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intervention_types = intervention_types
        self.counterfactual_depth = counterfactual_depth
        self.causal_strength_threshold = causal_strength_threshold
        
        # Intervention generators
        self.intervention_generators = nn.ModuleDict({
            intervention_type: InterventionGenerator(hidden_size, intervention_type)
            for intervention_type in intervention_types
        })
        
        # Outcome predictor for counterfactual scenarios
        self.counterfactual_predictor = CounterfactualOutcomePredictor(hidden_size)
        
        # Causal importance estimator
        self.causal_importance_estimator = CausalImportanceEstimator(hidden_size)
        
        # Necessity and sufficiency analyzers
        self.necessity_analyzer = NecessityAnalyzer(hidden_size)
        self.sufficiency_analyzer = SufficiencyAnalyzer(hidden_size)
    
    def forward(
        self,
        step_sequence: torch.Tensor,
        step_mask: torch.Tensor,
        target_outcome: torch.Tensor,
        intervention_targets: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_steps, _ = step_sequence.shape
        
        # Analyze causal importance of each step
        causal_importance = self.causal_importance_estimator(step_sequence, step_mask, target_outcome)
        
        # Determine intervention targets if not provided
        if intervention_targets is None:
            intervention_targets = self._select_intervention_targets(causal_importance, step_mask)
        
        # Generate counterfactuals for each intervention type
        counterfactual_results = {}
        
        for intervention_type in self.intervention_types:
            counterfactuals = self._generate_counterfactuals(
                step_sequence, intervention_targets, intervention_type, step_mask
            )
            
            # Predict outcomes for counterfactual scenarios
            counterfactual_outcomes = self.counterfactual_predictor(counterfactuals, step_mask)
            
            counterfactual_results[intervention_type] = {
                "sequences": counterfactuals,
                "outcomes": counterfactual_outcomes,
                "causal_effects": target_outcome.unsqueeze(1) - counterfactual_outcomes
            }
        
        # Analyze necessity and sufficiency
        necessity_scores = self.necessity_analyzer(
            step_sequence, counterfactual_results["removal"]["outcomes"], target_outcome, intervention_targets
        )
        
        sufficiency_scores = self.sufficiency_analyzer(
            step_sequence, counterfactual_results["addition"]["outcomes"], target_outcome, intervention_targets
        )
        
        return {
            "causal_importance": causal_importance,
            "counterfactual_results": counterfactual_results,
            "necessity_scores": necessity_scores,
            "sufficiency_scores": sufficiency_scores,
            "intervention_targets": intervention_targets
        }
    
    def _select_intervention_targets(
        self,
        causal_importance: torch.Tensor,
        step_mask: torch.Tensor
    ) -> List[int]:
        # Select steps with highest causal importance for intervention
        valid_importance = causal_importance * step_mask
        top_k = min(3, step_mask.sum(dim=1).max().item())
        
        _, top_indices = torch.topk(valid_importance, top_k, dim=1)
        return top_indices[0].tolist()  # Use first batch for simplicity
    
    def _generate_counterfactuals(
        self,
        step_sequence: torch.Tensor,
        intervention_targets: List[int],
        intervention_type: str,
        step_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps, hidden_size = step_sequence.shape
        
        # Generate interventions
        interventions = self.intervention_generators[intervention_type](
            step_sequence, intervention_targets, step_mask
        )
        
        # Apply interventions to create counterfactual sequences
        counterfactual_sequences = step_sequence.clone()
        
        for target_idx in intervention_targets:
            if target_idx < num_steps:
                counterfactual_sequences[:, target_idx] = interventions[:, target_idx]
        
        return counterfactual_sequences

class InterventionGenerator(nn.Module):
    def __init__(self, hidden_size: int, intervention_type: str):
        super().__init__()
        self.intervention_type = intervention_type
        self.hidden_size = hidden_size
        
        if intervention_type == "removal":
            # Zero out the step
            self.generator = lambda x, targets, mask: torch.zeros_like(x)
        elif intervention_type == "replacement":
            # Replace with learned alternative
            self.generator = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh()
            )
        elif intervention_type == "addition":
            # Add learned perturbation
            self.generator = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size),
                nn.Tanh()
            )
        elif intervention_type == "noise":
            # Add Gaussian noise
            self.noise_std = nn.Parameter(torch.tensor(0.1))
        else:
            raise ValueError(f"Unknown intervention type: {intervention_type}")
    
    def forward(
        self,
        step_sequence: torch.Tensor,
        intervention_targets: List[int],
        step_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.intervention_type == "removal":
            return torch.zeros_like(step_sequence)
        elif self.intervention_type == "noise":
            noise = torch.randn_like(step_sequence) * self.noise_std
            return step_sequence + noise
        else:
            return self.generator(step_sequence)

class CounterfactualOutcomePredictor(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.sequence_encoder = nn.LSTM(
            hidden_size, hidden_size, 
            num_layers=2, batch_first=True, bidirectional=True
        )
        
        self.outcome_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, step_sequence: torch.Tensor, step_mask: torch.Tensor) -> torch.Tensor:
        # Encode sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            step_sequence, 
            step_mask.sum(dim=1).cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        encoded_sequence, (hidden, _) = self.sequence_encoder(packed_input)
        
        # Use final hidden state for outcome prediction
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # Concatenate bidirectional
        
        outcome = self.outcome_predictor(final_hidden).squeeze(-1)
        return outcome

class CausalImportanceEstimator(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )
        
        self.importance_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        self.outcome_encoder = nn.Linear(1, hidden_size)
    
    def forward(
        self,
        step_sequence: torch.Tensor,
        step_mask: torch.Tensor,
        target_outcome: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps, _ = step_sequence.shape
        
        # Encode target outcome
        outcome_encoding = self.outcome_encoder(target_outcome.unsqueeze(-1))
        outcome_encoding = outcome_encoding.unsqueeze(1).expand(-1, num_steps, -1)
        
        # Apply attention to capture dependencies
        attended_sequence, _ = self.attention(
            step_sequence, step_sequence, step_sequence,
            key_padding_mask=~step_mask.bool()
        )
        
        # Combine with outcome information
        combined_features = torch.cat([attended_sequence, outcome_encoding], dim=-1)
        
        # Predict causal importance
        importance_scores = self.importance_predictor(combined_features).squeeze(-1)
        
        # Apply mask
        importance_scores = importance_scores * step_mask
        
        return importance_scores

class NecessityAnalyzer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.necessity_network = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size // 2),  # step + original_outcome + counterfactual_outcome
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        step_sequence: torch.Tensor,
        counterfactual_outcomes: torch.Tensor,
        original_outcomes: torch.Tensor,
        intervention_targets: List[int]
    ) -> torch.Tensor:
        batch_size, num_steps, hidden_size = step_sequence.shape
        necessity_scores = torch.zeros(batch_size, num_steps, device=step_sequence.device)
        
        for target_idx in intervention_targets:
            if target_idx < num_steps:
                step_features = step_sequence[:, target_idx, :]
                
                # Combine step features with outcome information
                combined_input = torch.cat([
                    step_features,
                    original_outcomes.unsqueeze(-1),
                    counterfactual_outcomes.unsqueeze(-1)
                ], dim=-1)
                
                necessity_score = self.necessity_network(combined_input).squeeze(-1)
                necessity_scores[:, target_idx] = necessity_score
        
        return necessity_scores

class SufficiencyAnalyzer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.sufficiency_network = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        step_sequence: torch.Tensor,
        counterfactual_outcomes: torch.Tensor,
        original_outcomes: torch.Tensor,
        intervention_targets: List[int]
    ) -> torch.Tensor:
        batch_size, num_steps, hidden_size = step_sequence.shape
        sufficiency_scores = torch.zeros(batch_size, num_steps, device=step_sequence.device)
        
        for target_idx in intervention_targets:
            if target_idx < num_steps:
                step_features = step_sequence[:, target_idx, :]
                
                combined_input = torch.cat([
                    step_features,
                    original_outcomes.unsqueeze(-1),
                    counterfactual_outcomes.unsqueeze(-1)
                ], dim=-1)
                
                sufficiency_score = self.sufficiency_network(combined_input).squeeze(-1)
                sufficiency_scores[:, target_idx] = sufficiency_score
        
        return sufficiency_scores

class CounterfactualExplanationGenerator:
    def __init__(
        self,
        counterfactual_model: CounterfactualReasoningModel,
        explanation_depth: int = 3,
        min_causal_effect: float = 0.1
    ):
        self.counterfactual_model = counterfactual_model
        self.explanation_depth = explanation_depth
        self.min_causal_effect = min_causal_effect
    
    def generate_explanations(
        self,
        step_sequence: torch.Tensor,
        step_mask: torch.Tensor,
        target_outcome: torch.Tensor,
        step_descriptions: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        with torch.no_grad():
            results = self.counterfactual_model(step_sequence, step_mask, target_outcome)
        
        explanations = {
            "necessity": [],
            "sufficiency": [],
            "counterfactual": []
        }
        
        # Generate necessity explanations
        necessity_scores = results["necessity_scores"][0]  # First batch
        top_necessary_steps = torch.topk(necessity_scores, min(self.explanation_depth, necessity_scores.size(0)))[1]
        
        for step_idx in top_necessary_steps:
            if necessity_scores[step_idx] > 0.5:
                if step_descriptions:
                    explanation = f"Step {step_idx} ({step_descriptions[step_idx]}) is necessary for the outcome."
                else:
                    explanation = f"Step {step_idx} is necessary for the outcome."
                explanations["necessity"].append(explanation)
        
        # Generate sufficiency explanations
        sufficiency_scores = results["sufficiency_scores"][0]
        top_sufficient_steps = torch.topk(sufficiency_scores, min(self.explanation_depth, sufficiency_scores.size(0)))[1]
        
        for step_idx in top_sufficient_steps:
            if sufficiency_scores[step_idx] > 0.5:
                if step_descriptions:
                    explanation = f"Step {step_idx} ({step_descriptions[step_idx]}) is sufficient to significantly influence the outcome."
                else:
                    explanation = f"Step {step_idx} is sufficient to significantly influence the outcome."
                explanations["sufficiency"].append(explanation)
        
        # Generate counterfactual explanations
        removal_effects = results["counterfactual_results"]["removal"]["causal_effects"][0]
        for i, effect in enumerate(removal_effects):
            if abs(effect) > self.min_causal_effect:
                direction = "increase" if effect > 0 else "decrease"
                if step_descriptions:
                    explanation = f"If step {i} ({step_descriptions[i]}) had not occurred, the outcome would {direction} by {abs(effect):.2f}."
                else:
                    explanation = f"If step {i} had not occurred, the outcome would {direction} by {abs(effect):.2f}."
                explanations["counterfactual"].append(explanation)
        
        return explanations

class CausalChainAnalyzer:
    def __init__(self, counterfactual_model: CounterfactualReasoningModel):
        self.counterfactual_model = counterfactual_model
    
    def analyze_causal_chains(
        self,
        step_sequence: torch.Tensor,
        step_mask: torch.Tensor,
        target_outcome: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_steps, _ = step_sequence.shape
        
        # Analyze all possible step combinations up to a certain depth
        causal_chains = {}
        
        for chain_length in range(1, min(4, num_steps + 1)):  # Analyze chains up to length 3
            chain_effects = self._analyze_chains_of_length(
                step_sequence, step_mask, target_outcome, chain_length
            )
            causal_chains[f"length_{chain_length}"] = chain_effects
        
        return causal_chains
    
    def _analyze_chains_of_length(
        self,
        step_sequence: torch.Tensor,
        step_mask: torch.Tensor,
        target_outcome: torch.Tensor,
        chain_length: int
    ) -> Dict[Tuple[int, ...], float]:
        num_steps = step_mask.sum(dim=1)[0].item()
        chain_effects = {}
        
        # Generate all combinations of steps of given length
        for step_combination in itertools.combinations(range(num_steps), chain_length):
            # Create intervention that removes this combination
            modified_sequence = step_sequence.clone()
            for step_idx in step_combination:
                modified_sequence[:, step_idx] = 0  # Remove step
            
            # Predict counterfactual outcome
            with torch.no_grad():
                counterfactual_outcome = self.counterfactual_model.counterfactual_predictor(
                    modified_sequence, step_mask
                )
            
            # Compute causal effect
            causal_effect = target_outcome[0] - counterfactual_outcome[0]
            chain_effects[step_combination] = causal_effect.item()
        
        return chain_effects

class InteractionalCounterfactualAnalyzer:
    def __init__(self, counterfactual_model: CounterfactualReasoningModel):
        self.counterfactual_model = counterfactual_model
    
    def analyze_step_interactions(
        self,
        step_sequence: torch.Tensor,
        step_mask: torch.Tensor,
        target_outcome: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_steps, _ = step_sequence.shape
        interaction_matrix = torch.zeros(num_steps, num_steps, device=step_sequence.device)
        
        # Analyze pairwise interactions
        for i in range(num_steps):
            for j in range(i + 1, num_steps):
                # Effect of removing step i alone
                effect_i = self._compute_removal_effect(step_sequence, step_mask, target_outcome, [i])
                
                # Effect of removing step j alone
                effect_j = self._compute_removal_effect(step_sequence, step_mask, target_outcome, [j])
                
                # Effect of removing both steps
                effect_ij = self._compute_removal_effect(step_sequence, step_mask, target_outcome, [i, j])
                
                # Interaction effect (non-additivity)
                interaction_effect = effect_ij - effect_i - effect_j
                interaction_matrix[i, j] = interaction_effect
                interaction_matrix[j, i] = interaction_effect
        
        return interaction_matrix
    
    def _compute_removal_effect(
        self,
        step_sequence: torch.Tensor,
        step_mask: torch.Tensor,
        target_outcome: torch.Tensor,
        removal_indices: List[int]
    ) -> float:
        modified_sequence = step_sequence.clone()
        for idx in removal_indices:
            modified_sequence[:, idx] = 0
        
        with torch.no_grad():
            counterfactual_outcome = self.counterfactual_model.counterfactual_predictor(
                modified_sequence, step_mask
            )
        
        return (target_outcome[0] - counterfactual_outcome[0]).item()

class CounterfactualConsistencyChecker:
    def __init__(self):
        self.tolerance = 1e-6
    
    def check_consistency(
        self,
        counterfactual_results: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, bool]:
        consistency_checks = {}
        
        # Monotonicity check: removing more steps should not increase beneficial outcome
        consistency_checks["monotonicity"] = self._check_monotonicity(counterfactual_results)
        
        # Symmetry check: intervention order shouldn't matter for independent steps
        consistency_checks["symmetry"] = self._check_symmetry(counterfactual_results)
        
        # Composition check: sequential interventions should be consistent
        consistency_checks["composition"] = self._check_composition(counterfactual_results)
        
        return consistency_checks
    
    def _check_monotonicity(self, results: Dict[str, Dict[str, torch.Tensor]]) -> bool:
        # Simplified monotonicity check
        if "removal" in results:
            causal_effects = results["removal"]["causal_effects"]
            # Check if effects are mostly positive (removing steps hurts performance)
            return (causal_effects >= -self.tolerance).float().mean() > 0.8
        return True
    
    def _check_symmetry(self, results: Dict[str, Dict[str, torch.Tensor]]) -> bool:
        # Placeholder for symmetry check
        return True
    
    def _check_composition(self, results: Dict[str, Dict[str, torch.Tensor]]) -> bool:
        # Placeholder for composition check
        return True

class CounterfactualTrainer:
    def __init__(
        self,
        counterfactual_model: CounterfactualReasoningModel,
        learning_rate: float = 1e-4,
        consistency_weight: float = 0.5,
        causal_regularization_weight: float = 0.1
    ):
        self.counterfactual_model = counterfactual_model
        self.consistency_weight = consistency_weight
        self.causal_regularization_weight = causal_regularization_weight
        
        self.optimizer = torch.optim.AdamW(
            counterfactual_model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        self.counterfactual_model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        results = self.counterfactual_model(
            batch["step_sequence"],
            batch["step_mask"],
            batch["target_outcome"]
        )
        
        # Prediction loss
        predicted_outcomes = results["counterfactual_results"]["removal"]["outcomes"]
        if "counterfactual_outcomes" in batch:
            prediction_loss = F.mse_loss(predicted_outcomes, batch["counterfactual_outcomes"])
        else:
            prediction_loss = torch.tensor(0.0)
        
        # Consistency loss
        consistency_loss = self._compute_consistency_loss(results)
        
        # Causal regularization
        causal_importance = results["causal_importance"]
        sparsity_loss = torch.norm(causal_importance, p=1) / causal_importance.numel()
        
        # Total loss
        total_loss = (
            prediction_loss +
            self.consistency_weight * consistency_loss +
            self.causal_regularization_weight * sparsity_loss
        )
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.counterfactual_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "prediction_loss": prediction_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "sparsity_loss": sparsity_loss.item()
        }
    
    def _compute_consistency_loss(self, results: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Necessity and sufficiency should be somewhat complementary
        necessity_scores = results["necessity_scores"]
        sufficiency_scores = results["sufficiency_scores"]
        
        # High necessity should correlate with high sufficiency for important steps
        consistency_loss = F.mse_loss(
            necessity_scores * sufficiency_scores,
            (necessity_scores + sufficiency_scores) / 2
        )
        
        return consistency_loss