import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
import networkx as nx

from reward_models.base_reward_model import BaseRewardModel, RewardOutput
from .process_reward_model import ProcessRewardModel

class StepwiseRewardAssigner:
    """Assigns rewards to individual reasoning steps with various attribution methods"""
    
    def __init__(
        self,
        attribution_method: str = "shapley",
        temporal_discount: float = 0.95,
        step_importance_threshold: float = 0.1,
        use_step_dependencies: bool = True
    ):
        self.attribution_method = attribution_method
        self.temporal_discount = temporal_discount
        self.step_importance_threshold = step_importance_threshold
        self.use_step_dependencies = use_step_dependencies
        
        # Attribution methods
        self.attribution_methods = {
            "shapley": self._shapley_attribution,
            "integrated_gradients": self._integrated_gradients_attribution,
            "attention_flow": self._attention_flow_attribution,
            "causal_intervention": self._causal_intervention_attribution,
            "counterfactual": self._counterfactual_attribution,
            "temporal_difference": self._temporal_difference_attribution
        }
    
    def assign_step_rewards(
        self,
        model: ProcessRewardModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: List[List[Tuple[int, int]]],
        final_reward: torch.Tensor
    ) -> List[List[float]]:
        """Assign rewards to individual steps"""
        
        attribution_fn = self.attribution_methods.get(
            self.attribution_method, 
            self._shapley_attribution
        )
        
        return attribution_fn(model, input_ids, attention_mask, step_boundaries, final_reward)
    
    def _shapley_attribution(
        self,
        model: ProcessRewardModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: List[List[Tuple[int, int]]],
        final_reward: torch.Tensor
    ) -> List[List[float]]:
        """Compute Shapley values for step contributions"""
        batch_size = input_ids.size(0)
        batch_step_rewards = []
        
        for batch_idx in range(batch_size):
            steps = step_boundaries[batch_idx]
            if not steps:
                batch_step_rewards.append([])
                continue
            
            step_values = self._compute_shapley_values(
                model, 
                input_ids[batch_idx:batch_idx+1],
                attention_mask[batch_idx:batch_idx+1], 
                steps,
                final_reward[batch_idx]
            )
            
            batch_step_rewards.append(step_values)
        
        return batch_step_rewards
    
    def _compute_shapley_values(
        self,
        model: ProcessRewardModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        steps: List[Tuple[int, int]],
        final_reward: torch.Tensor
    ) -> List[float]:
        """Compute Shapley values for individual steps"""
        n_steps = len(steps)
        if n_steps == 0:
            return []
        
        # Efficient Shapley approximation using sampling
        n_samples = min(2**n_steps, 1000)  # Limit computational cost
        shapley_values = [0.0] * n_steps
        
        # Generate all possible coalitions (or sample them)
        if n_steps <= 10:
            # Exact Shapley for small number of steps
            from itertools import combinations
            
            for step_idx in range(n_steps):
                marginal_contributions = []
                
                # Iterate over all possible coalitions not containing this step
                for r in range(n_steps):
                    for coalition in combinations([i for i in range(n_steps) if i != step_idx], r):
                        # Compute value with coalition
                        value_without = self._compute_coalition_value(
                            model, input_ids, attention_mask, steps, list(coalition)
                        )
                        
                        # Compute value with coalition + step
                        value_with = self._compute_coalition_value(
                            model, input_ids, attention_mask, steps, list(coalition) + [step_idx]
                        )
                        
                        marginal_contrib = value_with - value_without
                        marginal_contributions.append(marginal_contrib)
                
                shapley_values[step_idx] = np.mean(marginal_contributions)
        else:
            # Monte Carlo approximation for large number of steps
            for _ in range(n_samples):
                # Random permutation of steps
                permutation = np.random.permutation(n_steps)
                
                for i, step_idx in enumerate(permutation):
                    # Coalition before this step in permutation
                    coalition_before = permutation[:i].tolist()
                    coalition_with = permutation[:i+1].tolist()
                    
                    value_before = self._compute_coalition_value(
                        model, input_ids, attention_mask, steps, coalition_before
                    )
                    value_with = self._compute_coalition_value(
                        model, input_ids, attention_mask, steps, coalition_with
                    )
                    
                    marginal_contrib = value_with - value_before
                    shapley_values[step_idx] += marginal_contrib / n_samples
        
        return shapley_values
    
    def _compute_coalition_value(
        self,
        model: ProcessRewardModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        steps: List[Tuple[int, int]],
        coalition: List[int]
    ) -> float:
        """Compute the value of a coalition of steps"""
        if not coalition:
            return 0.0
        
        # Create masked input with only coalition steps
        masked_input_ids = input_ids.clone()
        masked_attention_mask = attention_mask.clone()
        
        # Mask out non-coalition steps (set to padding)
        for step_idx in range(len(steps)):
            if step_idx not in coalition:
                start_pos, end_pos = steps[step_idx]
                masked_input_ids[0, start_pos:end_pos+1] = 0  # Padding token
                masked_attention_mask[0, start_pos:end_pos+1] = 0
        
        # Compute reward with masked input
        with torch.no_grad():
            output = model(masked_input_ids, masked_attention_mask, return_dict=True)
            coalition_reward = output.rewards.item()
        
        return coalition_reward
    
    def _integrated_gradients_attribution(
        self,
        model: ProcessRewardModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: List[List[Tuple[int, int]]],
        final_reward: torch.Tensor
    ) -> List[List[float]]:
        """Compute step attributions using Integrated Gradients"""
        batch_step_rewards = []
        
        for batch_idx in range(input_ids.size(0)):
            steps = step_boundaries[batch_idx]
            if not steps:
                batch_step_rewards.append([])
                continue
            
            step_attributions = []
            
            # Get embeddings for gradient computation
            model.train()  # Enable gradients
            
            # Baseline: zero embeddings
            baseline_input = torch.zeros_like(input_ids[batch_idx:batch_idx+1])
            
            # Target: actual input
            target_input = input_ids[batch_idx:batch_idx+1]
            
            # Compute integrated gradients for each step
            for step_idx, (start_pos, end_pos) in enumerate(steps):
                attribution = self._compute_integrated_gradient_for_step(
                    model, baseline_input, target_input, 
                    attention_mask[batch_idx:batch_idx+1],
                    start_pos, end_pos
                )
                step_attributions.append(attribution)
            
            model.eval()
            batch_step_rewards.append(step_attributions)
        
        return batch_step_rewards
    
    def _compute_integrated_gradient_for_step(
        self,
        model: ProcessRewardModel,
        baseline: torch.Tensor,
        target: torch.Tensor,
        attention_mask: torch.Tensor,
        start_pos: int,
        end_pos: int,
        n_steps: int = 50
    ) -> float:
        """Compute integrated gradient for a specific step span"""
        
        # Create interpolated inputs
        alphas = torch.linspace(0, 1, n_steps)
        integrated_grad = 0.0
        
        for alpha in alphas:
            # Interpolate only the step region
            interpolated = baseline.clone().float()
            step_interpolation = baseline[:, start_pos:end_pos+1] + alpha * (
                target[:, start_pos:end_pos+1] - baseline[:, start_pos:end_pos+1]
            )
            interpolated[:, start_pos:end_pos+1] = step_interpolation
            interpolated = interpolated.long()
            
            # Enable gradients
            interpolated.requires_grad_(True)
            
            # Forward pass
            output = model(interpolated, attention_mask, return_dict=True)
            reward = output.rewards.sum()
            
            # Backward pass
            reward.backward()
            
            # Get gradients for this step
            if interpolated.grad is not None:
                step_grad = interpolated.grad[:, start_pos:end_pos+1].sum().item()
                integrated_grad += step_grad
        
        # Average and multiply by step difference
        step_diff = (target[:, start_pos:end_pos+1] - baseline[:, start_pos:end_pos+1]).sum().item()
        attribution = (integrated_grad / n_steps) * step_diff
        
        return attribution
    
    def _attention_flow_attribution(
        self,
        model: ProcessRewardModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: List[List[Tuple[int, int]]],
        final_reward: torch.Tensor
    ) -> List[List[float]]:
        """Compute step attributions using attention flow analysis"""
        batch_step_rewards = []
        
        # Get attention weights from model
        with torch.no_grad():
            outputs = model.backbone(
                input_ids, attention_mask, 
                output_attentions=True, return_dict=True
            )
            attention_weights = outputs.attentions  # List of attention matrices
        
        for batch_idx in range(input_ids.size(0)):
            steps = step_boundaries[batch_idx]
            if not steps:
                batch_step_rewards.append([])
                continue
            
            # Aggregate attention across layers and heads
            aggregated_attention = self._aggregate_attention_weights(
                attention_weights, batch_idx
            )
            
            # Compute attention flow for each step
            step_flows = []
            for start_pos, end_pos in steps:
                # Sum attention weights for tokens in this step
                step_attention = aggregated_attention[start_pos:end_pos+1, :].sum()
                step_flows.append(step_attention.item())
            
            # Normalize by total attention and multiply by final reward
            total_flow = sum(step_flows) if step_flows else 1.0
            normalized_flows = [
                (flow / total_flow) * final_reward[batch_idx].item() 
                for flow in step_flows
            ]
            
            batch_step_rewards.append(normalized_flows)
        
        return batch_step_rewards
    
    def _aggregate_attention_weights(
        self, 
        attention_weights: Tuple[torch.Tensor, ...], 
        batch_idx: int
    ) -> torch.Tensor:
        """Aggregate attention weights across layers and heads"""
        
        # Take last layer attention as most relevant for final decision
        last_layer_attention = attention_weights[-1][batch_idx]  # [heads, seq_len, seq_len]
        
        # Average across heads
        aggregated = last_layer_attention.mean(dim=0)  # [seq_len, seq_len]
        
        return aggregated
    
    def _causal_intervention_attribution(
        self,
        model: ProcessRewardModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: List[List[Tuple[int, int]]],
        final_reward: torch.Tensor
    ) -> List[List[float]]:
        """Compute step attributions using causal interventions"""
        batch_step_rewards = []
        
        for batch_idx in range(input_ids.size(0)):
            steps = step_boundaries[batch_idx]
            if not steps:
                batch_step_rewards.append([])
                continue
            
            baseline_reward = final_reward[batch_idx].item()
            step_contributions = []
            
            # Intervention: remove each step and measure reward change
            for step_idx, (start_pos, end_pos) in enumerate(steps):
                # Create intervened input (mask out this step)
                intervened_input = input_ids[batch_idx:batch_idx+1].clone()
                intervened_mask = attention_mask[batch_idx:batch_idx+1].clone()
                
                # Replace step tokens with neutral tokens or mask
                intervened_input[0, start_pos:end_pos+1] = model.tokenizer.pad_token_id
                intervened_mask[0, start_pos:end_pos+1] = 0
                
                # Compute reward without this step
                with torch.no_grad():
                    output = model(intervened_input, intervened_mask, return_dict=True)
                    intervened_reward = output.rewards.item()
                
                # Step contribution = baseline - intervened
                contribution = baseline_reward - intervened_reward
                step_contributions.append(contribution)
            
            batch_step_rewards.append(step_contributions)
        
        return batch_step_rewards
    
    def _counterfactual_attribution(
        self,
        model: ProcessRewardModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: List[List[Tuple[int, int]]],
        final_reward: torch.Tensor
    ) -> List[List[float]]:
        """Compute counterfactual attributions by step replacement"""
        batch_step_rewards = []
        
        for batch_idx in range(input_ids.size(0)):
            steps = step_boundaries[batch_idx]
            if not steps:
                batch_step_rewards.append([])
                continue
            
            baseline_reward = final_reward[batch_idx].item()
            step_contributions = []
            
            for step_idx, (start_pos, end_pos) in enumerate(steps):
                # Generate counterfactual step (simplified: random replacement)
                counterfactual_input = input_ids[batch_idx:batch_idx+1].clone()
                
                # Replace with random tokens from vocabulary
                step_length = end_pos - start_pos + 1
                random_tokens = torch.randint(
                    1000, 5000, (1, step_length), 
                    device=input_ids.device
                )
                counterfactual_input[0, start_pos:end_pos+1] = random_tokens[0]
                
                # Compute counterfactual reward
                with torch.no_grad():
                    output = model(
                        counterfactual_input, 
                        attention_mask[batch_idx:batch_idx+1], 
                        return_dict=True
                    )
                    counterfactual_reward = output.rewards.item()
                
                # Contribution = original - counterfactual
                contribution = baseline_reward - counterfactual_reward
                step_contributions.append(contribution)
            
            batch_step_rewards.append(step_contributions)
        
        return batch_step_rewards
    
    def _temporal_difference_attribution(
        self,
        model: ProcessRewardModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: List[List[Tuple[int, int]]],
        final_reward: torch.Tensor
    ) -> List[List[float]]:
        """Compute temporal difference-based step attributions"""
        batch_step_rewards = []
        
        for batch_idx in range(input_ids.size(0)):
            steps = step_boundaries[batch_idx]
            if not steps:
                batch_step_rewards.append([])
                continue
            
            # Compute cumulative rewards at each step
            cumulative_rewards = []
            
            for i in range(len(steps) + 1):
                if i == 0:
                    # No steps included
                    cumulative_rewards.append(0.0)
                else:
                    # Include steps 0 to i-1
                    partial_input = input_ids[batch_idx:batch_idx+1].clone()
                    partial_mask = attention_mask[batch_idx:batch_idx+1].clone()
                    
                    # Mask out steps after position i
                    if i < len(steps):
                        mask_start = steps[i][0]
                        partial_input[0, mask_start:] = 0
                        partial_mask[0, mask_start:] = 0
                    
                    with torch.no_grad():
                        output = model(partial_input, partial_mask, return_dict=True)
                        cumulative_reward = output.rewards.item()
                    
                    cumulative_rewards.append(cumulative_reward)
            
            # Compute temporal differences
            step_contributions = []
            for i in range(len(steps)):
                # TD error with discount factor
                if i < len(cumulative_rewards) - 1:
                    td_error = (
                        cumulative_rewards[i + 1] - 
                        self.temporal_discount * cumulative_rewards[i]
                    )
                else:
                    td_error = cumulative_rewards[i]
                
                step_contributions.append(td_error)
            
            batch_step_rewards.append(step_contributions)
        
        return batch_step_rewards

class StepRewardOptimizer:
    """Optimizes step-level reward assignments using various criteria"""
    
    def __init__(
        self,
        optimization_objective: str = "consistency",
        regularization_weight: float = 0.1
    ):
        self.optimization_objective = optimization_objective
        self.regularization_weight = regularization_weight
    
    def optimize_step_rewards(
        self,
        step_attributions: List[List[float]],
        final_rewards: torch.Tensor,
        constraints: Optional[Dict[str, float]] = None
    ) -> List[List[float]]:
        """Optimize step rewards to satisfy various constraints"""
        
        optimized_rewards = []
        
        for batch_idx, (attributions, final_reward) in enumerate(
            zip(step_attributions, final_rewards)
        ):
            if not attributions:
                optimized_rewards.append([])
                continue
            
            # Optimization problem: minimize objective subject to constraints
            result = self._solve_optimization_problem(
                attributions, final_reward.item(), constraints
            )
            
            optimized_rewards.append(result)
        
        return optimized_rewards
    
    def _solve_optimization_problem(
        self,
        initial_rewards: List[float],
        final_reward: float,
        constraints: Optional[Dict[str, float]]
    ) -> List[float]:
        """Solve constrained optimization for step rewards"""
        
        n_steps = len(initial_rewards)
        if n_steps == 0:
            return []
        
        # Objective function
        def objective(x):
            if self.optimization_objective == "consistency":
                # Minimize deviation from initial attributions
                deviation = np.sum((x - np.array(initial_rewards))**2)
                # Regularization: prefer smoother rewards
                smoothness = np.sum(np.diff(x)**2) if len(x) > 1 else 0
                return deviation + self.regularization_weight * smoothness
            
            elif self.optimization_objective == "monotonic":
                # Encourage monotonic increase
                violations = np.sum(np.maximum(0, x[:-1] - x[1:]))
                deviation = np.sum((x - np.array(initial_rewards))**2)
                return deviation + self.regularization_weight * violations
            
            else:
                return np.sum((x - np.array(initial_rewards))**2)
        
        # Constraints
        constraint_list = []
        
        # Sum constraint: step rewards should sum to final reward
        constraint_list.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - final_reward
        })
        
        # Optional additional constraints
        if constraints:
            if 'min_reward' in constraints:
                constraint_list.append({
                    'type': 'ineq', 
                    'fun': lambda x: x - constraints['min_reward']
                })
            
            if 'max_reward' in constraints:
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda x: constraints['max_reward'] - x
                })
        
        # Solve optimization
        initial_guess = np.array(initial_rewards)
        
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            constraints=constraint_list,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return result.x.tolist()
        else:
            # Fallback: normalize to satisfy sum constraint
            normalized = np.array(initial_rewards)
            current_sum = np.sum(normalized)
            if current_sum != 0:
                normalized = normalized * (final_reward / current_sum)
            return normalized.tolist()

class StepDependencyAnalyzer:
    """Analyzes dependencies between reasoning steps"""
    
    def __init__(self):
        self.dependency_types = [
            "causal", "temporal", "logical", "evidential"
        ]
    
    def build_dependency_graph(
        self,
        step_texts: List[str],
        step_embeddings: torch.Tensor
    ) -> nx.DiGraph:
        """Build a directed graph of step dependencies"""
        
        n_steps = len(step_texts)
        G = nx.DiGraph()
        
        # Add nodes
        for i, text in enumerate(step_texts):
            G.add_node(i, text=text, embedding=step_embeddings[i])
        
        # Add edges based on dependencies
        for i in range(n_steps):
            for j in range(i + 1, n_steps):
                dependency_strength = self._compute_dependency_strength(
                    step_texts[i], step_texts[j],
                    step_embeddings[i], step_embeddings[j]
                )
                
                if dependency_strength > 0.3:  # Threshold for significant dependency
                    G.add_edge(i, j, weight=dependency_strength)
        
        return G
    
    def _compute_dependency_strength(
        self,
        text1: str,
        text2: str,
        emb1: torch.Tensor,
        emb2: torch.Tensor
    ) -> float:
        """Compute dependency strength between two steps"""
        
        # Semantic similarity
        semantic_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        
        # Textual indicators
        dependency_indicators = [
            "therefore", "because", "since", "thus", "hence",
            "given that", "it follows", "consequently"
        ]
        
        text2_lower = text2.lower()
        textual_score = sum(1 for indicator in dependency_indicators if indicator in text2_lower)
        textual_score = min(textual_score / len(dependency_indicators), 1.0)
        
        # Combined score
        dependency_strength = 0.7 * semantic_sim + 0.3 * textual_score
        
        return max(0, dependency_strength)
    
    def propagate_rewards_through_dependencies(
        self,
        step_rewards: List[float],
        dependency_graph: nx.DiGraph,
        propagation_rate: float = 0.1
    ) -> List[float]:
        """Propagate rewards through step dependencies"""
        
        if not step_rewards or dependency_graph.number_of_nodes() == 0:
            return step_rewards
        
        propagated_rewards = step_rewards.copy()
        
        # Iterative propagation
        for _ in range(5):  # Fixed number of iterations
            new_rewards = propagated_rewards.copy()
            
            for node in dependency_graph.nodes():
                # Collect rewards from dependencies
                incoming_reward = 0.0
                total_weight = 0.0
                
                for pred in dependency_graph.predecessors(node):
                    edge_weight = dependency_graph[pred][node]['weight']
                    incoming_reward += propagated_rewards[pred] * edge_weight
                    total_weight += edge_weight
                
                # Update reward with propagated component
                if total_weight > 0:
                    propagated_component = propagation_rate * (incoming_reward / total_weight)
                    new_rewards[node] += propagated_component
            
            propagated_rewards = new_rewards
        
        return propagated_rewards