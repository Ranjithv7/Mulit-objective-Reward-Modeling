import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import random

class PCGradOptimizer:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        reduction: str = "mean",
        random_order: bool = True,
        conflict_threshold: float = 0.0
    ):
        """
        PCGrad optimizer that projects conflicting gradients.
        
        Args:
            optimizer: Base optimizer (e.g., Adam, SGD)
            reduction: How to reduce multiple objectives ('mean', 'sum')
            random_order: Whether to randomize gradient projection order
            conflict_threshold: Threshold for considering gradients conflicting
        """
        self.optimizer = optimizer
        self.reduction = reduction
        self.random_order = random_order
        self.conflict_threshold = conflict_threshold
        
        # Track statistics
        self.projection_history = []
        self.conflict_statistics = []
    
    def step(self, objectives: List[torch.Tensor]) -> Dict[str, float]:
        """
        Perform PCGrad optimization step.
        
        Args:
            objectives: List of objective tensors
            
        Returns:
            Dictionary with optimization statistics
        """
        # Compute gradients for each objective
        gradients = self._compute_gradients(objectives)
        
        # Project conflicting gradients
        projected_gradients, stats = self._project_conflicting_gradients(gradients)
        
        # Aggregate projected gradients
        final_gradient = self._aggregate_gradients(projected_gradients)
        
        # Apply final gradient
        self._apply_gradient(final_gradient)
        
        # Update optimizer
        self.optimizer.step()
        
        # Store statistics
        self.projection_history.append(projected_gradients)
        self.conflict_statistics.append(stats)
        
        return stats
    
    def _compute_gradients(self, objectives: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute gradients for each objective."""
        gradients = []
        
        for i, objective in enumerate(objectives):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute gradients
            objective.backward(retain_graph=i < len(objectives) - 1)
            
            # Extract gradient vector
            grad_vec = self._get_gradient_vector()
            gradients.append(grad_vec)
        
        return gradients
    
    def _get_gradient_vector(self) -> torch.Tensor:
        """Extract gradients from parameters and flatten."""
        gradients = []
        
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    gradients.append(param.grad.view(-1))
                else:
                    gradients.append(torch.zeros_like(param).view(-1))
        
        return torch.cat(gradients)
    
    def _project_conflicting_gradients(
        self, 
        gradients: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """
        Project conflicting gradients using PCGrad algorithm.
        
        Args:
            gradients: List of gradient vectors
            
        Returns:
            Tuple of (projected_gradients, statistics)
        """
        num_objectives = len(gradients)
        projected_gradients = [grad.clone() for grad in gradients]
        
        # Statistics tracking
        num_projections = 0
        total_conflicts = 0
        projection_magnitudes = []
        
        # Determine processing order
        if self.random_order:
            indices = list(range(num_objectives))
            random.shuffle(indices)
        else:
            indices = list(range(num_objectives))
        
        # Project gradients sequentially
        for i in indices:
            grad_i = projected_gradients[i]
            
            for j in range(num_objectives):
                if i != j:
                    grad_j = projected_gradients[j]
                    
                    # Check for conflict
                    conflict_score = self._compute_conflict(grad_i, grad_j)
                    
                    if conflict_score > self.conflict_threshold:
                        total_conflicts += 1
                        
                        # Project grad_i onto grad_j
                        projected_grad_i, projection_magnitude = self._project_gradient(grad_i, grad_j)
                        
                        # Update projected gradient
                        projected_gradients[i] = projected_grad_i
                        grad_i = projected_grad_i  # Update for next iteration
                        
                        num_projections += 1
                        projection_magnitudes.append(projection_magnitude)
        
        # Compute statistics
        stats = {
            "num_projections": num_projections,
            "total_conflicts": total_conflicts,
            "avg_projection_magnitude": np.mean(projection_magnitudes) if projection_magnitudes else 0.0,
            "conflict_rate": total_conflicts / (num_objectives * (num_objectives - 1)) if num_objectives > 1 else 0.0
        }
        
        return projected_gradients, stats
    
    def _compute_conflict(self, grad1: torch.Tensor, grad2: torch.Tensor) -> float:
        """Compute conflict score between two gradients."""
        # Normalize gradients
        norm1 = torch.norm(grad1)
        norm2 = torch.norm(grad2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        # Compute cosine similarity
        cosine_sim = torch.dot(grad1, grad2) / (norm1 * norm2)
        
        # Conflict occurs when cosine similarity is negative
        return max(0.0, -cosine_sim.item())
    
    def _project_gradient(
        self, 
        grad_to_project: torch.Tensor, 
        reference_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Project grad_to_project onto the orthogonal complement of reference_grad.
        
        Args:
            grad_to_project: Gradient to be projected
            reference_grad: Reference gradient
            
        Returns:
            Tuple of (projected_gradient, projection_magnitude)
        """
        # Compute projection onto reference gradient
        ref_norm_sq = torch.dot(reference_grad, reference_grad)
        
        if ref_norm_sq < 1e-8:
            return grad_to_project, 0.0
        
        projection_coeff = torch.dot(grad_to_project, reference_grad) / ref_norm_sq
        
        # Only project if there's a conflict (negative dot product)
        if projection_coeff < 0:
            projection = projection_coeff * reference_grad
            projected_grad = grad_to_project - projection
            projection_magnitude = torch.norm(projection).item()
        else:
            projected_grad = grad_to_project
            projection_magnitude = 0.0
        
        return projected_grad, projection_magnitude
    
    def _aggregate_gradients(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate projected gradients."""
        if self.reduction == "mean":
            return torch.stack(gradients).mean(dim=0)
        elif self.reduction == "sum":
            return torch.stack(gradients).sum(dim=0)
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction}")
    
    def _apply_gradient(self, gradient_vector: torch.Tensor):
        """Apply aggregated gradient to parameters."""
        start_idx = 0
        
        for group in self.optimizer.param_groups:
            for param in group['params']:
                param_size = param.numel()
                param_grad = gradient_vector[start_idx:start_idx + param_size].view(param.shape)
                
                # Set gradient for optimizer
                param.grad = param_grad.clone()
                
                start_idx += param_size

class AdaptivePCGradOptimizer(PCGradOptimizer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        adaptation_method: str = "conflict_aware",
        adaptation_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(optimizer, **kwargs)
        self.adaptation_method = adaptation_method
        self.adaptation_rate = adaptation_rate
        
        # Adaptive parameters
        self.objective_weights = None
        self.conflict_history = []
    
    def step(self, objectives: List[torch.Tensor]) -> Dict[str, float]:
        """Adaptive PCGrad step with dynamic objective weighting."""
        
        # Compute gradients
        gradients = self._compute_gradients(objectives)
        
        # Update objective weights based on conflicts
        if self.objective_weights is None:
            self.objective_weights = torch.ones(len(objectives)) / len(objectives)
        
        self._update_objective_weights(gradients)
        
        # Weight gradients before projection
        weighted_gradients = [
            weight * grad for weight, grad in zip(self.objective_weights, gradients)
        ]
        
        # Project conflicting gradients
        projected_gradients, stats = self._project_conflicting_gradients(weighted_gradients)
        
        # Aggregate and apply
        final_gradient = self._aggregate_gradients(projected_gradients)
        self._apply_gradient(final_gradient)
        self.optimizer.step()
        
        # Add weight statistics
        stats.update({
            "objective_weights": self.objective_weights.tolist(),
            "weight_entropy": self._compute_weight_entropy()
        })
        
        return stats
    
    def _update_objective_weights(self, gradients: List[torch.Tensor]):
        """Update objective weights based on adaptation method."""
        
        if self.adaptation_method == "conflict_aware":
            self._conflict_aware_adaptation(gradients)
        elif self.adaptation_method == "magnitude_based":
            self._magnitude_based_adaptation(gradients)
        elif self.adaptation_method == "variance_based":
            self._variance_based_adaptation(gradients)
    
    def _conflict_aware_adaptation(self, gradients: List[torch.Tensor]):
        """Adapt weights based on gradient conflicts."""
        num_objectives = len(gradients)
        conflict_scores = torch.zeros(num_objectives)
        
        # Compute conflict score for each objective
        for i in range(num_objectives):
            total_conflict = 0.0
            for j in range(num_objectives):
                if i != j:
                    conflict = self._compute_conflict(gradients[i], gradients[j])
                    total_conflict += conflict
            
            conflict_scores[i] = total_conflict / (num_objectives - 1) if num_objectives > 1 else 0.0
        
        # Reduce weights for high-conflict objectives
        adaptation = torch.exp(-self.adaptation_rate * conflict_scores)
        self.objective_weights = self.objective_weights * adaptation
        
        # Normalize weights
        self.objective_weights = self.objective_weights / self.objective_weights.sum()
    
    def _magnitude_based_adaptation(self, gradients: List[torch.Tensor]):
        """Adapt weights based on gradient magnitudes."""
        magnitudes = torch.tensor([torch.norm(grad).item() for grad in gradients])
        
        # Increase weights for larger gradients
        adaptation = torch.exp(self.adaptation_rate * magnitudes / magnitudes.max())
        self.objective_weights = self.objective_weights * adaptation
        self.objective_weights = self.objective_weights / self.objective_weights.sum()
    
    def _variance_based_adaptation(self, gradients: List[torch.Tensor]):
        """Adapt weights based on gradient variance over time."""
        if len(self.projection_history) < 10:
            return  # Need sufficient history
        
        # Compute gradient variance over recent history
        recent_gradients = self.projection_history[-10:]
        gradient_vars = []
        
        for obj_idx in range(len(gradients)):
            obj_gradients = [grads[obj_idx] for grads in recent_gradients]
            grad_stack = torch.stack(obj_gradients)
            variance = torch.var(grad_stack, dim=0).mean().item()
            gradient_vars.append(variance)
        
        gradient_vars = torch.tensor(gradient_vars)
        
        # Increase weights for more stable (lower variance) objectives
        adaptation = torch.exp(-self.adaptation_rate * gradient_vars / gradient_vars.max())
        self.objective_weights = self.objective_weights * adaptation
        self.objective_weights = self.objective_weights / self.objective_weights.sum()
    
    def _compute_weight_entropy(self) -> float:
        """Compute entropy of objective weights."""
        return -torch.sum(self.objective_weights * torch.log(self.objective_weights + 1e-8)).item()

class HierarchicalPCGradOptimizer(PCGradOptimizer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        objective_hierarchy: Dict[str, List[int]],
        **kwargs
    ):
        super().__init__(optimizer, **kwargs)
        self.objective_hierarchy = objective_hierarchy
    
    def step(self, objectives: List[torch.Tensor]) -> Dict[str, float]:
        """Hierarchical PCGrad with objective grouping."""
        
        # Compute gradients
        gradients = self._compute_gradients(objectives)
        
        # Apply hierarchical projection
        projected_gradients = self._hierarchical_projection(gradients)
        
        # Aggregate and apply
        final_gradient = self._aggregate_gradients(projected_gradients)
        self._apply_gradient(final_gradient)
        self.optimizer.step()
        
        # Compute statistics
        stats = self._compute_hierarchical_stats(gradients, projected_gradients)
        
        return stats
    
    def _hierarchical_projection(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply hierarchical gradient projection."""
        projected_gradients = [grad.clone() for grad in gradients]
        
        # Process each hierarchy level
        for level_name, objective_indices in self.objective_hierarchy.items():
            if len(objective_indices) < 2:
                continue
            
            # Apply PCGrad within this level
            level_gradients = [projected_gradients[i] for i in objective_indices]
            level_projected, _ = self._project_conflicting_gradients(level_gradients)
            
            # Update projected gradients
            for i, obj_idx in enumerate(objective_indices):
                projected_gradients[obj_idx] = level_projected[i]
        
        return projected_gradients
    
    def _compute_hierarchical_stats(
        self, 
        original_gradients: List[torch.Tensor], 
        projected_gradients: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Compute statistics for hierarchical projection."""
        stats = {}
        
        # Overall statistics
        total_projection_change = 0.0
        for orig, proj in zip(original_gradients, projected_gradients):
            change = torch.norm(orig - proj).item()
            total_projection_change += change
        
        stats["total_projection_change"] = total_projection_change
        
        # Level-specific statistics
        for level_name, objective_indices in self.objective_hierarchy.items():
            if len(objective_indices) < 2:
                continue
            
            level_change = 0.0
            for obj_idx in objective_indices:
                change = torch.norm(
                    original_gradients[obj_idx] - projected_gradients[obj_idx]
                ).item()
                level_change += change
            
            stats[f"{level_name}_projection_change"] = level_change
        
        return stats

class PCGradAnalyzer:
    def __init__(self):
        pass
    
    @staticmethod
    def analyze_projection_effectiveness(
        projection_history: List[List[torch.Tensor]],
        objective_values: List[List[float]]
    ) -> Dict[str, float]:
        """Analyze effectiveness of PCGrad projections."""
        
        if len(projection_history) < 2 or len(objective_values) < 2:
            return {}
        
        # Track objective improvements
        improvements = []
        for i in range(1, len(objective_values)):
            prev_objectives = objective_values[i-1]
            curr_objectives = objective_values[i]
            
            # Count improved objectives
            improved = sum(1 for prev, curr in zip(prev_objectives, curr_objectives) if curr < prev)
            improvements.append(improved / len(prev_objectives))
        
        # Analyze projection patterns
        projection_counts = []
        for projections in projection_history:
            # Count how many gradients were significantly projected
            significant_projections = 0
            for i in range(len(projections)):
                # Compare with original (assuming first is least projected)
                if i > 0:
                    projection_magnitude = torch.norm(projections[0] - projections[i]).item()
                    if projection_magnitude > 0.01:  # Threshold for significant projection
                        significant_projections += 1
            
            projection_counts.append(significant_projections)
        
        analysis = {
            "mean_improvement_rate": np.mean(improvements),
            "improvement_trend": np.polyfit(range(len(improvements)), improvements, 1)[0] if len(improvements) > 1 else 0.0,
            "mean_projections_per_step": np.mean(projection_counts),
            "projection_variance": np.var(projection_counts)
        }
        
        return analysis
    
    @staticmethod
    def compute_gradient_alignment_metrics(
        original_gradients: List[torch.Tensor],
        projected_gradients: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Compute gradient alignment metrics."""
        
        num_objectives = len(original_gradients)
        
        # Original gradient conflicts
        original_conflicts = []
        for i in range(num_objectives):
            for j in range(i + 1, num_objectives):
                norm_i = torch.norm(original_gradients[i])
                norm_j = torch.norm(original_gradients[j])
                
                if norm_i > 1e-8 and norm_j > 1e-8:
                    cosine_sim = torch.dot(original_gradients[i], original_gradients[j]) / (norm_i * norm_j)
                    conflict = max(0.0, -cosine_sim.item())
                    original_conflicts.append(conflict)
        
        # Projected gradient conflicts
        projected_conflicts = []
        for i in range(num_objectives):
            for j in range(i + 1, num_objectives):
                norm_i = torch.norm(projected_gradients[i])
                norm_j = torch.norm(projected_gradients[j])
                
                if norm_i > 1e-8 and norm_j > 1e-8:
                    cosine_sim = torch.dot(projected_gradients[i], projected_gradients[j]) / (norm_i * norm_j)
                    conflict = max(0.0, -cosine_sim.item())
                    projected_conflicts.append(conflict)
        
        # Magnitude preservation
        magnitude_ratios = []
        for orig, proj in zip(original_gradients, projected_gradients):
            orig_norm = torch.norm(orig).item()
            proj_norm = torch.norm(proj).item()
            
            if orig_norm > 1e-8:
                ratio = proj_norm / orig_norm
                magnitude_ratios.append(ratio)
        
        metrics = {
            "original_mean_conflict": np.mean(original_conflicts) if original_conflicts else 0.0,
            "projected_mean_conflict": np.mean(projected_conflicts) if projected_conflicts else 0.0,
            "conflict_reduction": (np.mean(original_conflicts) - np.mean(projected_conflicts)) if original_conflicts and projected_conflicts else 0.0,
            "mean_magnitude_ratio": np.mean(magnitude_ratios) if magnitude_ratios else 1.0,
            "magnitude_variance": np.var(magnitude_ratios) if magnitude_ratios else 0.0
        }
        
        return metrics

def create_pcgrad_optimizer(
    model: nn.Module,
    optimizer_type: str = "adam",
    learning_rate: float = 1e-3,
    pcgrad_kwargs: Optional[Dict] = None
) -> PCGradOptimizer:
    """Factory function to create PCGrad optimizer."""
    
    # Create base optimizer
    if optimizer_type.lower() == "adam":
        base_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == "sgd":
        base_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == "adamw":
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Create PCGrad optimizer
    pcgrad_kwargs = pcgrad_kwargs or {}
    return PCGradOptimizer(base_optimizer, **pcgrad_kwargs)