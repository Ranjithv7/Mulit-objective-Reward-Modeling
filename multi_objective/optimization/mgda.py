import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize

class MGDAOptimizer:
    def __init__(
        self,
        parameters: List[torch.nn.Parameter],
        lr: float = 1e-3,
        mgda_mode: str = "l2",
        normalization: str = "l2",
        eps: float = 1e-8
    ):
        self.parameters = list(parameters)
        self.lr = lr
        self.mgda_mode = mgda_mode  # 'l2', 'cosine', or 'none'
        self.normalization = normalization
        self.eps = eps
        
        # Initialize base optimizer
        self.base_optimizer = torch.optim.Adam(self.parameters, lr=lr)
        
        # Track gradient history for analysis
        self.gradient_history = []
        self.alpha_history = []
    
    def step(self, objectives: List[torch.Tensor], retain_graph: bool = False) -> Dict[str, float]:
        """
        Perform MGDA optimization step.
        
        Args:
            objectives: List of objective tensors to optimize
            retain_graph: Whether to retain computation graph
            
        Returns:
            Dictionary with optimization statistics
        """
        # Compute gradients for each objective
        gradients = []
        
        for i, objective in enumerate(objectives):
            # Zero gradients
            self.base_optimizer.zero_grad()
            
            # Compute gradients
            objective.backward(retain_graph=retain_graph or i < len(objectives) - 1)
            
            # Collect gradients
            grad_vec = self._get_gradient_vector()
            gradients.append(grad_vec)
            
            # Store gradient for analysis
            self.gradient_history.append(grad_vec.clone())
        
        # Stack gradients into matrix
        gradient_matrix = torch.stack(gradients, dim=0)  # Shape: (num_objectives, num_params)
        
        # Solve MGDA optimization problem
        alpha = self._solve_mgda_problem(gradient_matrix)
        
        # Compute consensus gradient
        consensus_gradient = torch.sum(alpha.unsqueeze(1) * gradient_matrix, dim=0)
        
        # Apply consensus gradient
        self._apply_gradient(consensus_gradient)
        
        # Store alpha for analysis
        self.alpha_history.append(alpha.clone())
        
        # Compute statistics
        stats = self._compute_statistics(gradient_matrix, alpha, consensus_gradient)
        
        return stats
    
    def _get_gradient_vector(self) -> torch.Tensor:
        """Extract gradients from parameters and flatten into vector."""
        gradients = []
        
        for param in self.parameters:
            if param.grad is not None:
                gradients.append(param.grad.view(-1))
            else:
                gradients.append(torch.zeros_like(param).view(-1))
        
        return torch.cat(gradients)
    
    def _apply_gradient(self, gradient_vector: torch.Tensor):
        """Apply gradient vector to parameters."""
        start_idx = 0
        
        for param in self.parameters:
            param_size = param.numel()
            param_grad = gradient_vector[start_idx:start_idx + param_size].view(param.shape)
            
            # Apply gradient manually
            with torch.no_grad():
                param -= self.lr * param_grad
            
            start_idx += param_size
    
    def _solve_mgda_problem(self, gradient_matrix: torch.Tensor) -> torch.Tensor:
        """
        Solve MGDA optimization problem to find consensus weights.
        
        Args:
            gradient_matrix: Matrix of gradients (num_objectives x num_params)
            
        Returns:
            Alpha weights for objectives
        """
        num_objectives = gradient_matrix.size(0)
        
        if num_objectives == 1:
            return torch.ones(1, device=gradient_matrix.device)
        
        # Normalize gradients if specified
        if self.normalization == "l2":
            gradient_matrix = F.normalize(gradient_matrix, p=2, dim=1)
        elif self.normalization == "l1":
            gradient_matrix = F.normalize(gradient_matrix, p=1, dim=1)
        
        # Compute Gram matrix
        gram_matrix = torch.mm(gradient_matrix, gradient_matrix.t())
        
        # Solve quadratic programming problem
        if self.mgda_mode == "l2":
            alpha = self._solve_qp_l2(gram_matrix)
        elif self.mgda_mode == "cosine":
            alpha = self._solve_qp_cosine(gradient_matrix)
        else:  # no normalization
            alpha = self._solve_qp_frank_wolfe(gram_matrix)
        
        return alpha
    
    def _solve_qp_l2(self, gram_matrix: torch.Tensor) -> torch.Tensor:
        """Solve QP problem using L2 normalization."""
        num_objectives = gram_matrix.size(0)
        
        # Convert to numpy for scipy optimization
        gram_np = gram_matrix.detach().cpu().numpy()
        
        # Define QP problem: minimize 0.5 * alpha^T * G * alpha
        # subject to: sum(alpha) = 1, alpha >= 0
        
        def objective(alpha):
            return 0.5 * np.dot(alpha, np.dot(gram_np, alpha))
        
        def objective_grad(alpha):
            return np.dot(gram_np, alpha)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda alpha: np.sum(alpha) - 1.0}
        ]
        
        bounds = [(0, None) for _ in range(num_objectives)]
        
        # Initial guess (uniform weights)
        alpha_init = np.ones(num_objectives) / num_objectives
        
        # Solve optimization problem
        result = minimize(
            objective,
            alpha_init,
            method='SLSQP',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        alpha = torch.tensor(result.x, dtype=gram_matrix.dtype, device=gram_matrix.device)
        
        # Ensure non-negative and normalized
        alpha = torch.clamp(alpha, min=0)
        alpha = alpha / (alpha.sum() + self.eps)
        
        return alpha
    
    def _solve_qp_cosine(self, gradient_matrix: torch.Tensor) -> torch.Tensor:
        """Solve QP problem using cosine similarity."""
        num_objectives = gradient_matrix.size(0)
        
        # Compute pairwise cosine similarities
        normalized_grads = F.normalize(gradient_matrix, p=2, dim=1)
        cosine_matrix = torch.mm(normalized_grads, normalized_grads.t())
        
        # Find weights that minimize cosine similarity conflicts
        alpha = torch.ones(num_objectives, device=gradient_matrix.device) / num_objectives
        
        # Simple iterative refinement
        for _ in range(10):
            # Compute conflict scores
            conflicts = torch.sum(cosine_matrix * alpha.unsqueeze(0), dim=1)
            
            # Update weights (reduce weight for high-conflict objectives)
            alpha = alpha * torch.exp(-conflicts)
            alpha = alpha / (alpha.sum() + self.eps)
        
        return alpha
    
    def _solve_qp_frank_wolfe(self, gram_matrix: torch.Tensor) -> torch.Tensor:
        """Solve QP problem using Frank-Wolfe algorithm."""
        num_objectives = gram_matrix.size(0)
        
        # Initialize with uniform weights
        alpha = torch.ones(num_objectives, device=gram_matrix.device) / num_objectives
        
        # Frank-Wolfe iterations
        for iteration in range(50):
            # Compute gradient of objective w.r.t. alpha
            grad = torch.mv(gram_matrix, alpha)
            
            # Find extreme point (solve linear subproblem)
            min_idx = torch.argmin(grad)
            s = torch.zeros_like(alpha)
            s[min_idx] = 1.0
            
            # Line search
            direction = s - alpha
            if torch.norm(direction) < 1e-6:
                break
            
            # Optimal step size for quadratic objective
            numerator = torch.dot(grad, direction)
            denominator = torch.dot(direction, torch.mv(gram_matrix, direction))
            
            if denominator > self.eps:
                step_size = -numerator / denominator
                step_size = torch.clamp(step_size, 0, 1)
            else:
                step_size = 0.1  # Fallback step size
            
            # Update alpha
            alpha = alpha + step_size * direction
        
        return alpha

class AdvancedMGDAOptimizer(MGDAOptimizer):
    def __init__(
        self,
        parameters: List[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        adaptive_lr: bool = True,
        conflict_threshold: float = 0.5
    ):
        super().__init__(parameters, lr)
        
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.adaptive_lr = adaptive_lr
        self.conflict_threshold = conflict_threshold
        
        # Momentum buffers
        self.momentum_buffer = None
        
        # Learning rate adaptation
        self.lr_adaptation_factor = 1.0
        self.conflict_history = []
    
    def step(self, objectives: List[torch.Tensor], retain_graph: bool = False) -> Dict[str, float]:
        """Enhanced MGDA step with momentum and adaptive learning rate."""
        
        # Compute gradients for each objective
        gradients = []
        
        for i, objective in enumerate(objectives):
            self.base_optimizer.zero_grad()
            objective.backward(retain_graph=retain_graph or i < len(objectives) - 1)
            grad_vec = self._get_gradient_vector()
            gradients.append(grad_vec)
        
        gradient_matrix = torch.stack(gradients, dim=0)
        
        # Detect gradient conflicts
        conflict_level = self._compute_conflict_level(gradient_matrix)
        self.conflict_history.append(conflict_level)
        
        # Adapt learning rate based on conflicts
        if self.adaptive_lr:
            self._adapt_learning_rate(conflict_level)
        
        # Solve MGDA problem
        alpha = self._solve_mgda_problem(gradient_matrix)
        
        # Compute consensus gradient
        consensus_gradient = torch.sum(alpha.unsqueeze(1) * gradient_matrix, dim=0)
        
        # Apply weight decay
        if self.weight_decay > 0:
            consensus_gradient = self._apply_weight_decay(consensus_gradient)
        
        # Apply momentum
        if self.momentum > 0:
            consensus_gradient = self._apply_momentum(consensus_gradient)
        
        # Apply gradient with adapted learning rate
        adapted_lr = self.lr * self.lr_adaptation_factor
        self._apply_gradient_with_lr(consensus_gradient, adapted_lr)
        
        # Compute statistics
        stats = self._compute_statistics(gradient_matrix, alpha, consensus_gradient)
        stats['conflict_level'] = conflict_level
        stats['adapted_lr'] = adapted_lr
        
        return stats
    
    def _compute_conflict_level(self, gradient_matrix: torch.Tensor) -> float:
        """Compute level of conflict between gradients."""
        num_objectives = gradient_matrix.size(0)
        
        if num_objectives < 2:
            return 0.0
        
        # Normalize gradients
        normalized_grads = F.normalize(gradient_matrix, p=2, dim=1)
        
        # Compute pairwise cosine similarities
        conflicts = []
        for i in range(num_objectives):
            for j in range(i + 1, num_objectives):
                cosine_sim = torch.dot(normalized_grads[i], normalized_grads[j])
                conflict = max(0, -cosine_sim.item())  # Negative cosine indicates conflict
                conflicts.append(conflict)
        
        return np.mean(conflicts) if conflicts else 0.0
    
    def _adapt_learning_rate(self, conflict_level: float):
        """Adapt learning rate based on gradient conflicts."""
        if conflict_level > self.conflict_threshold:
            # Reduce learning rate when conflicts are high
            self.lr_adaptation_factor *= 0.95
        else:
            # Gradually increase learning rate when conflicts are low
            self.lr_adaptation_factor *= 1.01
        
        # Clamp adaptation factor
        self.lr_adaptation_factor = np.clip(self.lr_adaptation_factor, 0.1, 2.0)
    
    def _apply_weight_decay(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply weight decay to gradient."""
        param_vector = self._get_parameter_vector()
        return gradient + self.weight_decay * param_vector
    
    def _get_parameter_vector(self) -> torch.Tensor:
        """Get flattened parameter vector."""
        params = []
        for param in self.parameters:
            params.append(param.view(-1))
        return torch.cat(params)
    
    def _apply_momentum(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply momentum to gradient."""
        if self.momentum_buffer is None:
            self.momentum_buffer = torch.zeros_like(gradient)
        
        self.momentum_buffer = self.momentum * self.momentum_buffer + gradient
        return self.momentum_buffer
    
    def _apply_gradient_with_lr(self, gradient_vector: torch.Tensor, lr: float):
        """Apply gradient with specified learning rate."""
        start_idx = 0
        
        for param in self.parameters:
            param_size = param.numel()
            param_grad = gradient_vector[start_idx:start_idx + param_size].view(param.shape)
            
            with torch.no_grad():
                param -= lr * param_grad
            
            start_idx += param_size

class HierarchicalMGDAOptimizer(MGDAOptimizer):
    def __init__(
        self,
        parameters: List[torch.nn.Parameter],
        objective_hierarchy: Dict[str, List[int]],
        lr: float = 1e-3
    ):
        super().__init__(parameters, lr)
        self.objective_hierarchy = objective_hierarchy
    
    def step(self, objectives: List[torch.Tensor], retain_graph: bool = False) -> Dict[str, float]:
        """Hierarchical MGDA with objective grouping."""
        
        # Compute gradients
        gradients = []
        for i, objective in enumerate(objectives):
            self.base_optimizer.zero_grad()
            objective.backward(retain_graph=retain_graph or i < len(objectives) - 1)
            grad_vec = self._get_gradient_vector()
            gradients.append(grad_vec)
        
        gradient_matrix = torch.stack(gradients, dim=0)
        
        # Apply hierarchical optimization
        final_alpha = self._hierarchical_optimization(gradient_matrix)
        
        # Compute consensus gradient
        consensus_gradient = torch.sum(final_alpha.unsqueeze(1) * gradient_matrix, dim=0)
        
        # Apply gradient
        self._apply_gradient(consensus_gradient)
        
        # Compute statistics
        stats = self._compute_statistics(gradient_matrix, final_alpha, consensus_gradient)
        
        return stats
    
    def _hierarchical_optimization(self, gradient_matrix: torch.Tensor) -> torch.Tensor:
        """Apply hierarchical MGDA optimization."""
        num_objectives = gradient_matrix.size(0)
        alpha = torch.zeros(num_objectives, device=gradient_matrix.device)
        
        # Process each level of hierarchy
        for level_name, objective_indices in self.objective_hierarchy.items():
            if not objective_indices:
                continue
            
            # Extract gradients for this level
            level_gradients = gradient_matrix[objective_indices]
            
            # Solve MGDA for this level
            level_alpha = self._solve_mgda_problem(level_gradients)
            
            # Assign weights
            for i, obj_idx in enumerate(objective_indices):
                alpha[obj_idx] = level_alpha[i]
        
        # Normalize overall weights
        alpha = alpha / (alpha.sum() + self.eps)
        
        return alpha

class MGDAAnalyzer:
    def __init__(self):
        pass
    
    @staticmethod
    def analyze_gradient_conflicts(
        gradient_history: List[torch.Tensor],
        objective_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Analyze gradient conflicts over training history."""
        
        if len(gradient_history) < 2:
            return {}
        
        # Reshape history: (num_steps, num_objectives, num_params)
        gradients = torch.stack(gradient_history)
        num_steps, num_objectives, num_params = gradients.shape
        
        # Compute average conflict over time
        conflicts = []
        for step in range(num_steps):
            step_gradients = gradients[step]
            normalized_grads = F.normalize(step_gradients, p=2, dim=1)
            
            step_conflicts = []
            for i in range(num_objectives):
                for j in range(i + 1, num_objectives):
                    cosine_sim = torch.dot(normalized_grads[i], normalized_grads[j])
                    conflict = max(0, -cosine_sim.item())
                    step_conflicts.append(conflict)
            
            conflicts.append(np.mean(step_conflicts) if step_conflicts else 0.0)
        
        analysis = {
            "mean_conflict": np.mean(conflicts),
            "max_conflict": np.max(conflicts),
            "conflict_trend": np.polyfit(range(len(conflicts)), conflicts, 1)[0],
            "conflict_variance": np.var(conflicts)
        }
        
        return analysis
    
    @staticmethod
    def compute_pareto_improvement(
        objectives_before: List[torch.Tensor],
        objectives_after: List[torch.Tensor]
    ) -> Dict[str, bool]:
        """Check if MGDA step resulted in Pareto improvement."""
        
        improvements = {}
        
        for i, (before, after) in enumerate(zip(objectives_before, objectives_after)):
            improvements[f"objective_{i}_improved"] = after.item() < before.item()
        
        # Check for Pareto improvement (at least one objective improved, none worsened)
        any_improved = any(improvements.values())
        none_worsened = all(not (after.item() > before.item()) 
                           for before, after in zip(objectives_before, objectives_after))
        
        improvements["pareto_improvement"] = any_improved and none_worsened
        improvements["pareto_optimal"] = any_improved
        
        return improvements

def compute_mgda_statistics(
    gradient_matrix: torch.Tensor,
    alpha: torch.Tensor,
    consensus_gradient: torch.Tensor
) -> Dict[str, float]:
    """Compute comprehensive MGDA statistics."""
    
    num_objectives = gradient_matrix.size(0)
    
    # Gradient norms
    grad_norms = torch.norm(gradient_matrix, dim=1)
    
    # Consensus gradient norm
    consensus_norm = torch.norm(consensus_gradient)
    
    # Alpha statistics
    alpha_entropy = -torch.sum(alpha * torch.log(alpha + 1e-8))
    alpha_max = torch.max(alpha)
    alpha_concentration = 1.0 / (alpha_entropy.item() + 1e-8)
    
    # Gradient alignment
    normalized_grads = F.normalize(gradient_matrix, p=2, dim=1)
    normalized_consensus = F.normalize(consensus_gradient.unsqueeze(0), p=2, dim=1)
    
    alignments = torch.mm(normalized_grads, normalized_consensus.t()).squeeze()
    mean_alignment = torch.mean(alignments)
    min_alignment = torch.min(alignments)
    
    # Conflict measures
    pairwise_conflicts = []
    for i in range(num_objectives):
        for j in range(i + 1, num_objectives):
            cosine_sim = torch.dot(normalized_grads[i], normalized_grads[j])
            conflict = max(0, -cosine_sim.item())
            pairwise_conflicts.append(conflict)
    
    mean_conflict = np.mean(pairwise_conflicts) if pairwise_conflicts else 0.0
    
    return {
        "consensus_gradient_norm": consensus_norm.item(),
        "mean_gradient_norm": torch.mean(grad_norms).item(),
        "alpha_entropy": alpha_entropy.item(),
        "alpha_max": alpha_max.item(),
        "alpha_concentration": alpha_concentration,
        "mean_alignment": mean_alignment.item(),
        "min_alignment": min_alignment.item(),
        "mean_conflict": mean_conflict,
        "effective_objectives": (alpha > 0.01).sum().item()
    }