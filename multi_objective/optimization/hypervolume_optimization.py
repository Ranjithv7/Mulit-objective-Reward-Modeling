import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import math
from itertools import combinations

class HypervolumeOptimizer:
    def __init__(
        self,
        num_objectives: int,
        reference_point: torch.Tensor,
        algorithm: str = "wfg",
        exact_threshold: int = 4
    ):
        self.num_objectives = num_objectives
        self.reference_point = reference_point
        self.algorithm = algorithm
        self.exact_threshold = exact_threshold
        
        if algorithm == "wfg":
            self.calculator = WFGHypervolume(num_objectives)
        elif algorithm == "hms":
            self.calculator = HMSHypervolume(num_objectives)
        elif algorithm == "exact":
            self.calculator = ExactHypervolume(num_objectives)
        elif algorithm == "monte_carlo":
            self.calculator = MonteCarloHypervolume(num_objectives)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def compute_hypervolume(self, points: torch.Tensor) -> float:
        if points.size(0) == 0:
            return 0.0
        
        if self.num_objectives <= self.exact_threshold:
            return self.calculator.compute_exact(points, self.reference_point)
        else:
            return self.calculator.compute_approximate(points, self.reference_point)
    
    def compute_hypervolume_contribution(
        self,
        points: torch.Tensor,
        point_index: int
    ) -> float:
        if points.size(0) <= 1:
            return self.compute_hypervolume(points)
        
        full_hv = self.compute_hypervolume(points)
        reduced_points = torch.cat([points[:point_index], points[point_index+1:]], dim=0)
        reduced_hv = self.compute_hypervolume(reduced_points)
        
        return full_hv - reduced_hv
    
    def hypervolume_based_selection(
        self,
        points: torch.Tensor,
        target_size: int
    ) -> torch.Tensor:
        if points.size(0) <= target_size:
            return torch.arange(points.size(0))
        
        selected_indices = []
        remaining_indices = list(range(points.size(0)))
        
        while len(selected_indices) < target_size and remaining_indices:
            if len(remaining_indices) == 1:
                selected_indices.extend(remaining_indices)
                break
            
            best_contribution = -float('inf')
            best_idx = None
            
            for idx in remaining_indices:
                test_indices = selected_indices + [idx]
                test_points = points[test_indices]
                contribution = self.compute_hypervolume_contribution(test_points, len(test_indices) - 1)
                
                if contribution > best_contribution:
                    best_contribution = contribution
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        return torch.tensor(selected_indices[:target_size])

class WFGHypervolume:
    def __init__(self, num_objectives: int):
        self.num_objectives = num_objectives
    
    def compute_exact(self, points: torch.Tensor, reference_point: torch.Tensor) -> float:
        points_np = points.detach().cpu().numpy()
        ref_np = reference_point.detach().cpu().numpy()
        
        return self._wfg_hypervolume(points_np, ref_np)
    
    def compute_approximate(self, points: torch.Tensor, reference_point: torch.Tensor) -> float:
        return self.compute_exact(points, reference_point)
    
    def _wfg_hypervolume(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        if points.shape[0] == 0:
            return 0.0
        
        if points.shape[1] == 1:
            return np.max(reference_point[0] - points[:, 0], initial=0.0)
        
        if points.shape[1] == 2:
            return self._compute_2d_hypervolume(points, reference_point)
        
        return self._wfg_recursive(points, reference_point, points.shape[1] - 1)
    
    def _compute_2d_hypervolume(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]
        
        hypervolume = 0.0
        prev_y = reference_point[1]
        
        for point in sorted_points:
            x, y = point
            if x < reference_point[0] and y < prev_y:
                width = reference_point[0] - x
                height = prev_y - y
                hypervolume += width * height
                prev_y = y
        
        return hypervolume
    
    def _wfg_recursive(self, points: np.ndarray, reference_point: np.ndarray, dim: int) -> float:
        if dim == 0:
            return np.max(reference_point[0] - points[:, 0], initial=0.0)
        
        if dim == 1:
            return self._compute_2d_hypervolume(points[:, [0, dim]], reference_point[[0, dim]])
        
        sorted_indices = np.argsort(points[:, dim])
        sorted_points = points[sorted_indices]
        
        hypervolume = 0.0
        prev_value = reference_point[dim]
        
        for i, point in enumerate(sorted_points):
            if point[dim] < prev_value:
                height = prev_value - point[dim]
                
                dominated_points = sorted_points[:i+1]
                dominated_points = dominated_points[dominated_points[:, dim] >= point[dim]]
                
                if dominated_points.shape[0] > 0:
                    projected_points = dominated_points[:, :dim]
                    projected_ref = reference_point[:dim]
                    
                    base_area = self._wfg_recursive(projected_points, projected_ref, dim - 1)
                    hypervolume += height * base_area
                
                prev_value = point[dim]
        
        return hypervolume

class HMSHypervolume:
    def __init__(self, num_objectives: int):
        self.num_objectives = num_objectives
    
    def compute_exact(self, points: torch.Tensor, reference_point: torch.Tensor) -> float:
        points_np = points.detach().cpu().numpy()
        ref_np = reference_point.detach().cpu().numpy()
        
        return self._hms_hypervolume(points_np, ref_np)
    
    def compute_approximate(self, points: torch.Tensor, reference_point: torch.Tensor) -> float:
        return self.compute_exact(points, reference_point)
    
    def _hms_hypervolume(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        if points.shape[0] == 0:
            return 0.0
        
        non_dominated = self._get_non_dominated_points(points)
        
        if non_dominated.shape[0] == 1:
            point = non_dominated[0]
            if np.all(point < reference_point):
                return np.prod(reference_point - point)
            return 0.0
        
        return self._hms_recursive(non_dominated, reference_point)
    
    def _get_non_dominated_points(self, points: np.ndarray) -> np.ndarray:
        n = points.shape[0]
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i != j and not is_dominated[i]:
                    if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                        is_dominated[i] = True
                        break
        
        return points[~is_dominated]
    
    def _hms_recursive(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        if points.shape[0] == 1:
            point = points[0]
            if np.all(point < reference_point):
                return np.prod(reference_point - point)
            return 0.0
        
        hypervolume = 0.0
        
        for i in range(points.shape[0]):
            point = points[i]
            
            if np.all(point < reference_point):
                remaining_points = np.delete(points, i, axis=0)
                
                exclusive_volume = np.prod(reference_point - point)
                
                if remaining_points.shape[0] > 0:
                    overlapping_points = remaining_points[
                        np.all(remaining_points <= point, axis=1)
                    ]
                    
                    if overlapping_points.shape[0] > 0:
                        overlap_volume = self._hms_recursive(overlapping_points, point)
                        exclusive_volume -= overlap_volume
                
                hypervolume += exclusive_volume
        
        return hypervolume

class ExactHypervolume:
    def __init__(self, num_objectives: int):
        self.num_objectives = num_objectives
    
    def compute_exact(self, points: torch.Tensor, reference_point: torch.Tensor) -> float:
        points_np = points.detach().cpu().numpy()
        ref_np = reference_point.detach().cpu().numpy()
        
        return self._inclusion_exclusion_hypervolume(points_np, ref_np)
    
    def compute_approximate(self, points: torch.Tensor, reference_point: torch.Tensor) -> float:
        return self.compute_exact(points, reference_point)
    
    def _inclusion_exclusion_hypervolume(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        n = points.shape[0]
        
        if n == 0:
            return 0.0
        
        total_volume = 0.0
        
        for subset_size in range(1, n + 1):
            for subset in combinations(range(n), subset_size):
                subset_points = points[list(subset)]
                
                lower_bounds = np.min(subset_points, axis=0)
                
                if np.all(lower_bounds < reference_point):
                    volume = np.prod(reference_point - lower_bounds)
                    
                    if subset_size % 2 == 1:
                        total_volume += volume
                    else:
                        total_volume -= volume
        
        return total_volume

class MonteCarloHypervolume:
    def __init__(self, num_objectives: int, num_samples: int = 100000):
        self.num_objectives = num_objectives
        self.num_samples = num_samples
    
    def compute_exact(self, points: torch.Tensor, reference_point: torch.Tensor) -> float:
        return self.compute_approximate(points, reference_point)
    
    def compute_approximate(self, points: torch.Tensor, reference_point: torch.Tensor) -> float:
        points_np = points.detach().cpu().numpy()
        ref_np = reference_point.detach().cpu().numpy()
        
        if points_np.shape[0] == 0:
            return 0.0
        
        min_bounds = np.min(points_np, axis=0)
        
        total_volume = np.prod(ref_np - min_bounds)
        
        if total_volume <= 0:
            return 0.0
        
        random_samples = np.random.uniform(
            min_bounds,
            ref_np,
            size=(self.num_samples, self.num_objectives)
        )
        
        dominated_count = 0
        
        for sample in random_samples:
            is_dominated = np.any(np.all(points_np <= sample, axis=1))
            if is_dominated:
                dominated_count += 1
        
        return total_volume * (dominated_count / self.num_samples)

class HypervolumeIndicator:
    def __init__(self, reference_point: torch.Tensor):
        self.reference_point = reference_point
        self.num_objectives = reference_point.size(0)
        self.calculator = HypervolumeOptimizer(self.num_objectives, reference_point)
    
    def compute_indicator(self, pareto_front: torch.Tensor) -> float:
        return self.calculator.compute_hypervolume(pareto_front)
    
    def compute_hypervolume_difference(
        self,
        front_a: torch.Tensor,
        front_b: torch.Tensor
    ) -> float:
        hv_a = self.compute_indicator(front_a)
        hv_b = self.compute_indicator(front_b)
        return hv_a - hv_b
    
    def compute_hypervolume_ratio(
        self,
        approximation_front: torch.Tensor,
        true_front: torch.Tensor
    ) -> float:
        hv_approx = self.compute_indicator(approximation_front)
        hv_true = self.compute_indicator(true_front)
        return hv_approx / (hv_true + 1e-8)

class AdaptiveHypervolumeOptimizer:
    def __init__(
        self,
        num_objectives: int,
        initial_reference_point: torch.Tensor,
        adaptation_rate: float = 0.1
    ):
        self.num_objectives = num_objectives
        self.reference_point = initial_reference_point.clone()
        self.adaptation_rate = adaptation_rate
        self.calculator = HypervolumeOptimizer(num_objectives, self.reference_point)
        
        self.history = []
        self.improvement_threshold = 1e-6
    
    def update_reference_point(self, current_front: torch.Tensor):
        if current_front.size(0) == 0:
            return
        
        worst_point = torch.max(current_front, dim=0)[0]
        
        adaptive_ref = self.reference_point * (1 - self.adaptation_rate) + \
                      (worst_point + 1.0) * self.adaptation_rate
        
        self.reference_point = adaptive_ref
        self.calculator.reference_point = self.reference_point
    
    def optimize_step(
        self,
        current_population: torch.Tensor,
        target_size: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self.update_reference_point(current_population)
        
        current_hv = self.calculator.compute_hypervolume(current_population)
        
        selected_indices = self.calculator.hypervolume_based_selection(
            current_population, target_size
        )
        
        selected_population = current_population[selected_indices]
        selected_hv = self.calculator.compute_hypervolume(selected_population)
        
        improvement = selected_hv - (self.history[-1] if self.history else 0)
        self.history.append(selected_hv)
        
        stats = {
            "hypervolume": selected_hv,
            "improvement": improvement,
            "reference_point_norm": torch.norm(self.reference_point).item(),
            "population_size": selected_population.size(0)
        }
        
        return selected_indices, stats

class HypervolumeContributionRanking:
    def __init__(self, optimizer: HypervolumeOptimizer):
        self.optimizer = optimizer
    
    def rank_by_contribution(self, points: torch.Tensor) -> torch.Tensor:
        contributions = []
        
        for i in range(points.size(0)):
            contribution = self.optimizer.compute_hypervolume_contribution(points, i)
            contributions.append(contribution)
        
        contributions = torch.tensor(contributions)
        ranking = torch.argsort(contributions, descending=True)
        
        return ranking
    
    def select_top_contributors(
        self,
        points: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ranking = self.rank_by_contribution(points)
        top_k_indices = ranking[:k]
        contributions = torch.tensor([
            self.optimizer.compute_hypervolume_contribution(points, i)
            for i in top_k_indices
        ])
        
        return top_k_indices, contributions

class MultiObjectiveHypervolumeOptimizer(nn.Module):
    def __init__(
        self,
        reward_models: List[nn.Module],
        reference_point: torch.Tensor,
        population_size: int = 100,
        elite_fraction: float = 0.2
    ):
        super().__init__()
        self.reward_models = nn.ModuleList(reward_models)
        self.num_objectives = len(reward_models)
        self.population_size = population_size
        self.elite_size = int(population_size * elite_fraction)
        
        self.hv_optimizer = AdaptiveHypervolumeOptimizer(
            self.num_objectives,
            reference_point
        )
        
        self.contribution_ranker = HypervolumeContributionRanking(
            self.hv_optimizer.calculator
        )
    
    def evaluate_population(
        self,
        population_inputs: List[torch.Tensor],
        attention_masks: List[torch.Tensor]
    ) -> torch.Tensor:
        objectives = []
        
        for i, model in enumerate(self.reward_models):
            with torch.no_grad():
                rewards = model(population_inputs[i], attention_masks[i])
                if isinstance(rewards, dict):
                    rewards = rewards['rewards']
                objectives.append(rewards.squeeze(-1))
        
        return torch.stack(objectives, dim=-1)
    
    def hypervolume_based_selection(
        self,
        population_objectives: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        elite_indices, stats = self.hv_optimizer.optimize_step(
            population_objectives,
            self.elite_size
        )
        
        remaining_indices = list(range(population_objectives.size(0)))
        for idx in elite_indices:
            remaining_indices.remove(idx.item())
        
        if len(remaining_indices) > 0:
            remaining_objectives = population_objectives[remaining_indices]
            additional_indices = self.hv_optimizer.calculator.hypervolume_based_selection(
                remaining_objectives,
                self.population_size - self.elite_size
            )
            
            additional_indices = torch.tensor([remaining_indices[i] for i in additional_indices])
            selected_indices = torch.cat([elite_indices, additional_indices])
        else:
            selected_indices = elite_indices
        
        return selected_indices, stats
    
    def compute_pareto_front_quality(self, objectives: torch.Tensor) -> Dict[str, float]:
        hv_indicator = HypervolumeIndicator(self.hv_optimizer.reference_point)
        
        non_dominated = self._get_non_dominated_front(objectives)
        hypervolume = hv_indicator.compute_indicator(non_dominated)
        
        if objectives.size(0) > 1:
            crowding_distances = self._compute_crowding_distances(non_dominated)
            diversity = crowding_distances.std().item()
        else:
            diversity = 0.0
        
        return {
            "hypervolume": hypervolume,
            "diversity": diversity,
            "front_size": non_dominated.size(0),
            "convergence": self._compute_convergence_metric(non_dominated)
        }
    
    def _get_non_dominated_front(self, objectives: torch.Tensor) -> torch.Tensor:
        n = objectives.size(0)
        is_dominated = torch.zeros(n, dtype=torch.bool)
        
        for i in range(n):
            for j in range(n):
                if i != j and not is_dominated[i]:
                    if torch.all(objectives[j] <= objectives[i]) and torch.any(objectives[j] < objectives[i]):
                        is_dominated[i] = True
                        break
        
        return objectives[~is_dominated]
    
    def _compute_crowding_distances(self, front: torch.Tensor) -> torch.Tensor:
        n = front.size(0)
        distances = torch.zeros(n)
        
        if n <= 2:
            return torch.full((n,), float('inf'))
        
        for m in range(self.num_objectives):
            sorted_indices = torch.argsort(front[:, m])
            
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            obj_range = front[sorted_indices[-1], m] - front[sorted_indices[0], m]
            
            if obj_range > 0:
                for i in range(1, n - 1):
                    distances[sorted_indices[i]] += (
                        front[sorted_indices[i + 1], m] - front[sorted_indices[i - 1], m]
                    ) / obj_range
        
        return distances
    
    def _compute_convergence_metric(self, front: torch.Tensor) -> float:
        if front.size(0) == 0:
            return float('inf')
        
        distances_to_ref = torch.norm(front - self.hv_optimizer.reference_point, dim=1)
        return distances_to_ref.mean().item()

class HypervolumeBasedNSGA:
    def __init__(
        self,
        num_objectives: int,
        reference_point: torch.Tensor,
        use_contribution_selection: bool = True
    ):
        self.num_objectives = num_objectives
        self.reference_point = reference_point
        self.use_contribution_selection = use_contribution_selection
        
        self.hv_optimizer = HypervolumeOptimizer(num_objectives, reference_point)
    
    def environmental_selection(
        self,
        population: torch.Tensor,
        objectives: torch.Tensor,
        target_size: int
    ) -> torch.Tensor:
        fronts = self._fast_non_dominated_sort(objectives)
        
        selected_indices = []
        
        for front in fronts:
            if len(selected_indices) + len(front) <= target_size:
                selected_indices.extend(front)
            else:
                remaining = target_size - len(selected_indices)
                
                if self.use_contribution_selection and remaining > 0:
                    front_objectives = objectives[front]
                    contributions = []
                    
                    for i, idx in enumerate(front):
                        contribution = self.hv_optimizer.compute_hypervolume_contribution(
                            front_objectives, i
                        )
                        contributions.append((contribution, idx))
                    
                    contributions.sort(reverse=True)
                    selected_indices.extend([idx for _, idx in contributions[:remaining]])
                else:
                    selected_indices.extend(front[:remaining])
                break
        
        return torch.tensor(selected_indices)
    
    def _fast_non_dominated_sort(self, objectives: torch.Tensor) -> List[List[int]]:
        n = objectives.size(0)
        domination_count = torch.zeros(n)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        front_idx = 0
        while len(fronts[front_idx]) > 0:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
            front_idx += 1
        
        return fronts[:-1]
    
    def _dominates(self, obj1: torch.Tensor, obj2: torch.Tensor) -> bool:
        return torch.all(obj1 <= obj2) and torch.any(obj1 < obj2)