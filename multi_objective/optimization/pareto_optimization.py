import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import math

class ParetoOptimizer:
    def __init__(
        self,
        num_objectives: int,
        optimization_method: str = "nsga2",
        population_size: int = 100,
        reference_point: Optional[torch.Tensor] = None
    ):
        self.num_objectives = num_objectives
        self.optimization_method = optimization_method
        self.population_size = population_size
        self.reference_point = reference_point
        
        if optimization_method == "nsga2":
            self.optimizer = NSGA2Optimizer(num_objectives, population_size)
        elif optimization_method == "nsga3":
            self.optimizer = NSGA3Optimizer(num_objectives, population_size)
        elif optimization_method == "moead":
            self.optimizer = MOEADOptimizer(num_objectives, population_size)
        elif optimization_method == "hv_emo":
            self.optimizer = HypervolumeEMOOptimizer(num_objectives, population_size, reference_point)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
    
    def optimize(
        self,
        objective_functions: List[callable],
        parameter_bounds: torch.Tensor,
        num_generations: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.optimizer.optimize(objective_functions, parameter_bounds, num_generations)

class NSGA2Optimizer:
    def __init__(self, num_objectives: int, population_size: int):
        self.num_objectives = num_objectives
        self.population_size = population_size
        self.crossover_prob = 0.9
        self.mutation_prob = 0.1
        self.eta_c = 20  # Crossover distribution index
        self.eta_m = 20  # Mutation distribution index
    
    def optimize(
        self,
        objective_functions: List[callable],
        parameter_bounds: torch.Tensor,
        num_generations: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_params = parameter_bounds.size(0)
        
        # Initialize population
        population = self._initialize_population(num_params, parameter_bounds)
        
        for generation in range(num_generations):
            # Evaluate objectives
            objectives = self._evaluate_population(population, objective_functions)
            
            # Create offspring
            offspring = self._create_offspring(population, parameter_bounds)
            offspring_objectives = self._evaluate_population(offspring, objective_functions)
            
            # Combine parent and offspring populations
            combined_pop = torch.cat([population, offspring], dim=0)
            combined_obj = torch.cat([objectives, offspring_objectives], dim=0)
            
            # Non-dominated sorting and selection
            population, objectives = self._nsga2_selection(combined_pop, combined_obj)
        
        return population, objectives
    
    def _initialize_population(self, num_params: int, bounds: torch.Tensor) -> torch.Tensor:
        population = torch.rand(self.population_size, num_params)
        
        for i in range(num_params):
            population[:, i] = bounds[i, 0] + population[:, i] * (bounds[i, 1] - bounds[i, 0])
        
        return population
    
    def _evaluate_population(self, population: torch.Tensor, objective_functions: List[callable]) -> torch.Tensor:
        objectives = torch.zeros(population.size(0), len(objective_functions))
        
        for i, individual in enumerate(population):
            for j, obj_func in enumerate(objective_functions):
                objectives[i, j] = obj_func(individual)
        
        return objectives
    
    def _create_offspring(self, population: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        offspring = torch.zeros_like(population)
        
        for i in range(0, self.population_size, 2):
            # Tournament selection
            parent1_idx = self._tournament_selection(population)
            parent2_idx = self._tournament_selection(population)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            if torch.rand(1) < self.crossover_prob:
                child1, child2 = self._sbx_crossover(parent1, parent2, bounds)
            else:
                child1, child2 = parent1.clone(), parent2.clone()
            
            # Mutation
            child1 = self._polynomial_mutation(child1, bounds)
            child2 = self._polynomial_mutation(child2, bounds)
            
            offspring[i] = child1
            if i + 1 < self.population_size:
                offspring[i + 1] = child2
        
        return offspring
    
    def _tournament_selection(self, population: torch.Tensor, tournament_size: int = 2) -> int:
        candidates = torch.randint(0, population.size(0), (tournament_size,))
        return candidates[torch.randint(0, tournament_size, (1,))].item()
    
    def _sbx_crossover(self, parent1: torch.Tensor, parent2: torch.Tensor, bounds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        for i in range(len(parent1)):
            if torch.rand(1) <= 0.5:
                y1, y2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
                
                if abs(y2 - y1) > 1e-14:
                    beta_q = self._get_beta_q(y1, y2, bounds[i, 0], bounds[i, 1])
                    
                    alpha = 2.0 - beta_q ** (-(self.eta_c + 1))
                    if torch.rand(1) <= 1.0 / alpha:
                        beta_q = (torch.rand(1) * alpha) ** (1.0 / (self.eta_c + 1))
                    else:
                        beta_q = (1.0 / (2.0 - torch.rand(1) * alpha)) ** (1.0 / (self.eta_c + 1))
                    
                    child1[i] = 0.5 * ((y1 + y2) - beta_q * abs(y2 - y1))
                    child2[i] = 0.5 * ((y1 + y2) + beta_q * abs(y2 - y1))
                    
                    child1[i] = torch.clamp(child1[i], bounds[i, 0], bounds[i, 1])
                    child2[i] = torch.clamp(child2[i], bounds[i, 0], bounds[i, 1])
        
        return child1, child2
    
    def _get_beta_q(self, y1: float, y2: float, lower: float, upper: float) -> float:
        beta = 1.0 + 2.0 * min((y1 - lower) / abs(y2 - y1), (upper - y2) / abs(y2 - y1))
        return (2.0 / beta) ** (1.0 / (self.eta_c + 1))
    
    def _polynomial_mutation(self, individual: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        mutated = individual.clone()
        
        for i in range(len(individual)):
            if torch.rand(1) < self.mutation_prob:
                y = individual[i]
                delta1 = (y - bounds[i, 0]) / (bounds[i, 1] - bounds[i, 0])
                delta2 = (bounds[i, 1] - y) / (bounds[i, 1] - bounds[i, 0])
                
                rnd = torch.rand(1)
                mut_pow = 1.0 / (self.eta_m + 1.0)
                
                if rnd <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (self.eta_m + 1))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (self.eta_m + 1))
                    deltaq = 1.0 - val ** mut_pow
                
                y = y + deltaq * (bounds[i, 1] - bounds[i, 0])
                mutated[i] = torch.clamp(y, bounds[i, 0], bounds[i, 1])
        
        return mutated
    
    def _nsga2_selection(self, population: torch.Tensor, objectives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Non-dominated sorting
        fronts = self._fast_non_dominated_sort(objectives)
        
        # Calculate crowding distance for each front
        for front in fronts:
            if len(front) > 2:
                distances = self._crowding_distance(objectives[front])
                front_with_distances = list(zip(front, distances))
                front_with_distances.sort(key=lambda x: x[1], reverse=True)
                fronts[fronts.index(front)] = [x[0] for x in front_with_distances]
        
        # Select individuals for next generation
        selected_indices = []
        for front in fronts:
            if len(selected_indices) + len(front) <= self.population_size:
                selected_indices.extend(front)
            else:
                remaining = self.population_size - len(selected_indices)
                selected_indices.extend(front[:remaining])
                break
        
        return population[selected_indices], objectives[selected_indices]
    
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
            
            if len(next_front) > 0:
                fronts.append(next_front)
            front_idx += 1
        
        return fronts[:-1]  # Remove empty last front
    
    def _dominates(self, obj1: torch.Tensor, obj2: torch.Tensor) -> bool:
        # Assumes minimization problem
        return torch.all(obj1 <= obj2) and torch.any(obj1 < obj2)
    
    def _crowding_distance(self, objectives: torch.Tensor) -> List[float]:
        n = objectives.size(0)
        distances = [0.0] * n
        
        for m in range(objectives.size(1)):
            sorted_indices = torch.argsort(objectives[:, m])
            
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            obj_range = objectives[sorted_indices[-1], m] - objectives[sorted_indices[0], m]
            if obj_range > 0:
                for i in range(1, n - 1):
                    distances[sorted_indices[i]] += (
                        objectives[sorted_indices[i + 1], m] - objectives[sorted_indices[i - 1], m]
                    ).item() / obj_range
        
        return distances

class NSGA3Optimizer(NSGA2Optimizer):
    def __init__(self, num_objectives: int, population_size: int):
        super().__init__(num_objectives, population_size)
        self.reference_points = self._generate_reference_points()
    
    def _generate_reference_points(self) -> torch.Tensor:
        # Generate reference points using Das and Dennis's systematic approach
        if self.num_objectives <= 3:
            divisions = 12
        elif self.num_objectives <= 8:
            divisions = 6
        else:
            divisions = 3
        
        reference_points = self._das_dennis_reference_points(self.num_objectives, divisions)
        return torch.tensor(reference_points, dtype=torch.float32)
    
    def _das_dennis_reference_points(self, m: int, p: int) -> np.ndarray:
        def generate_recursive(ref_points, m, left, total, depth):
            if depth == m - 1:
                ref_points.append([left / total] + [0] * (m - 1))
            else:
                for i in range(left + 1):
                    point = [0] * m
                    point[depth] = i / total
                    generate_recursive(ref_points, m, left - i, total, depth + 1)
        
        ref_points = []
        generate_recursive(ref_points, m, p, p, 0)
        return np.array(ref_points)
    
    def _nsga3_selection(self, population: torch.Tensor, objectives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Non-dominated sorting (same as NSGA-II)
        fronts = self._fast_non_dominated_sort(objectives)
        
        # Select individuals based on reference points
        selected_indices = []
        for front in fronts:
            if len(selected_indices) + len(front) <= self.population_size:
                selected_indices.extend(front)
            else:
                # Use reference point based selection for the last front
                remaining = self.population_size - len(selected_indices)
                front_objectives = objectives[front]
                
                # Normalize objectives
                normalized_obj = self._normalize_objectives(front_objectives)
                
                # Associate with reference points and select
                selected_from_front = self._reference_point_selection(normalized_obj, remaining)
                selected_indices.extend([front[i] for i in selected_from_front])
                break
        
        return population[selected_indices], objectives[selected_indices]
    
    def _normalize_objectives(self, objectives: torch.Tensor) -> torch.Tensor:
        min_vals = torch.min(objectives, dim=0)[0]
        max_vals = torch.max(objectives, dim=0)[0]
        
        normalized = (objectives - min_vals) / (max_vals - min_vals + 1e-10)
        return normalized
    
    def _reference_point_selection(self, objectives: torch.Tensor, k: int) -> List[int]:
        # Simplified reference point association and selection
        n = objectives.size(0)
        distances = torch.zeros(n, self.reference_points.size(0))
        
        for i in range(n):
            for j in range(self.reference_points.size(0)):
                distances[i, j] = torch.norm(objectives[i] - self.reference_points[j])
        
        # Select k individuals with smallest distances to reference points
        min_distances, _ = torch.min(distances, dim=1)
        _, selected_indices = torch.topk(min_distances, k, largest=False)
        
        return selected_indices.tolist()

class MOEADOptimizer:
    def __init__(self, num_objectives: int, population_size: int):
        self.num_objectives = num_objectives
        self.population_size = population_size
        self.neighbor_size = 20
        
        # Generate weight vectors
        self.weight_vectors = self._generate_weight_vectors()
        
        # Initialize neighborhood structure
        self.neighborhoods = self._compute_neighborhoods()
    
    def _generate_weight_vectors(self) -> torch.Tensor:
        # Simple uniform weight vector generation
        weights = torch.zeros(self.population_size, self.num_objectives)
        
        for i in range(self.population_size):
            # Generate random weights and normalize
            w = torch.rand(self.num_objectives)
            weights[i] = w / w.sum()
        
        return weights
    
    def _compute_neighborhoods(self) -> List[List[int]]:
        neighborhoods = []
        
        for i in range(self.population_size):
            # Compute distances to all other weight vectors
            distances = torch.norm(self.weight_vectors - self.weight_vectors[i], dim=1)
            _, neighbor_indices = torch.topk(distances, self.neighbor_size, largest=False)
            neighborhoods.append(neighbor_indices.tolist())
        
        return neighborhoods
    
    def optimize(
        self,
        objective_functions: List[callable],
        parameter_bounds: torch.Tensor,
        num_generations: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initialize population
        population = self._initialize_population(parameter_bounds.size(0), parameter_bounds)
        objectives = self._evaluate_population(population, objective_functions)
        
        # Initialize ideal point
        ideal_point = torch.min(objectives, dim=0)[0]
        
        for generation in range(num_generations):
            for i in range(self.population_size):
                # Generate offspring from neighborhood
                offspring = self._generate_offspring(population, i, parameter_bounds)
                offspring_obj = self._evaluate_individual(offspring, objective_functions)
                
                # Update ideal point
                ideal_point = torch.min(torch.stack([ideal_point, offspring_obj]), dim=0)[0]
                
                # Update neighboring solutions
                for j in self.neighborhoods[i]:
                    if self._tchebycheff(offspring_obj, self.weight_vectors[j], ideal_point) < \
                       self._tchebycheff(objectives[j], self.weight_vectors[j], ideal_point):
                        population[j] = offspring.clone()
                        objectives[j] = offspring_obj.clone()
        
        return population, objectives
    
    def _initialize_population(self, num_params: int, bounds: torch.Tensor) -> torch.Tensor:
        population = torch.rand(self.population_size, num_params)
        
        for i in range(num_params):
            population[:, i] = bounds[i, 0] + population[:, i] * (bounds[i, 1] - bounds[i, 0])
        
        return population
    
    def _evaluate_population(self, population: torch.Tensor, objective_functions: List[callable]) -> torch.Tensor:
        objectives = torch.zeros(population.size(0), len(objective_functions))
        
        for i, individual in enumerate(population):
            objectives[i] = self._evaluate_individual(individual, objective_functions)
        
        return objectives
    
    def _evaluate_individual(self, individual: torch.Tensor, objective_functions: List[callable]) -> torch.Tensor:
        objectives = torch.zeros(len(objective_functions))
        
        for j, obj_func in enumerate(objective_functions):
            objectives[j] = obj_func(individual)
        
        return objectives
    
    def _generate_offspring(self, population: torch.Tensor, parent_idx: int, bounds: torch.Tensor) -> torch.Tensor:
        # Simple differential evolution operator
        neighborhood = self.neighborhoods[parent_idx]
        
        # Select three different individuals from neighborhood
        candidates = torch.tensor(neighborhood)[torch.randperm(len(neighborhood))[:3]]
        
        # DE/rand/1 mutation
        offspring = population[candidates[0]] + 0.5 * (population[candidates[1]] - population[candidates[2]])
        
        # Bound constraints
        for i in range(len(offspring)):
            offspring[i] = torch.clamp(offspring[i], bounds[i, 0], bounds[i, 1])
        
        return offspring
    
    def _tchebycheff(self, objective: torch.Tensor, weight: torch.Tensor, ideal_point: torch.Tensor) -> float:
        # Tchebycheff scalarizing function
        return torch.max(weight * torch.abs(objective - ideal_point)).item()

class HypervolumeEMOOptimizer:
    def __init__(self, num_objectives: int, population_size: int, reference_point: Optional[torch.Tensor] = None):
        self.num_objectives = num_objectives
        self.population_size = population_size
        
        if reference_point is None:
            self.reference_point = torch.ones(num_objectives) * 11.0  # Assume objectives in [0, 10]
        else:
            self.reference_point = reference_point
    
    def optimize(
        self,
        objective_functions: List[callable],
        parameter_bounds: torch.Tensor,
        num_generations: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initialize population
        population = self._initialize_population(parameter_bounds.size(0), parameter_bounds)
        
        for generation in range(num_generations):
            # Evaluate objectives
            objectives = self._evaluate_population(population, objective_functions)
            
            # Create offspring
            offspring = self._create_offspring(population, parameter_bounds)
            offspring_objectives = self._evaluate_population(offspring, objective_functions)
            
            # Combine populations
            combined_pop = torch.cat([population, offspring], dim=0)
            combined_obj = torch.cat([objectives, offspring_objectives], dim=0)
            
            # Hypervolume-based selection
            population, objectives = self._hypervolume_selection(combined_pop, combined_obj)
        
        return population, objectives
    
    def _initialize_population(self, num_params: int, bounds: torch.Tensor) -> torch.Tensor:
        population = torch.rand(self.population_size, num_params)
        
        for i in range(num_params):
            population[:, i] = bounds[i, 0] + population[:, i] * (bounds[i, 1] - bounds[i, 0])
        
        return population
    
    def _evaluate_population(self, population: torch.Tensor, objective_functions: List[callable]) -> torch.Tensor:
        objectives = torch.zeros(population.size(0), len(objective_functions))
        
        for i, individual in enumerate(population):
            for j, obj_func in enumerate(objective_functions):
                objectives[i, j] = obj_func(individual)
        
        return objectives
    
    def _create_offspring(self, population: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        # Simple genetic operators for offspring generation
        offspring = torch.zeros_like(population)
        
        for i in range(self.population_size):
            # Tournament selection
            parent1_idx = torch.randint(0, self.population_size, (1,)).item()
            parent2_idx = torch.randint(0, self.population_size, (1,)).item()
            
            # Uniform crossover
            mask = torch.rand(population.size(1)) < 0.5
            offspring[i] = torch.where(mask, population[parent1_idx], population[parent2_idx])
            
            # Gaussian mutation
            mutation_mask = torch.rand(population.size(1)) < 0.1
            noise = torch.randn(population.size(1)) * 0.1
            offspring[i] = torch.where(mutation_mask, offspring[i] + noise, offspring[i])
            
            # Apply bounds
            for j in range(bounds.size(0)):
                offspring[i, j] = torch.clamp(offspring[i, j], bounds[j, 0], bounds[j, 1])
        
        return offspring
    
    def _hypervolume_selection(self, population: torch.Tensor, objectives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Remove dominated solutions first
        non_dominated_indices = self._get_non_dominated_indices(objectives)
        
        if len(non_dominated_indices) <= self.population_size:
            # If we have fewer non-dominated solutions than needed, keep all
            selected_indices = non_dominated_indices
            
            # Fill remaining slots with best dominated solutions
            if len(selected_indices) < self.population_size:
                remaining_indices = [i for i in range(objectives.size(0)) if i not in selected_indices]
                # Select based on distance from reference point
                remaining_objectives = objectives[remaining_indices]
                distances = torch.norm(remaining_objectives - self.reference_point, dim=1)
                _, sorted_indices = torch.sort(distances)
                
                needed = self.population_size - len(selected_indices)
                selected_indices.extend([remaining_indices[i] for i in sorted_indices[:needed]])
        else:
            # Use hypervolume contribution for selection
            selected_indices = self._hypervolume_contribution_selection(objectives, non_dominated_indices)
        
        return population[selected_indices], objectives[selected_indices]
    
    def _get_non_dominated_indices(self, objectives: torch.Tensor) -> List[int]:
        n = objectives.size(0)
        non_dominated = []
        
        for i in range(n):
            is_dominated = False
            for j in range(n):
                if i != j and self._dominates(objectives[j], objectives[i]):
                    is_dominated = True
                    break
            
            if not is_dominated:
                non_dominated.append(i)
        
        return non_dominated
    
    def _dominates(self, obj1: torch.Tensor, obj2: torch.Tensor) -> bool:
        # Assumes minimization problem
        return torch.all(obj1 <= obj2) and torch.any(obj1 < obj2)
    
    def _hypervolume_contribution_selection(self, objectives: torch.Tensor, candidates: List[int]) -> List[int]:
        # Simplified hypervolume contribution calculation
        # In practice, use more efficient algorithms like WFG or HMS
        
        selected = []
        remaining = candidates.copy()
        
        while len(selected) < self.population_size and remaining:
            if len(remaining) == 1:
                selected.extend(remaining)
                break
            
            best_contribution = -float('inf')
            best_idx = None
            
            for candidate in remaining:
                # Calculate hypervolume contribution
                test_set = selected + [candidate]
                contribution = self._calculate_hypervolume_contribution(objectives[test_set], len(test_set) - 1)
                
                if contribution > best_contribution:
                    best_contribution = contribution
                    best_idx = candidate
            
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return selected
    
    def _calculate_hypervolume_contribution(self, objectives: torch.Tensor, individual_idx: int) -> float:
        # Simplified hypervolume contribution calculation
        # Remove the individual and calculate difference in hypervolume
        
        full_hv = self._calculate_hypervolume(objectives)
        
        if objectives.size(0) <= 1:
            return full_hv
        
        reduced_objectives = torch.cat([objectives[:individual_idx], objectives[individual_idx+1:]], dim=0)
        reduced_hv = self._calculate_hypervolume(reduced_objectives)
        
        return full_hv - reduced_hv
    
    def _calculate_hypervolume(self, objectives: torch.Tensor) -> float:
        # Simplified hypervolume calculation (2D case)
        # For higher dimensions, use proper algorithms like WFG
        
        if objectives.size(0) == 0:
            return 0.0
        
        if objectives.size(1) == 2:
            # 2D hypervolume calculation
            # Sort by first objective
            sorted_indices = torch.argsort(objectives[:, 0])
            sorted_objectives = objectives[sorted_indices]
            
            hypervolume = 0.0
            prev_y = self.reference_point[1].item()
            
            for i in range(sorted_objectives.size(0)):
                x = sorted_objectives[i, 0].item()
                y = sorted_objectives[i, 1].item()
                
                if x < self.reference_point[0] and y < prev_y:
                    width = self.reference_point[0].item() - x
                    height = prev_y - y
                    hypervolume += width * height
                    prev_y = y
            
            return hypervolume
        else:
            # For higher dimensions, use approximation
            volumes = []
            for obj in objectives:
                if torch.all(obj < self.reference_point):
                    volume = torch.prod(self.reference_point - obj).item()
                    volumes.append(volume)
            
            return sum(volumes) if volumes else 0.0