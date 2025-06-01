import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import random
import math

class MultiObjectiveEvolutionary:
    def __init__(
        self,
        population_size: int = 100,
        num_objectives: int = 2,
        algorithm: str = "nsga2",
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        tournament_size: int = 2
    ):
        self.population_size = population_size
        self.num_objectives = num_objectives
        self.algorithm = algorithm
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        
        # Algorithm-specific parameters
        if algorithm == "nsga2":
            self.selector = NSGA2Selector()
        elif algorithm == "spea2":
            self.selector = SPEA2Selector()
        elif algorithm == "moead":
            self.selector = MOEADSelector(num_objectives)
        elif algorithm == "sms_emoa":
            self.selector = SMSEMOASelector()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Population and fitness tracking
        self.population = None
        self.fitness_values = None
        self.generation = 0
        
        # Statistics
        self.evolution_history = []
        self.diversity_history = []
        self.convergence_history = []
    
    def optimize(
        self,
        objective_functions: List[Callable],
        parameter_bounds: torch.Tensor,
        num_generations: int = 100,
        neural_network: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run multi-objective evolutionary optimization.
        
        Args:
            objective_functions: List of objective functions to optimize
            parameter_bounds: Bounds for parameters (shape: [num_params, 2])
            num_generations: Number of evolutionary generations
            neural_network: Optional neural network for parameter evolution
            
        Returns:
            Tuple of (final_population, final_fitness_values)
        """
        
        # Initialize population
        self.population = self._initialize_population(parameter_bounds)
        
        for generation in range(num_generations):
            self.generation = generation
            
            # Evaluate fitness
            self.fitness_values = self._evaluate_population(objective_functions)
            
            # Create offspring
            offspring = self._create_offspring(parameter_bounds, neural_network)
            offspring_fitness = self._evaluate_population_subset(objective_functions, offspring)
            
            # Environmental selection
            self.population, self.fitness_values = self.selector.select(
                self.population, self.fitness_values,
                offspring, offspring_fitness,
                self.population_size
            )
            
            # Track statistics
            self._update_statistics()
            
            # Optional: Early stopping based on convergence
            if self._check_convergence():
                break
        
        return self.population, self.fitness_values
    
    def _initialize_population(self, bounds: torch.Tensor) -> torch.Tensor:
        """Initialize random population within bounds."""
        num_params = bounds.size(0)
        population = torch.rand(self.population_size, num_params)
        
        for i in range(num_params):
            population[:, i] = bounds[i, 0] + population[:, i] * (bounds[i, 1] - bounds[i, 0])
        
        return population
    
    def _evaluate_population(self, objective_functions: List[Callable]) -> torch.Tensor:
        """Evaluate entire population on all objectives."""
        fitness = torch.zeros(self.population_size, len(objective_functions))
        
        for i, individual in enumerate(self.population):
            for j, obj_func in enumerate(objective_functions):
                fitness[i, j] = obj_func(individual)
        
        return fitness
    
    def _evaluate_population_subset(
        self, 
        objective_functions: List[Callable], 
        population_subset: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate a subset of population."""
        fitness = torch.zeros(population_subset.size(0), len(objective_functions))
        
        for i, individual in enumerate(population_subset):
            for j, obj_func in enumerate(objective_functions):
                fitness[i, j] = obj_func(individual)
        
        return fitness
    
    def _create_offspring(
        self, 
        bounds: torch.Tensor, 
        neural_network: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """Create offspring through selection, crossover, and mutation."""
        
        offspring = []
        
        for _ in range(self.population_size):
            # Parent selection
            parent1_idx = self._tournament_selection()
            parent2_idx = self._tournament_selection()
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Crossover
            if random.random() < self.crossover_prob:
                if neural_network is not None:
                    child = self._neural_crossover(parent1, parent2, neural_network)
                else:
                    child = self._simulated_binary_crossover(parent1, parent2)
            else:
                child = parent1.clone()
            
            # Mutation
            if random.random() < self.mutation_prob:
                if neural_network is not None:
                    child = self._neural_mutation(child, neural_network, bounds)
                else:
                    child = self._polynomial_mutation(child, bounds)
            
            # Ensure bounds
            child = self._apply_bounds(child, bounds)
            offspring.append(child)
        
        return torch.stack(offspring)
    
    def _tournament_selection(self) -> int:
        """Tournament selection for parent selection."""
        candidates = random.sample(range(self.population_size), self.tournament_size)
        
        # Select best candidate based on dominance
        best_candidate = candidates[0]
        for candidate in candidates[1:]:
            if self._dominates(
                self.fitness_values[candidate], 
                self.fitness_values[best_candidate]
            ):
                best_candidate = candidate
        
        return best_candidate
    
    def _dominates(self, obj1: torch.Tensor, obj2: torch.Tensor) -> bool:
        """Check if obj1 dominates obj2 (minimization)."""
        return torch.all(obj1 <= obj2) and torch.any(obj1 < obj2)
    
    def _simulated_binary_crossover(
        self, 
        parent1: torch.Tensor, 
        parent2: torch.Tensor,
        eta_c: float = 20.0
    ) -> torch.Tensor:
        """Simulated Binary Crossover (SBX)."""
        child = torch.zeros_like(parent1)
        
        for i in range(len(parent1)):
            if random.random() <= 0.5:
                y1, y2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
                
                if abs(y2 - y1) > 1e-14:
                    # SBX crossover
                    rand = random.random()
                    if rand <= 0.5:
                        beta = (2 * rand) ** (1 / (eta_c + 1))
                    else:
                        beta = (1 / (2 * (1 - rand))) ** (1 / (eta_c + 1))
                    
                    child[i] = 0.5 * ((y1 + y2) - beta * abs(y2 - y1))
                else:
                    child[i] = y1
            else:
                child[i] = parent1[i]
        
        return child
    
    def _polynomial_mutation(
        self, 
        individual: torch.Tensor, 
        bounds: torch.Tensor,
        eta_m: float = 20.0
    ) -> torch.Tensor:
        """Polynomial mutation."""
        mutated = individual.clone()
        
        for i in range(len(individual)):
            if random.random() < (1.0 / len(individual)):  # Parameter-wise mutation probability
                y = individual[i]
                delta1 = (y - bounds[i, 0]) / (bounds[i, 1] - bounds[i, 0])
                delta2 = (bounds[i, 1] - y) / (bounds[i, 1] - bounds[i, 0])
                
                rand = random.random()
                mut_pow = 1.0 / (eta_m + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1))
                    deltaq = 1.0 - val ** mut_pow
                
                y = y + deltaq * (bounds[i, 1] - bounds[i, 0])
                mutated[i] = torch.clamp(y, bounds[i, 0], bounds[i, 1])
        
        return mutated
    
    def _neural_crossover(
        self, 
        parent1: torch.Tensor, 
        parent2: torch.Tensor, 
        neural_network: nn.Module
    ) -> torch.Tensor:
        """Neural network-guided crossover."""
        
        # Combine parents as input
        combined_input = torch.cat([parent1, parent2]).unsqueeze(0)
        
        with torch.no_grad():
            # Neural network predicts optimal combination
            crossover_weights = torch.sigmoid(neural_network(combined_input)).squeeze(0)
            
            # Weighted combination
            child = crossover_weights * parent1 + (1 - crossover_weights) * parent2
        
        return child
    
    def _neural_mutation(
        self, 
        individual: torch.Tensor, 
        neural_network: nn.Module, 
        bounds: torch.Tensor
    ) -> torch.Tensor:
        """Neural network-guided mutation."""
        
        with torch.no_grad():
            # Neural network predicts mutation direction and magnitude
            mutation_input = individual.unsqueeze(0)
            mutation_vector = torch.tanh(neural_network(mutation_input)).squeeze(0)
            
            # Apply adaptive mutation
            mutation_strength = 0.1  # Could be learned too
            mutated = individual + mutation_strength * mutation_vector * (bounds[:, 1] - bounds[:, 0])
        
        return mutated
    
    def _apply_bounds(self, individual: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        """Ensure individual is within bounds."""
        for i in range(len(individual)):
            individual[i] = torch.clamp(individual[i], bounds[i, 0], bounds[i, 1])
        return individual
    
    def _update_statistics(self):
        """Update evolution statistics."""
        
        # Diversity metrics
        diversity = self._compute_diversity()
        self.diversity_history.append(diversity)
        
        # Convergence metrics
        convergence = self._compute_convergence()
        self.convergence_history.append(convergence)
        
        # Store current state
        self.evolution_history.append({
            'generation': self.generation,
            'population': self.population.clone(),
            'fitness': self.fitness_values.clone(),
            'diversity': diversity,
            'convergence': convergence
        })
    
    def _compute_diversity(self) -> float:
        """Compute population diversity."""
        if self.population.size(0) < 2:
            return 0.0
        
        # Average pairwise distance in parameter space
        distances = []
        for i in range(self.population.size(0)):
            for j in range(i + 1, self.population.size(0)):
                dist = torch.norm(self.population[i] - self.population[j]).item()
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _compute_convergence(self) -> float:
        """Compute convergence metric."""
        if len(self.evolution_history) < 2:
            return 0.0
        
        # Compare fitness improvement from last generation
        prev_fitness = self.evolution_history[-1]['fitness']
        curr_fitness = self.fitness_values
        
        # Average improvement across all objectives
        improvements = []
        for i in range(curr_fitness.size(0)):
            for j in range(curr_fitness.size(1)):
                if i < prev_fitness.size(0):
                    improvement = prev_fitness[i, j] - curr_fitness[i, j]
                    improvements.append(improvement.item())
        
        return np.mean(improvements) if improvements else 0.0
    
    def _check_convergence(self) -> bool:
        """Check if evolution has converged."""
        if len(self.convergence_history) < 10:
            return False
        
        # Check if improvement has stagnated
        recent_improvements = self.convergence_history[-10:]
        avg_improvement = np.mean(recent_improvements)
        
        return avg_improvement < 1e-6

class NSGA2Selector:
    def select(
        self,
        population: torch.Tensor,
        fitness: torch.Tensor,
        offspring: torch.Tensor,
        offspring_fitness: torch.Tensor,
        target_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """NSGA-II environmental selection."""
        
        # Combine parent and offspring populations
        combined_pop = torch.cat([population, offspring], dim=0)
        combined_fitness = torch.cat([fitness, offspring_fitness], dim=0)
        
        # Non-dominated sorting
        fronts = self._fast_non_dominated_sort(combined_fitness)
        
        # Select individuals
        selected_indices = []
        
        for front in fronts:
            if len(selected_indices) + len(front) <= target_size:
                selected_indices.extend(front)
            else:
                # Calculate crowding distance for remaining slots
                remaining = target_size - len(selected_indices)
                if remaining > 0:
                    crowding_distances = self._crowding_distance(combined_fitness[front])
                    
                    # Sort by crowding distance (descending)
                    sorted_indices = sorted(
                        range(len(front)), 
                        key=lambda i: crowding_distances[i], 
                        reverse=True
                    )
                    
                    selected_indices.extend([front[i] for i in sorted_indices[:remaining]])
                break
        
        return combined_pop[selected_indices], combined_fitness[selected_indices]
    
    def _fast_non_dominated_sort(self, fitness: torch.Tensor) -> List[List[int]]:
        """Fast non-dominated sorting algorithm."""
        n = fitness.size(0)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(fitness[i], fitness[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(fitness[j], fitness[i]):
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
        """Check if obj1 dominates obj2."""
        return torch.all(obj1 <= obj2) and torch.any(obj1 < obj2)
    
    def _crowding_distance(self, fitness: torch.Tensor) -> List[float]:
        """Calculate crowding distance for solutions."""
        n = fitness.size(0)
        distances = [0.0] * n
        
        for m in range(fitness.size(1)):
            # Sort by objective m
            sorted_indices = torch.argsort(fitness[:, m])
            
            # Boundary solutions get infinite distance
            distances[sorted_indices[0].item()] = float('inf')
            distances[sorted_indices[-1].item()] = float('inf')
            
            # Calculate distances for intermediate solutions
            obj_range = fitness[sorted_indices[-1], m] - fitness[sorted_indices[0], m]
            if obj_range > 0:
                for i in range(1, n - 1):
                    idx = sorted_indices[i].item()
                    distances[idx] += (
                        fitness[sorted_indices[i + 1], m] - fitness[sorted_indices[i - 1], m]
                    ).item() / obj_range
        
        return distances

class SPEA2Selector:
    def select(
        self,
        population: torch.Tensor,
        fitness: torch.Tensor,
        offspring: torch.Tensor,
        offspring_fitness: torch.Tensor,
        target_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """SPEA2 environmental selection."""
        
        # Combine populations
        combined_pop = torch.cat([population, offspring], dim=0)
        combined_fitness = torch.cat([fitness, offspring_fitness], dim=0)
        
        # Calculate SPEA2 fitness
        spea2_fitness = self._calculate_spea2_fitness(combined_fitness)
        
        # Environmental selection
        non_dominated = self._get_non_dominated_indices(combined_fitness)
        
        if len(non_dominated) < target_size:
            # Fill with dominated solutions
            dominated = [i for i in range(combined_fitness.size(0)) if i not in non_dominated]
            sorted_dominated = sorted(dominated, key=lambda i: spea2_fitness[i])
            
            selected_indices = non_dominated + sorted_dominated[:target_size - len(non_dominated)]
        elif len(non_dominated) > target_size:
            # Truncate non-dominated set
            selected_indices = self._truncate_non_dominated(
                combined_fitness[non_dominated], non_dominated, target_size
            )
        else:
            selected_indices = non_dominated
        
        return combined_pop[selected_indices], combined_fitness[selected_indices]
    
    def _calculate_spea2_fitness(self, fitness: torch.Tensor) -> List[float]:
        """Calculate SPEA2 fitness values."""
        n = fitness.size(0)
        
        # Raw fitness (number of dominated solutions)
        raw_fitness = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and self._dominates(fitness[i], fitness[j]):
                    raw_fitness[i] += 1
        
        # Density estimation
        density = self._calculate_density(fitness)
        
        # SPEA2 fitness = raw fitness + density
        spea2_fitness = [raw_fitness[i] + density[i] for i in range(n)]
        
        return spea2_fitness
    
    def _calculate_density(self, fitness: torch.Tensor, k: Optional[int] = None) -> List[float]:
        """Calculate density estimation for SPEA2."""
        n = fitness.size(0)
        if k is None:
            k = int(math.sqrt(n))
        
        densities = []
        
        for i in range(n):
            # Calculate distances to all other solutions
            distances = []
            for j in range(n):
                if i != j:
                    dist = torch.norm(fitness[i] - fitness[j]).item()
                    distances.append(dist)
            
            # Sort distances and take k-th nearest neighbor
            distances.sort()
            if len(distances) >= k:
                density = 1.0 / (distances[k - 1] + 2.0)
            else:
                density = 1.0
            
            densities.append(density)
        
        return densities
    
    def _get_non_dominated_indices(self, fitness: torch.Tensor) -> List[int]:
        """Get indices of non-dominated solutions."""
        n = fitness.size(0)
        non_dominated = []
        
        for i in range(n):
            is_dominated = False
            for j in range(n):
                if i != j and self._dominates(fitness[j], fitness[i]):
                    is_dominated = True
                    break
            
            if not is_dominated:
                non_dominated.append(i)
        
        return non_dominated
    
    def _dominates(self, obj1: torch.Tensor, obj2: torch.Tensor) -> bool:
        """Check if obj1 dominates obj2."""
        return torch.all(obj1 <= obj2) and torch.any(obj1 < obj2)
    
    def _truncate_non_dominated(
        self, 
        fitness: torch.Tensor, 
        indices: List[int], 
        target_size: int
    ) -> List[int]:
        """Truncate non-dominated set using clustering."""
        # Simple truncation based on crowding distance
        distances = self._crowding_distance_spea2(fitness)
        
        # Sort by distance (descending) and select top target_size
        sorted_indices = sorted(
            range(len(indices)), 
            key=lambda i: distances[i], 
            reverse=True
        )
        
        return [indices[i] for i in sorted_indices[:target_size]]
    
    def _crowding_distance_spea2(self, fitness: torch.Tensor) -> List[float]:
        """Calculate crowding distance for SPEA2."""
        n = fitness.size(0)
        distances = [0.0] * n
        
        for m in range(fitness.size(1)):
            sorted_indices = torch.argsort(fitness[:, m])
            
            distances[sorted_indices[0].item()] = float('inf')
            distances[sorted_indices[-1].item()] = float('inf')
            
            obj_range = fitness[sorted_indices[-1], m] - fitness[sorted_indices[0], m]
            if obj_range > 0:
                for i in range(1, n - 1):
                    idx = sorted_indices[i].item()
                    distances[idx] += (
                        fitness[sorted_indices[i + 1], m] - fitness[sorted_indices[i - 1], m]
                    ).item() / obj_range
        
        return distances

class MOEADSelector:
    def __init__(self, num_objectives: int, num_neighbors: int = 20):
        self.num_objectives = num_objectives
        self.num_neighbors = num_neighbors
        self.weight_vectors = None
        self.neighborhoods = None
    
    def select(
        self,
        population: torch.Tensor,
        fitness: torch.Tensor,
        offspring: torch.Tensor,
        offspring_fitness: torch.Tensor,
        target_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MOEA/D environmental selection."""
        
        # Initialize weight vectors if needed
        if self.weight_vectors is None:
            self.weight_vectors = self._generate_weight_vectors(target_size)
            self.neighborhoods = self._compute_neighborhoods()
        
        # For simplicity, return NSGA-II selection
        # Full MOEA/D implementation would require more complex decomposition
        nsga2_selector = NSGA2Selector()
        return nsga2_selector.select(population, fitness, offspring, offspring_fitness, target_size)
    
    def _generate_weight_vectors(self, pop_size: int) -> torch.Tensor:
        """Generate weight vectors for decomposition."""
        weights = torch.zeros(pop_size, self.num_objectives)
        
        for i in range(pop_size):
            w = torch.rand(self.num_objectives)
            weights[i] = w / w.sum()
        
        return weights
    
    def _compute_neighborhoods(self) -> List[List[int]]:
        """Compute neighborhood structure."""
        neighborhoods = []
        
        for i in range(self.weight_vectors.size(0)):
            distances = torch.norm(self.weight_vectors - self.weight_vectors[i], dim=1)
            _, neighbor_indices = torch.topk(distances, self.num_neighbors, largest=False)
            neighborhoods.append(neighbor_indices.tolist())
        
        return neighborhoods

class SMSEMOASelector:
    def select(
        self,
        population: torch.Tensor,
        fitness: torch.Tensor,
        offspring: torch.Tensor,
        offspring_fitness: torch.Tensor,
        target_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """SMS-EMOA environmental selection based on hypervolume contribution."""
        
        # Combine populations
        combined_pop = torch.cat([population, offspring], dim=0)
        combined_fitness = torch.cat([fitness, offspring_fitness], dim=0)
        
        # Non-dominated sorting
        nsga2_selector = NSGA2Selector()
        fronts = nsga2_selector._fast_non_dominated_sort(combined_fitness)
        
        selected_indices = []
        
        for front in fronts:
            if len(selected_indices) + len(front) <= target_size:
                selected_indices.extend(front)
            else:
                # Select based on hypervolume contribution
                remaining = target_size - len(selected_indices)
                if remaining > 0:
                    hv_contributions = self._calculate_hypervolume_contributions(
                        combined_fitness[front]
                    )
                    
                    # Sort by hypervolume contribution (descending)
                    sorted_indices = sorted(
                        range(len(front)),
                        key=lambda i: hv_contributions[i],
                        reverse=True
                    )
                    
                    selected_indices.extend([front[i] for i in sorted_indices[:remaining]])
                break
        
        return combined_pop[selected_indices], combined_fitness[selected_indices]
    
    def _calculate_hypervolume_contributions(self, fitness: torch.Tensor) -> List[float]:
        """Calculate hypervolume contributions (simplified 2D case)."""
        n = fitness.size(0)
        contributions = []
        
        # Reference point (assuming minimization)
        ref_point = torch.max(fitness, dim=0)[0] + 1.0
        
        for i in range(n):
            # Calculate hypervolume with and without individual i
            full_set = fitness
            reduced_set = torch.cat([fitness[:i], fitness[i+1:]], dim=0)
            
            full_hv = self._calculate_hypervolume_2d(full_set, ref_point)
            reduced_hv = self._calculate_hypervolume_2d(reduced_set, ref_point)
            
            contributions.append(full_hv - reduced_hv)
        
        return contributions
    
    def _calculate_hypervolume_2d(self, fitness: torch.Tensor, ref_point: torch.Tensor) -> float:
        """Calculate 2D hypervolume (simplified)."""
        if fitness.size(0) == 0:
            return 0.0
        
        if fitness.size(1) != 2:
            # For higher dimensions, use approximation
            volumes = []
            for point in fitness:
                if torch.all(point < ref_point):
                    volume = torch.prod(ref_point - point).item()
                    volumes.append(volume)
            return sum(volumes)
        
        # 2D case
        sorted_indices = torch.argsort(fitness[:, 0])
        sorted_fitness = fitness[sorted_indices]
        
        hypervolume = 0.0
        prev_y = ref_point[1].item()
        
        for point in sorted_fitness:
            x, y = point[0].item(), point[1].item()
            if x < ref_point[0] and y < prev_y:
                width = ref_point[0].item() - x
                height = prev_y - y
                hypervolume += width * height
                prev_y = y
        
        return hypervolume